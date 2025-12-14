import os.path
from numpy.random import randint
import torch
from torch.utils import data
import glob
import os
from dataloader.video_transform import *
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import cv2
from PIL import Image
from PIL import ImageDraw
import numpy as np
import json
import random


from torch_geometric.data import Data

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self): # 路径
        return self._data[0]

    @property       # 帧数
    def num_frames(self):
        return int(self._data[1])

    @property       # 标签
    def label(self):
        return int(self._data[2])

class VideoDataset(data.Dataset):
    def __init__(self, list_file,
                    num_segments,
                    duration, mode,
                    transform,
                    image_size,
                    bounding_box_face,
                    bounding_box_body,
                    graph_dir):
        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.bounding_box_face = bounding_box_face
        self.bounding_box_body = bounding_box_body
        self.graph_dir = graph_dir

        self.graph_map = self._build_graph_index()
        self._read_sample()
        self._parse_list()
        self._read_boxs()
        self._read_body_boxes()
        

    def _read_boxs(self):
        with open(self.bounding_box_face, 'r') as f:
            self.boxs = json.load(f)


    
    def _read_body_boxes(self):
        with open(self.bounding_box_body, 'r') as f:
            self.body_boxes = json.load(f)


    def _cv2pil(self,im_cv):
        cv_img_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        pillow_img = Image.fromarray(cv_img_rgb.astype('uint8'))
        return pillow_img

    def _pil2cv(self,im_pil):
        cv_img_rgb = np.array(im_pil)
        cv_img_bgr = cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2BGR)
        return cv_img_bgr

    def _resize_image(self,im, width, height):
        w, h = im.shape[1], im.shape[0]
        r = min(width / w, height / h)
        new_w, new_h = int(w * r), int(h * r)
        im = cv2.resize(im, (new_w, new_h))
        pw = (width - new_w) // 2
        ph = (height - new_h) // 2
        top, bottom = ph, ph
        left, right = pw, pw
        if top + bottom + new_h < height:
            bottom += 1
        if left + right + new_w < width:
            right += 1
        im = cv2.copyMakeBorder(im, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return im, r

    def _face_detect(self,img,box,margin,mode = 'face'):
        if box is None:
            return img
        else:
            left, upper, right, lower = box
            left = int(left)
            upper = int(upper)
            right = int(right)
            lower = int(lower)
            left = max(0, left - margin)
            upper = max(0, upper - margin)
            right = min(img.width, right + margin)
            lower = min(img.height, lower + margin)
            if mode == 'face':
                img = img.crop((left, upper, right, lower))
                return img
            elif mode == 'body':
                occluded_image = img.copy()
                draw = ImageDraw.Draw(occluded_image)
                draw.rectangle([left, upper, right, lower], fill=(0, 0, 0))
                return occluded_image
    
    def _read_sample(self):
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        self.sample_list = [item for item in tmp]


    def _parse_list(self):
        #
        # Data Form: [video_id, num_frames, class_idx]
        #
        self.video_list = [VideoRecord(item) for item in self.sample_list]  
        print(('video number:%d' % (len(self.video_list))))

    def _get_train_indices(self, record):
        # 
        # Split all frames into seg parts, then select frame in each part randomly
        #
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def _get_test_indices(self, record):
        # 
        # Split all frames into seg parts, then select frame in the mid of each part
        #
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def _build_graph_index(self):
      mapping = {}

      print(f"Indexing graph files from {self.graph_dir}...")

      npy_files = glob.glob(os.path.join(self.graph_dir, "*.npy"))

      for f_path in npy_files:
        try:
              # Load nhẹ file để lấy metadata (không tốn nhiều RAM vì chỉ đọc dict)
              data = np.load(f_path, allow_pickle=True).item()
              
              # Lấy key 'original_path' mà bạn đã lưu lúc preprocessing
              orig_path = data.get('original_path') 
              
              if orig_path:
                  # Tạo map: key là đường dẫn video gốc -> value là đường dẫn file npy
                  mapping[orig_path] = f_path
        except Exception as e:
            continue
          
      print(f"Indexed {len(mapping)} graph files.")
      return mapping
        
    def _load_graph(self, record_path, num_required):
        """
        Load file .npy and return list of tuples (x, pos, edge_index, edge_attr)
        """
        npy_path = self.graph_map.get(record_path)

        if npy_path is None:
            print(f"WARNING: Không tìm thấy graph cho video: {record_path}")
            # Trả về graph rỗng để không crash code
            dummy_x = torch.zeros(1, 512, dtype=torch.float)
            dummy_pos = torch.zeros(1, 2, dtype=torch.float)
            dummy_edge_index = torch.empty(2, 0, dtype=torch.long)
            dummy_edge_attr = torch.empty(0, 1, dtype=torch.float)  # Giả sử edge_attr_dim = 1 cho dummy; điều chỉnh nếu cần
            dummy = (dummy_x, dummy_pos, dummy_edge_index, dummy_edge_attr)
            return [dummy] * num_required

        loaded_graphs = []

        attr_dim = None  # Để xác định từ graph đầu tiên

        if npy_path and os.path.exists(npy_path):
            try:
                data_dict = np.load(npy_path, allow_pickle=True).item()
                raw_graphs = data_dict['graphs']

                for G in raw_graphs:
                    if G is None: continue

                    x = torch.tensor(G['x'], dtype=torch.float)
                    # Ensure x is always 2D [num_nodes, 512]
                    if x.dim() == 1:
                        if x.shape[0] == 512:
                            x = x.unsqueeze(0)  # Single node case
                        else:
                            raise ValueError(f"Unexpected shape for x: {x.shape}")
                    elif x.dim() != 2 or x.shape[1] != 512:
                        raise ValueError(f"Invalid node feature shape: {x.shape}; expected [num_nodes, 512]")

                    pos = torch.tensor(G.get('pos', np.zeros((x.shape[0], 2))), dtype=torch.float)
                    # Ensure pos is always 2D [num_nodes, 2]
                    if pos.dim() == 1:
                        if pos.shape[0] == 2:
                            pos = pos.unsqueeze(0)
                        else:
                            raise ValueError(f"Unexpected shape for pos: {pos.shape}")
                    elif pos.dim() != 2 or pos.shape[1] != 2:
                        raise ValueError(f"Invalid pos shape: {pos.shape}; expected [num_nodes, 2]")

                    edge_index = torch.tensor(G['edge_index'], dtype=torch.long)
                    # Ensure edge_index is [2, num_edges]
                    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
                        raise ValueError(f"Invalid edge_index shape: {edge_index.shape}; expected [2, num_edges]")

                    edge_attr = torch.tensor(G['edge_attr'], dtype=torch.float)
                    # Ensure edge_attr is 2D [num_edges, attr_dim] (assuming attr_dim=1 for scalars)
                    if edge_attr.numel() > 0:
                        if edge_attr.dim() == 1:
                            edge_attr = edge_attr.unsqueeze(1)  # [num_edges] -> [num_edges, 1]
                        elif edge_attr.dim() != 2:
                            raise ValueError(f"Invalid edge_attr shape: {edge_attr.shape}; expected [num_edges, attr_dim]")

                    if attr_dim is None:
                        attr_dim = edge_attr.shape[1] if edge_attr.numel() > 0 else 1
                    elif edge_attr.shape[1] != attr_dim and edge_attr.numel() > 0:
                        raise ValueError("Edge_attr dimensions không nhất quán")

                    loaded_graphs.append((x, pos, edge_index, edge_attr))

            except Exception as e:
                print(f"Error loading graph {npy_path}: {e}")
            
            if len(loaded_graphs) == 0:
                dummy_x = torch.zeros(1, 512, dtype=torch.float)
                dummy_pos = torch.zeros(1, 2, dtype=torch.float)
                dummy_edge_index = torch.empty(2, 0, dtype=torch.long)
                dummy_edge_attr = torch.empty(0, attr_dim or 1, dtype=torch.float)
                dummy = (dummy_x, dummy_pos, dummy_edge_index, dummy_edge_attr)
                loaded_graphs = [dummy] * num_required

            if len(loaded_graphs) < num_required:
                loaded_graphs += [loaded_graphs[-1]] * (num_required - len(loaded_graphs))

            return loaded_graphs[:num_required]

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)

        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        path_sq = f"/content/drive/MyDrive/Graph_Classroom/RAER-Education/{record.path}"
        video_frames_path = glob.glob(os.path.join(path_sq, '*'))
        video_frames_path.sort()  

        random_num = random.random()
        images = list()
        images_face = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                img_path = os.path.join(video_frames_path[p])
                parent_dir = os.path.dirname(img_path)
                file_name = os.path.basename(img_path)

                if parent_dir in self.boxs:
                    if file_name in self.boxs[parent_dir]:
                        box = self.boxs[parent_dir][file_name]
                    else:
                        box = None
                else:
                    box = None

                img_pil = Image.open(img_path)
                img_pil_face = Image.open(img_path)
                body_box_path = parent_dir
                body_box = self.body_boxes[body_box_path] if body_box_path in self.body_boxes else None
                if body_box is not None:
                    left, upper, right, lower = body_box
                    img_pil_body = img_pil.crop((left, upper, right, lower))
                else:
                    img_pil_body = img_pil

                img_cv_body = self._pil2cv(img_pil_body)
                img_cv_body, r = self._resize_image(img_cv_body, self.image_size, self.image_size)
                img_pil_body = self._cv2pil(img_cv_body)
                seg_imgs = [img_pil_body]
                

                seg_imgs_face = [self._face_detect(img_pil_face,box,margin=20,mode='face')]

                images.extend(seg_imgs)
                images_face.extend(seg_imgs_face)
                if p < record.num_frames - 1:
                    p += 1

        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))

        images_face = self.transform(images_face)
        images_face = torch.reshape(images_face, (-1, 3, self.image_size, self.image_size))
        graphs = self._load_graph(record.path, self.num_segments)
        return images_face, images, graphs, record.label-1

    def __len__(self):
        return len(self.video_list)


def custom_collate_fn(batch):
    from torch.utils.data import default_collate  # Import ở đây nếu chưa có

    images_face_list, images_list, graphs_list, labels_list = zip(*batch)

    images_face_batch = default_collate(images_face_list)
    images_batch = default_collate(images_list)
    labels_batch = default_collate(labels_list)

    # Batch tất cả graphs thành các tensor riêng biệt
    all_x = []
    all_pos = []
    all_edge_index = []
    all_edge_attr = []
    graph_batch = []  # Gán node thuộc graph nào (global graph id)

    global_graph_id = 0
    node_offset = 0

    for sample_graphs in graphs_list:
        for g in sample_graphs:
            x, pos, edge_index, edge_attr = g
            all_x.append(x)
            all_pos.append(pos)
            all_edge_index.append(edge_index + node_offset)
            all_edge_attr.append(edge_attr)
            num_nodes = x.shape[0]
            graph_batch.append(torch.full((num_nodes,), global_graph_id, dtype=torch.long))
            node_offset += num_nodes
            global_graph_id += 1

    x_batch = torch.cat(all_x, dim=0)
    pos_batch = torch.cat(all_pos, dim=0)
    edge_index_batch = torch.cat(all_edge_index, dim=1)
    edge_attr_batch = torch.cat(all_edge_attr, dim=0) if all(e.numel() > 0 for e in all_edge_attr) else None

    graph_batch = torch.cat(graph_batch, dim=0)

    return images_face_batch, images_batch, x_batch, pos_batch, edge_index_batch, edge_attr_batch, graph_batch, labels_batch

def train_data_loader(list_file, num_segments, duration, image_size,dataset_name,bounding_box_face,bounding_box_body, graph_dir):
    if dataset_name == "RAER":
         train_transforms = torchvision.transforms.Compose([
            RandomRotation(4),
            GroupResize(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor()])
            
    
    train_data = VideoDataset(list_file=list_file,
                              num_segments=num_segments, #16
                              duration=duration, #1
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size,
                              bounding_box_face=bounding_box_face,
                              bounding_box_body=bounding_box_body,
                              graph_dir=graph_dir
                              )
    return train_data


def test_data_loader(list_file, num_segments, duration, image_size,bounding_box_face,bounding_box_body, graph_dir):
    
    test_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                     Stack(),
                                                     ToTorchFormatTensor()])
    
    test_data = VideoDataset(list_file=list_file,
                             num_segments=num_segments,
                             duration=duration,
                             mode='test',
                             transform=test_transform,
                             image_size=image_size,
                             bounding_box_face=bounding_box_face,
                             bounding_box_body=bounding_box_body,
                             graph_dir=graph_dir
                             )
    return test_data
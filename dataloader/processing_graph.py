import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm



# --- CẤU HÌNH MẶC ĐỊNH ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_CONTEXT = 3
IMAGE_SIZE = 224

# --- PHẦN 1: MODEL TRÍCH XUẤT (GIỮ NGUYÊN) ---
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.eval()
        self.to(device)
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_vector(self, img_pil_list):
        if not img_pil_list: return torch.empty(0, 512).to(device)
        batch = torch.stack([self.transform(img) for img in img_pil_list]).to(device)
        with torch.no_grad():
            features = self.backbone(batch)
        return features.squeeze()

# --- PHẦN 2: XỬ LÝ 1 FRAME (GIỮ NGUYÊN) ---
def process_single_frame(img_path, box_face, box_body, yolo_model, feature_extractor):
    # 1. Load ảnh & Chuyển sang Numpy (Fix lỗi YOLO)
    img_pil = Image.open(img_path).convert('RGB')
    img_for_yolo = np.array(img_pil) 
    W, H = img_pil.size
    
    # Main Person Logic
    if box_body:
        l, u, r, b = map(int, box_body)
        l = max(0, l); u = max(0, u); r = min(W, r); b = min(H, b)
        main_crop = img_pil.crop((l, u, r, b))
        cx, cy = (l+r)/2, (u+b)/2
    else:
        main_crop = img_pil
        cx, cy = W/2, H/2
        
    # 2. Detect Context (ALL CLASSES)
    # classes=None nghĩa là lấy tất cả 80 class COCO
    results = yolo_model(img_for_yolo, verbose=False, conf=0.4, classes=None)
    
    context_crops = []
    context_coords = []
    
    for result in results:
        for box in result.boxes:
            
            xyxy = box.xyxy[0].cpu().numpy()
            obj_cx = (xyxy[0] + xyxy[2]) / 2
            obj_cy = (xyxy[1] + xyxy[3]) / 2
            
            # Vẫn giữ logic: Tránh lấy chính người đó làm ngữ cảnh
            if abs(obj_cx - cx) < 20 and abs(obj_cy - cy) < 20:
                continue
            
            l, u, r, b = map(int, xyxy)
            l = max(0, l); u = max(0, u); r = min(W, r); b = min(H, b)
            
            crop = img_pil.crop((l, u, r, b))
            context_crops.append(crop)
            context_coords.append([obj_cx, obj_cy])
            
            # Dừng nếu đã đủ số lượng node tối đa
            if len(context_crops) >= MAX_CONTEXT: break
        if len(context_crops) >= MAX_CONTEXT: break

    # 3. Trích xuất đặc trưng (Phần còn lại giữ nguyên)
    # Lưu ý check ảnh rỗng để tránh lỗi
    if main_crop.size[0] == 0 or main_crop.size[1] == 0: return None

    all_imgs = [main_crop] + context_crops
    all_vectors = feature_extractor.get_vector(all_imgs) # [1+K, 512]
    node_features = all_vectors.cpu().numpy()
    
    main_coord_norm = np.array([cx/W, cy/H])
    
    # ... (Giữ nguyên phần tạo Graph bên dưới) ...
    
    if len(context_crops) == 0:
        return {
            "x": node_features, 
            "pos": np.array([main_coord_norm]),
            "edge_index": np.empty((2, 0), dtype=int),
            "edge_attr": np.empty((0, 1))
        }

    ctx_coords_norm = np.array([[x/W, y/H] for x, y in context_coords])
    all_coords = np.vstack([main_coord_norm, ctx_coords_norm])
    
    sources = np.arange(1, len(all_coords))
    targets = np.zeros_like(sources)
    edge_index = np.vstack([sources, targets])
    
    diff = ctx_coords_norm - main_coord_norm
    dists = np.linalg.norm(diff, axis=1).reshape(-1, 1)
    
    return {"x": node_features, "pos": all_coords, "edge_index": edge_index, "edge_attr": dists}

# --- PHẦN 3: MAIN VỚI ARGPARSE (ĐÃ CẬP NHẬT & SỬA LỖI) ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', type=str, help='Path to train_list.txt')
    parser.add_argument('--test_list', type=str, help='Path to test_list.txt')
    parser.add_argument('--json_file', type=str, required=True, help='Path to body_box.json')
    parser.add_argument('--data_root', type=str, required=True, help='Root folder of images')
    parser.add_argument('--output_dir', type=str, default='./graph_data_npy', help='Output folder')
    args = parser.parse_args()

    tasks = []
    if args.train_list: tasks.append(('train', args.train_list))
    if args.test_list:  tasks.append(('test', args.test_list))
    
    if not tasks:
        print("Bạn chưa cung cấp train_list hoặc test_list!")
        return

    # Tạo thư mục gốc (nếu chưa có)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Files will be saved to: {args.output_dir}")

    print("Loading Models (ResNet + YOLOv8)...")
    cnn_extractor = FeatureExtractor()
    yolo = YOLO("yolov8n.pt")
    
    print(f"Loading JSON: {args.json_file}")
    with open(args.json_file, 'r') as f: body_boxes = json.load(f)

    for split_name, list_file in tasks:
        print(f"\n>>> PROCESSING LIST: {list_file}")
        try:
            video_list = [x.strip().split(' ') for x in open(list_file)]
        except FileNotFoundError:
            print(f"File not found: {list_file}")
            continue
        
        # Duyệt qua từng video
        for item in tqdm(video_list, desc=f"Running {split_name}"):
            rel_path = item[0]
            label = int(item[2])
            
            # --- 1. LOGIC KIỂM TRA TRƯỚC KHI XỬ LÝ (CHECK BEFORE PROCESS) ---
            base_name = os.path.splitext(os.path.basename(rel_path))[0]
            save_name = f"{base_name}.npy"
            final_save_path = os.path.join(args.output_dir, save_name)
            
            should_skip = False
            counter = 0
            
            # Vòng lặp kiểm tra file trùng
            while os.path.exists(final_save_path):
                try:
                    # Load nhẹ file npy để kiểm tra metadata
                    existing_data = np.load(final_save_path, allow_pickle=True).item()
                    
                    # Nếu đường dẫn gốc khớp nhau -> ĐÃ CHẠY RỒI -> SKIP
                    if existing_data.get('original_path') == rel_path:
                        should_skip = True
                        break
                    else:
                        # File trùng tên nhưng KHÁC VIDEO (Collision) -> Tăng số đếm để tìm tên mới
                        counter += 1
                        save_name = f"{base_name}_{counter}.npy"
                        final_save_path = os.path.join(args.output_dir, save_name)
                except Exception:
                    # Nếu file lỗi không đọc được -> Coi như chưa có -> Ghi đè
                    break
            
            if should_skip:
                # tqdm.write(f"Skipping {rel_path} (Already exists)")
                continue

            # --- 2. NẾU KHÔNG SKIP THÌ MỚI BẮT ĐẦU XỬ LÝ ---
            video_path = os.path.join(args.data_root, rel_path)
            frame_files = sorted(glob.glob(os.path.join(video_path, '*')))
            
            if not frame_files: continue

            video_graphs = []
            # Lấy mẫu 16 frame
            indices = np.linspace(0, len(frame_files)-1, 16, dtype=int)
            
            for idx in indices:
                img_path = frame_files[idx]
                parent = os.path.dirname(img_path)
                b_box = body_boxes.get(parent) 
                if b_box is None: b_box = body_boxes.get(os.path.basename(parent))

                graph_data = process_single_frame(img_path, None, b_box, yolo, cnn_extractor)
                
                if graph_data is None:
                     if len(video_graphs) > 0: graph_data = video_graphs[-1]
                     else: continue
                video_graphs.append(graph_data)
            
            # --- 3. LƯU FILE (SỬA LỖI TẠI ĐÂY) ---
            # Kiểm tra và tạo thư mục cha trước khi lưu để tránh FileNotFoundError
            save_dir_path = os.path.dirname(final_save_path)
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path, exist_ok=True)

            np.save(final_save_path, {
                "graphs": video_graphs,
                "label": label,
                "split": split_name,
                "original_path": rel_path 
            })
            
    print(f"\n>>> DONE! All files are inside: {args.output_dir}")

if __name__ == "__main__":
    main()
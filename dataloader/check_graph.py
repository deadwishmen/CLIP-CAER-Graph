import os
import glob
import numpy as np
import argparse
from tqdm import tqdm

def validate_single_graph(graph_data, frame_idx, file_name):
    """
    Kiểm tra logic của một đồ thị đơn lẻ (của 1 frame).
    Trả về: (is_valid, error_message)
    """
    # 1. Lấy dữ liệu
    x = graph_data.get('x')
    edge_index = graph_data.get('edge_index')
    edge_attr = graph_data.get('edge_attr')
    
    # 2. Kiểm tra Node Features (x)
    if x is None or len(x) == 0:
        return False, f"[Frame {frame_idx}] Lỗi: Không có Node Features (x empty)"
    
    num_nodes = x.shape[0]
    feature_dim = x.shape[1]
    
    if np.isnan(x).any() or np.isinf(x).any():
        return False, f"[Frame {frame_idx}] Lỗi: Node features chứa NaN hoặc Inf"

    # 3. Kiểm tra Edges (edge_index)
    if edge_index is None:
        return False, f"[Frame {frame_idx}] Lỗi: edge_index is None"
    
    # Nếu không có cạnh nào (Graph rời rạc hoàn toàn) -> Có thể chấp nhận được tùy bài toán, 
    # nhưng cần kiểm tra tính nhất quán
    num_edges = edge_index.shape[1]
    if num_edges == 0:
        # Nếu không có cạnh, thì edge_attr cũng phải rỗng
        if edge_attr is not None and len(edge_attr) > 0:
             return False, f"[Frame {frame_idx}] Lỗi: Không có cạnh (edge_index rỗng) nhưng lại có edge_attr"
        return True, "OK (No edges)"

    # Kiểm tra kích thước edge_index phải là (2, E)
    if edge_index.shape[0] != 2:
        return False, f"[Frame {frame_idx}] Lỗi: edge_index có shape {edge_index.shape}, kỳ vọng (2, Num_Edges)"

    # 4. [QUAN TRỌNG] Kiểm tra tính hợp lệ của chỉ số cạnh (Bounds Check)
    max_idx = edge_index.max()
    min_idx = edge_index.min()
    
    if min_idx < 0:
        return False, f"[Frame {frame_idx}] Lỗi: edge_index chứa giá trị âm ({min_idx})"
    
    if max_idx >= num_nodes:
        return False, f"[Frame {frame_idx}] Lỗi: edge_index trỏ tới node {max_idx}, nhưng chỉ có {num_nodes} nodes"

    # 5. Kiểm tra Edge Attributes
    if edge_attr is not None:
        if edge_attr.shape[0] != num_edges:
            return False, f"[Frame {frame_idx}] Lỗi: Mismatch! Có {num_edges} cạnh nhưng edge_attr có {edge_attr.shape[0]} phần tử"
        
        if np.isnan(edge_attr).any() or np.isinf(edge_attr).any():
            return False, f"[Frame {frame_idx}] Lỗi: edge_attr chứa NaN hoặc Inf"

    return True, "OK"

def inspect_npy_file(file_path):
    """
    Load file .npy và kiểm tra cấu trúc bên trong.
    """
    try:
        data = np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        return False, f"Không thể load file (Corrupted): {e}"

    # Kiểm tra Keys cơ bản
    required_keys = ['graphs', 'label', 'original_path']
    for k in required_keys:
        if k not in data:
            return False, f"Thiếu key quan trọng: {k}"

    graphs_list = data['graphs']
    if not isinstance(graphs_list, list) or len(graphs_list) == 0:
        return False, "List 'graphs' bị rỗng hoặc không phải list"

    # Duyệt qua từng frame để kiểm tra
    for idx, g in enumerate(graphs_list):
        is_valid, msg = validate_single_graph(g, idx, os.path.basename(file_path))
        if not is_valid:
            return False, msg

    return True, f"OK. ({len(graphs_list)} frames, {data['graphs'][0]['x'].shape[0]} nodes/frame)"

def main():
    parser = argparse.ArgumentParser(description="Tool kiểm tra lỗi Graph Data (.npy)")
    parser.add_argument('--input_dir', type=str, required=True, help='Thư mục chứa file .npy (ví dụ: ./graph_data_npy)')
    args = parser.parse_args()

    npy_files = glob.glob(os.path.join(args.input_dir, "*.npy"))
    
    if len(npy_files) == 0:
        print(f"Không tìm thấy file .npy nào trong: {args.input_dir}")
        return

    print(f"Tìm thấy {len(npy_files)} files. Bắt đầu kiểm tra...\n")
    
    error_count = 0
    valid_count = 0
    
    # Progress bar
    pbar = tqdm(npy_files)
    
    errors_log = []

    for file_path in pbar:
        file_name = os.path.basename(file_path)
        pbar.set_description(f"Checking {file_name}")
        
        is_valid, msg = inspect_npy_file(file_path)
        
        if is_valid:
            valid_count += 1
        else:
            error_count += 1
            errors_log.append(f"❌ {file_name}: {msg}")

    print("\n" + "="*40)
    print("KẾT QUẢ KIỂM TRA")
    print("="*40)
    print(f"✅ Số file hợp lệ: {valid_count}")
    print(f"❌ Số file lỗi:    {error_count}")
    
    if error_count > 0:
        print("\nCHI TIẾT LỖI:")
        for log in errors_log:
            print(log)
        print("\n[KHUYẾN NGHỊ] Bạn nên xóa hoặc tạo lại các file bị lỗi trên.")
    else:
        print("\nTuyệt vời! Dữ liệu Graph của bạn có vẻ ổn định.")

if __name__ == "__main__":
    main()
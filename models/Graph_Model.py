import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

class GraphContextEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=256, edge_dim=1):
        super(GraphContextEncoder, self).__init__()
        
        # 1. Xử lý Edge Features (Quan trọng!)
        # Biến đổi edge scalar (khoảng cách) thành vector để GAT dễ học hơn
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16) # Output edge dim mới là 16
        )
        current_edge_dim = 16 

        # 2. GAT Layers
        # Layer 1
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=4, concat=True, edge_dim=current_edge_dim)
        # Output size: hidden_dim * 4 = 1024
        
        # Dùng LayerNorm thay vì BatchNorm để ổn định hơn với Graph
        self.ln1 = nn.LayerNorm(hidden_dim * 4) 
        
        # Layer 2
        self.conv2 = GATv2Conv(hidden_dim * 4, output_dim, heads=1, concat=False, edge_dim=current_edge_dim)
        
        # LayerNorm cuối cùng để output cùng scale với nhánh Face/Body
        self.ln2 = nn.LayerNorm(output_dim) 
        
        self.relu = nn.GELU() # GELU thường tốt hơn ReLU cho Transformer/GAT

    def forward(self, x, edge_index, edge_attr, batch):
        # --- [FIX QUAN TRỌNG] Kiểm tra edge_attr ---
        # Nếu edge_attr là None (do đồ thị không có cạnh), ta không thể đưa qua Linear Layer
        
        edge_embedding = None
        
        if edge_attr is not None:
            # Đôi khi edge_attr là tensor rỗng [0, 1], cần check numel() > 0
            if edge_attr.numel() > 0:
                # Nếu edge_attr là 1D [Num_Edges], cần unsqueeze thành [Num_Edges, 1]
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.unsqueeze(1)
                
                # Đưa qua encoder
                edge_embedding = self.edge_encoder(edge_attr)
        
        # --- Forward qua GAT ---
        # GATv2Conv của PyG chấp nhận edge_attr=None (nếu không có cạnh hoặc không dùng feature cạnh)
        
        # Layer 1
        x = self.conv1(x, edge_index, edge_attr=edge_embedding)
        x = self.ln1(x)
        x = self.relu(x)
        
        # Layer 2
        x = self.conv2(x, edge_index, edge_attr=edge_embedding)
        x = self.ln2(x)
        # Không dùng ReLU ở output cuối cùng
        
        # --- Pooling ---
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        graph_embedding = x_mean + x_max 
        
        return graph_embedding
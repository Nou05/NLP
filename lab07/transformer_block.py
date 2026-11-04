import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Tạo các ma trận chiếu W_q, W_k, W_v và W_o bằng nn.Linear cơ bản
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x có kích thước: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.size()
        
        # 1. Chiếu tuyến tính Q, K, V
        Q = self.W_q(x)  # [batch_size, seq_len, d_model]
        K = self.W_k(x)  # [batch_size, seq_len, d_model]
        V = self.W_v(x)  # [batch_size, seq_len, d_model]
        
        # 2. Tách thành n_heads: [batch_size, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Tính Scaled Dot-Product Attention
        # scores: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # attn_output: [batch_size, n_heads, seq_len, d_k]
        attn_output = torch.matmul(attn_weights, V)
        
        # 4. Concatenate các heads lại với nhau
        # [batch_size, seq_len, n_heads, d_k] -> [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 5. Chiếu qua ma trận đầu ra W_o
        output = self.W_o(attn_output)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Sử dụng các primitive nn.Linear, nn.ReLU, nn.Dropout
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> Linear -> ReLU -> Dropout -> Linear
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Tự lắp ghép các thành phần từ primitive layers
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bước 1: Multi-Head Self-Attention
        attn_out = self.attn(x)
        
        # Bước 2: Residual Connection + LayerNorm 1
        x1 = self.norm1(x + self.dropout1(attn_out))
        
        # Bước 3: Feed-Forward Network
        ffn_out = self.ffn(x1)
        
        # Bước 4: Residual Connection + LayerNorm 2
        output = self.norm2(x1 + self.dropout2(ffn_out))
        
        return output


# --- TEST CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # Đặt tham số mẫu
    batch_size = 2
    seq_len = 5
    d_model = 16
    n_heads = 4
    d_ff = 64

    # Khởi tạo dữ liệu giả lập (Dummy Input)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Khởi tạo mô hình tự custom
    custom_transformer_block = TransformerBlock(d_model, n_heads, d_ff)
    output = custom_transformer_block(x)
    
    print("=== TEST TRANSFORMER BLOCK ===")
    print(f"Kích thước Input : {x.shape}")
    print(f"Kích thước Output: {output.shape}")
    assert output.shape == x.shape, "Kích thước đầu ra phải trùng với kích thước đầu vào!"
    print("=> Tự cài đặt Transformer Block thành công!\n")
    
    # --- SO SÁNH VỚI PYTORCH BUILT-IN API (Chỉ dùng để so sánh theo đề bài) ---
    official_layer = nn.TransformerEncoderLayer(
        d_model=d_model, 
        nhead=n_heads, 
        dim_feedforward=d_ff, 
        batch_first=True
    )
    official_output = official_layer(x)
    print("=== SO SÁNH VỚI nn.TransformerEncoderLayer ===")
    print(f"Kích thước Official Output: {official_output.shape}")
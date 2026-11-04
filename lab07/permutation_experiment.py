import torch
import torch.nn as nn
import numpy as np

# Cố định random seed để kết quả tái lập được
torch.manual_seed(42)

# 1. Định nghĩa Vocabulary và Embeddings (d_model = 4)
vocab = {"dog": 0, "bites": 1, "man": 2}
d_model = 4

embedding_layer = nn.Embedding(num_embeddings=3, embedding_dim=d_model)

# Sequence 1: "dog bites man" -> Token IDs: [0, 1, 2]
S1_ids = torch.tensor([0, 1, 2])
# Sequence 2: "man bites dog" -> Token IDs: [2, 1, 0]
S2_ids = torch.tensor([2, 1, 0])

X1 = embedding_layer(S1_ids) # [3, 4]
X2 = embedding_layer(S2_ids) # [3, 4]

# 2. Định nghĩa Hàm Positional Encoding (Sinusoidal)
def get_positional_encoding(seq_len, d_model):
    PE = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            PE[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return PE

PE = get_positional_encoding(seq_len=3, d_model=d_model)

# 3. Định nghĩa Lớp Self-Attention
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = torch.matmul(Q, K.T) / np.sqrt(x.size(-1))
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, V)

self_attn = SelfAttention(d_model=d_model)

# -------------------------------------------------------------
# EXPERIMENT A: Self-Attention KHÔNG Positional Encoding
# -------------------------------------------------------------
Z1_A = self_attn(X1)
Z2_A = self_attn(X2)

# Để so sánh chính xác theo từ vựng, ta đảo ngược lại thứ tự hàng của Z2_A
# (vì trong S2, 'dog' ở hàng 2, 'man' ở hàng 0)
Z2_A_realigned = Z2_A[[2, 1, 0]] 

diff_A = torch.norm(Z1_A - Z2_A_realigned).item()

print("=== EXPERIMENT A (No Positional Encoding) ===")
print("Biểu diễn từ 'bites' trong S1:", Z1_A[1].detach().numpy())
print("Biểu diễn từ 'bites' trong S2:", Z2_A[1].detach().numpy())
print(f"Khoảng cách (Frobenius norm) giữa Z1 và Z2_realigned: {diff_A:.6f}")
if diff_A < 1e-5:
    print("=> KẾT LUẬN: Z1 và Z2 HOÀN TOÀN GIỐNG HỆT NHAU (mô hình không phân biệt được thứ tự)!\n")

# -------------------------------------------------------------
# EXPERIMENT B: Self-Attention CÓ Positional Encoding
# -------------------------------------------------------------
X1_PE = X1 + PE
X2_PE = X2 + PE

Z1_B = self_attn(X1_PE)
Z2_B = self_attn(X2_PE)

Z2_B_realigned = Z2_B[[2, 1, 0]]

diff_B = torch.norm(Z1_B - Z2_B_realigned).item()

print("=== EXPERIMENT B (With Positional Encoding) ===")
print("Biểu diễn từ 'bites' trong S1:", Z1_B[1].detach().numpy())
print("Biểu diễn từ 'bites' trong S2:", Z2_B[1].detach().numpy())
print(f"Khoảng cách (Frobenius norm) giữa Z1 và Z2_realigned: {diff_B:.6f}")
if diff_B > 1e-3:
    print("=> KẾT LUẬN: Z1 và Z2 KHÁC NHAU RÕ RỆT (mô hình đã phân biệt được thứ tự)!\n")
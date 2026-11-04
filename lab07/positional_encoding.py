import torch
import torch.nn as nn
import numpy as np

# 1. Định nghĩa Vocab và Embeddings giả định (d_model = 4)
vocab = {"I": 0, "love": 1, "NLP": 2}
embedding_layer = nn.Embedding(num_embeddings=3, embedding_dim=4)

# Cố định weights embedding để so sánh công bằng
torch.manual_seed(42)

# Chuỗi 1: ["I", "love", "NLP"] -> Token IDs: [0, 1, 2]
seq1_ids = torch.tensor([0, 1, 2])
# Chuỗi 2: ["NLP", "love", "I"] -> Token IDs: [2, 1, 0]
seq2_ids = torch.tensor([2, 1, 0])

X1 = embedding_layer(seq1_ids) # [3, 4]
X2 = embedding_layer(seq2_ids) # [3, 4]

# 2. Hàm tính Sinusoidal Positional Encoding (d_model = 4)
def get_positional_encoding(seq_len, d_model):
    PE = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            PE[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return PE

PE = get_positional_encoding(seq_len=3, d_model=4)

# 3. Định nghĩa Lớp Self-Attention đơn giản
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

self_attn = SelfAttention(d_model=4)

# --- THỰC NGHIỆM A: Without Positional Encoding ---
out1_no_pe = self_attn(X1)
out2_no_pe = self_attn(X2)

# --- THỰC NGHIỆM B: With Positional Encoding ---
X1_pe = X1 + PE
X2_pe = X2 + PE
out1_with_pe = self_attn(X1_pe)
out2_with_pe = self_attn(X2_pe)

print("--- WITHOUT POSITIONAL ENCODING ---")
# Token 'love' đứng ở vị trí 1 trong cả 2 câu
print("Representation từ 'love' trong câu 1:\n", out1_no_pe[1].detach().numpy())
print("Representation từ 'love' trong câu 2:\n", out2_no_pe[1].detach().numpy())
print("=> GIỐNG HỆT NHAU!")

print("\n--- WITH POSITIONAL ENCODING ---")
print("Representation từ 'love' trong câu 1:\n", out1_with_pe[1].detach().numpy())
print("Representation từ 'love' trong câu 2:\n", out2_with_pe[1].detach().numpy())
print("=> ĐÃ KHÁC NHAU nhờ ngữ cảnh vị trí xung quanh!")
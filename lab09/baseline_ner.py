# Lưu tại: lab09/baseline_ner.py
import json
import hashlib
import torch
import torch.nn as nn
from datasets import load_dataset
from metrics_utils import compute_entity_metrics

# 1. Tải dữ liệu CoNLL-2003
dataset = load_dataset("lhoestq/conll2003")
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
id2label = {i: l for i, l in enumerate(label_list)}

NUM_BUCKETS = 65536  # Kích thước bảng băm cố định (lũy thừa của 2)

def feature_to_id(feat_str):
    """Ánh xạ chuỗi đặc trưng sang ID cố định trong khoảng [0, NUM_BUCKETS - 1]."""
    h = hashlib.md5(feat_str.encode("utf-8")).hexdigest()
    return int(h, 16) % NUM_BUCKETS

# 2. Trích xuất đặc trưng mức token (cửa sổ ngữ cảnh [-1, +1] + hình thái từ)
def extract_token_features(tokens, i):
    w = tokens[i]
    feats = [
        f"w={w.lower()}",
        f"isupper={w.isupper()}",
        f"istitle={w.istitle()}",
        f"isdigit={w.isdigit()}",
        f"suf3={w[-3:]}",
        f"suf2={w[-2:]}",
        f"pre3={w[:3]}",
        f"BOS={i == 0}",
        f"EOS={i == len(tokens) - 1}",
    ]
    if i > 0:
        prev_w = tokens[i - 1]
        feats.extend([
            f"-1:w={prev_w.lower()}",
            f"-1:istitle={prev_w.istitle()}",
            f"-1:isupper={prev_w.isupper()}",
        ])
    if i < len(tokens) - 1:
        next_w = tokens[i + 1]
        feats.extend([
            f"+1:w={next_w.lower()}",
            f"+1:istitle={next_w.istitle()}",
            f"+1:isupper={next_w.isupper()}",
        ])
    return [feature_to_id(f) for f in feats]

print("1. Đang mã hóa đặc trưng tập train và test...")
def encode_split(split_data):
    flat_indices = []
    offsets = [0]
    labels = []
    for sample in split_data:
        tokens = sample["tokens"]
        for i, tag in enumerate(sample["ner_tags"]):
            feat_ids = extract_token_features(tokens, i)
            flat_indices.extend(feat_ids)
            offsets.append(len(flat_indices))
            labels.append(tag)
    return (
        torch.tensor(flat_indices, dtype=torch.long),
        torch.tensor(offsets[:-1], dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )

idx_train, off_train, y_train = encode_split(dataset["train"])
idx_test, off_test, y_test = encode_split(dataset["test"])

# 3. Định nghĩa Linear Classifier (Logistic Regression)
class BaselineLinearNER(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.bag = nn.EmbeddingBag(num_features, num_classes, mode="sum")
        self.bias = nn.Parameter(torch.zeros(num_classes))
        with torch.no_grad():
            self.bag.weight.zero_()

    def forward(self, indices, offsets):
        return self.bag(indices, offsets) + self.bias

print("2. Đang huấn luyện Baseline Linear Classifier...")
torch.manual_seed(42)
model = BaselineLinearNER(NUM_BUCKETS, len(label_list))
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(25):
    optimizer.zero_grad()
    logits = model(idx_train, off_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"   Epoch {epoch+1:02d}/25 - Loss: {loss.item():.4f}")

# 4. Đánh giá trên tập TEST
with torch.no_grad():
    y_pred_flat = model(idx_test, off_test).argmax(dim=1).tolist()

true_labels, true_predictions = [], []
idx = 0
for sample in dataset["test"]:
    n = len(sample["tokens"])
    true_labels.append([id2label[t] for t in sample["ner_tags"]])
    true_predictions.append([id2label[p] for p in y_pred_flat[idx : idx + n]])
    idx += n

p_base, r_base, f1_base = compute_entity_metrics(true_labels, true_predictions)
print(f"\n[BASELINE - Test Set] Precision: {p_base:.4f} | Recall: {r_base:.4f} | F1: {f1_base:.4f}")

# Lưu kết quả Baseline để tổng hợp với Transformer
with open("baseline_results.json", "w") as f:
    json.dump({"Model": "Baseline (Linear Classifier)", "Precision": p_base, "Recall": r_base, "F1": f1_base}, f)
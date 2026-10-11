# Lab 03: Word Embeddings & Semantic Search

Dự án này thực hiện nghiên cứu, cài đặt và đánh giá các phương pháp biểu diễn từ vựng (Word Representation) từ truyền thống đến hiện đại, bao gồm **Co-occurrence Matrix**, **Word2Vec (Skip-gram/CBOW)** và ứng dụng vào bài toán **Semantic Search** (Mở rộng truy vấn & Tìm kiếm dựa trên Dense Embedding).

---

## 🛠️ Cấu trúc thư mục dự án

```text
lab03/
├── cooccurrence.py         # Cài đặt thủ công Ma trận đồng xuất hiện & Cosine Similarity
├── word_embedding.ipynb    # Jupyter Notebook thực nghiệm Word2Vec, Hyperparameters & Semantic Search
├── results.csv             # Kết quả đánh giá thực nghiệm (Training time, Model size, Similarity, Analogy)
├── error_analysis.md       # Phân tích lỗi chi tiết cho ứng dụng Semantic Search
├── reflection.md           # Báo cáo so sánh các mô hình từ Word2Vec đến Transformer
└── README.md               # Tài liệu hướng dẫn dự án
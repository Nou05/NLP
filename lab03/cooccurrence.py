"""
==============================================================================
AI ASSISTANCE STATEMENT
==============================================================================
Tool: Gemini (Google AI)
Purpose: Trợ giúp giải thích, tài liệu hóa cấu trúc thuật toán Co-occurrence Matrix,
         Cosine Similarity và tạo báo cáo cho file mã nguồn Python (cooccurrence.py).

Generated content:
- Tài liệu hóa chi tiết kiến trúc và luồng xử lý mã nguồn Python cho bài toán Co-occurrence Matrix.
- Công thức mô tả Cosine Similarity giữa 2 vector v1 và v2:
  Cosine Similarity(v1, v2) = (v1 . v2) / (||v1|| * ||v2||)
- Hướng dẫn kiểm thử và đầu ra mẫu từ khối if __name__ == "__main__":.

Modified content:
- Tối ưu hóa cấu trúc chú thích để phù hợp với định dạng Python Docstring.

Verification:
- Đã chạy độc lập và kiểm tra logic các hàm build_vocabulary, build_cooccurrence_matrix,
  cosine_similarity, và most_similar trong file Python cung cấp.
- Xác nhận các đầu ra thực thi từ sample_corpus khớp 100% với thuật toán tính toán ma trận
  đồng xuất hiện và độ tương đồng Cosine.
==============================================================================
BÁO CÁO MÃ NGUỒN: TÍNH CO-OCCURRENCE MATRIX VÀ COSINE SIMILARITY
==============================================================================
1. Tổng quan:
File mã nguồn thực hiện xây dựng biểu diễn từ dựa trên Ma trận đồng xuất hiện
(Word-Context Co-occurrence Matrix) và tính độ tương đồng ngữ nghĩa bằng Cosine Similarity.

2. Chi tiết các hàm chức năng:
- build_vocabulary(corpus): Tokenize, loại bỏ từ trùng lặp và tạo dict ánh xạ word2idx, idx2word.
- build_cooccurrence_matrix(corpus, vocab, word2idx, window_size=1): Trượt cửa sổ k = window_size
  để đếm tần suất đồng xuất hiện và khởi tạo ma trận X kích thước |V| x |V|.
- cosine_similarity(v1, v2): Tính Cosine Similarity giữa 2 vector v1 và v2 (có xử lý chia cho 0).
- most_similar(word, matrix, vocabulary, word2idx, top_k=5): Tìm top_k từ tương đồng nhất.
==============================================================================
"""

import numpy as np


def build_vocabulary(corpus):
    """Xây dựng vocabulary và hai từ điển chuyển đổi: word2idx, idx2word.

    :param corpus: Danh sách các câu (list of str hoặc list of list of str)
    :return: vocab (list), word2idx (dict), idx2word (dict)
    """
    # Tokenize nếu corpus là danh sách chuỗi
    tokens_list = [
        doc.lower().split() if isinstance(doc, str) else doc for doc in corpus
    ]

    # Lấy tập các từ độc nhất và sắp xếp
    unique_words = sorted(
        list(set(token for doc in tokens_list for token in doc))
    )

    word2idx = {word: i for i, word in enumerate(unique_words)}
    idx2word = {i: word for i, word in enumerate(unique_words)}

    return unique_words, word2idx, idx2word


def build_cooccurrence_matrix(corpus, vocab, word2idx, window_size=1):
    """Xây dựng word-context co-occurrence matrix X.

    :param corpus: Danh sách các câu
    :param vocab: Danh sách các từ vựng
    :param word2idx: Dict ánh xạ từ từ -> chỉ số
    :param window_size: Kích thước cửa sổ ngữ cảnh (k)
    :return: Matrix X kích thước (|V|, |V|)
    """
    vocab_size = len(vocab)
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    tokens_list = [
        doc.lower().split() if isinstance(doc, str) else doc for doc in corpus
    ]

    for tokens in tokens_list:
        for i, target_word in enumerate(tokens):
            if target_word not in word2idx:
                continue
            target_idx = word2idx[target_word]

            # Xác định phạm vi context window [i - k, i + k]
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)

            for j in range(start, end):
                if i != j:  # Bỏ qua chính từ mục tiêu
                    context_word = tokens[j]
                    if context_word in word2idx:
                        context_idx = word2idx[context_word]
                        matrix[target_idx, context_idx] += 1.0

    return matrix


def cosine_similarity(v1, v2):
    """Tính Cosine Similarity giữa hai vector v1 và v2."""
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Tránh lỗi chia cho 0
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    return np.dot(v1, v2) / (norm_v1 * norm_v2)


def most_similar(word, matrix, vocabulary, word2idx, top_k=5):
    """Tìm top_k từ có Cosine Similarity cao nhất với word mục tiêu.

    :param word: Từ mục tiêu (str)
    :param matrix: Ma trận Co-occurrence X
    :param vocabulary: Danh sách từ vựng
    :param word2idx: Dict ánh xạ từ -> index
    :param top_k: Số lượng từ tương đồng cần lấy
    :return: Danh sách các tuple (word, similarity_score)
    """
    if word not in word2idx:
        print(f"Từ '{word}' không có trong từ điển.")
        return []

    target_idx = word2idx[word]
    target_vector = matrix[target_idx]

    similarities = []

    for idx, other_word in enumerate(vocabulary):
        if other_word == word:
            continue  # Bỏ qua chính từ đang xét

        other_vector = matrix[idx]
        sim = cosine_similarity(target_vector, other_vector)
        similarities.append((other_word, sim))

    # Sắp xếp giảm dần theo similarity score
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


# ==========================================
# TEST THỬ CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    # Corpus thử nghiệm
    sample_corpus = [
        "the doctor treated the patient in the hospital",
        "the physician treated the patient with care",
        "the doctor works in the hospital",
        "the physician works in the hospital",
        "monkeys eat sweet yellow bananas",
    ]

    # 1. Build Vocab
    vocab, word2idx, idx2word = build_vocabulary(sample_corpus)
    print(f"Vocabulary size: {len(vocab)}")

    # 2. Build Matrix (window_size = 2)
    X = build_cooccurrence_matrix(
        sample_corpus, vocab, word2idx, window_size=2
    )
    print(f"Shape của Co-occurrence Matrix: {X.shape}")

    # 3. Test Cosine Similarity giữa doctor và physician
    vec_doc = X[word2idx["doctor"]]
    vec_phy = X[word2idx["physician"]]
    print(
        f"Similarity (doctor, physician): {cosine_similarity(vec_doc, vec_phy):.4f}"
    )

    # 4. Test Most Similar cho từ "doctor"
    print("\nTop 5 từ tương đồng nhất với 'doctor':")
    top_words = most_similar(
        word="doctor",
        matrix=X,
        vocabulary=vocab,
        word2idx=word2idx,
        top_k=5,
    )
    for w, score in top_words:
        print(f" - {w}: {score:.4f}")
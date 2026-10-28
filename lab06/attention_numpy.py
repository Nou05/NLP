import numpy as np

def stable_softmax(x, axis=-1):
    """
    Tính Softmax ổn định về mặt số học (Numerical Stability).
    Trừ max(x) trước khi tính exp để tránh bị overflow khi score lớn.
    """
    # Trừ max theo trục chỉ định (thường là trục cuối cùng chứa các keys)
    max_x = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    Implement Scaled Dot-Product Attention bằng NumPy.
    
    Parameters:
    -----------
    Q : np.ndarray
        Ma trận Query, shape: (..., T_q, d_k) hoặc (d_k,)
    K : np.ndarray
        Ma trận Key, shape: (..., T_k, d_k)
    V : np.ndarray
        Ma trận Value, shape: (..., T_k, d_v)
        
    Returns:
    --------
    scores : np.ndarray
        Attention scores đã scale (S = Q K^T / sqrt(d_k))
    attention_weights : np.ndarray
        Attention weights sau khi qua Softmax (A = softmax(S))
    context : np.ndarray
        Context vector representation (C = A V)
    """
    # 1. Lấy kích thước d_k (chiều của Key)
    d_k = K.shape[-1]
    
    # 2. Tính Scaled Dot-Product Scores: S = (Q * K^T) / sqrt(d_k)
    # Sử dụng np.swapaxes hoặc np.matmul để tương thích cả dạng 2D lẫn 3D
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d_k)
    
    # 3. Tính Attention Weights bằng Softmax ổn định số học: A = softmax(S)
    attention_weights = stable_softmax(scores, axis=-1)
    
    # 4. Tính Context Vector: C = A * V
    context = np.matmul(attention_weights, V)
    
    return scores, attention_weights, context


# ==========================================
# KHU VỰC TEST HÀM (Self-Test / Unit Test)
# ==========================================
if __name__ == "__main__":
    print("--- Kiểm tra cài đặt Scaled Dot-Product Attention ---")
    
    # Giả lập dữ liệu đầu vào đơn giản (1 Query, 3 Keys/Values, d_k = 2)
    Q = np.array([[1.0, 0.0]])                 # Shape (1, 2)
    K = np.array([[1.0, 0.0],                  # Shape (3, 2)
                  [0.0, 1.0],
                  [1.0, 1.0]])
    V = np.array([[1.0, 0.0],                  # Shape (3, 2)
                  [0.0, 2.0],
                  [2.0, 1.0]])

    scores, weights, context = scaled_dot_product_attention(Q, K, V)
    
    print("\n1. Scaled Scores (S):")
    print(np.round(scores, 4))
    
    print("\n2. Attention Weights (A):")
    print(np.round(weights, 4))
    print(f"Tổng weights (phải bằng 1.0): {np.sum(weights):.4f}")
    
    print("\n3. Context Vector (C):")
    print(np.round(context, 4))
    
    # Test Numerical Stability với score cực lớn (Mục 17)
    print("\n--- Kiểm tra Numerical Stability với Score lớn ---")
    large_score = np.array([[1000.0, 1001.0, 1002.0]])
    stable_res = stable_softmax(large_score)
    print("Softmax kết quả (không bị NaN/Inf):", np.round(stable_res, 4))
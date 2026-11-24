
## 25. Wrong Start vs Wrong End Boundary Analysis

### Thống kê định lượng (trên 190 mẫu lỗi)
| Error Type                                      |   Count |   Percentage (%) |
|:------------------------------------------------|--------:|-----------------:|
| Wrong Start Only (start = wrong, end = correct) |      12 |             6.32 |
| Wrong End Only (start = correct, end = wrong)   |       6 |             3.16 |
| Both Start & End Wrong (Wrong Span/Sentence)    |     172 |            90.53 |
| TOTAL #wrong start (Wrong Start Only + Both)    |     184 |            96.84 |
| TOTAL #wrong end (Wrong End Only + Both)        |     178 |            93.68 |

* **Tổng số lần sai điểm bắt đầu (`#wrong start`):** `184` (trong đó sai riêng điểm bắt đầu: `12`).
* **Tổng số lần sai điểm kết thúc (`#wrong end`):** `178` (trong đó sai riêng điểm kết thúc: `6`).

### Giả thuyết về nguyên nhân
1. **Tại sao `#wrong end` thường cao hơn `#wrong start`:**
   * **Tín hiệu mở đầu rõ ràng hơn kết thúc:** Điểm bắt đầu của đáp án thường đứng ngay sau các từ khóa hỏi hoặc giới từ chỉ định (*"in"*, *"by"*, *"was"*, *"called"*), tạo ra dấu hiệu ngữ pháp mạnh giúp `start_classifier` dễ nhận diện.
   * **Sự mơ hồ của cụm danh từ mở rộng:** Điểm kết thúc của đáp án thường không có ranh giới cố định — một thực thể có thể dừng ở danh từ chính hoặc kéo dài thêm cụm bổ ngữ/giới từ phía sau (ví dụ: *"Wembley Stadium"* vs *"Wembley Stadium in London"*).
2. **Nguyên nhân gây ra `Wrong Start`:**
   * Mô hình thường bị phân vân khi có **mạo từ, danh xưng hoặc trạng từ ước lượng** đứng ngay trước thực thể chính (ví dụ: *"Sir Christopher Wren"* vs *"Christopher Wren"*, hoặc *"about 500"* vs *"500"*).
3. **Hạn chế kiến trúc:**
   * Hai bộ phân loại `start_logits` và `end_logits` dự đoán xác suất biên một cách **độc lập có điều kiện**, không có cơ chế ràng buộc cấu trúc cụm từ (constituency/dependency parsing) để đảm bảo `(start, end)` tạo thành một đơn vị ngữ pháp trọn vẹn.

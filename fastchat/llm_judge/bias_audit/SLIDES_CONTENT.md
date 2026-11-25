# Nội dung slide phần 2 – Thay GPT-4 bằng gpt-6-luna

Chèn sau slide "Bias and limitations" của phần paper. Ảnh lấy trong `bias_audit/figures/` (16:9, kéo thả
vào Canva thành nguyên slide). Chỗ `[[...]]` là thông tin nhóm phải tự điền. Chỗ **(CHỜ)** là kết quả cần chạy
API, điền sau khi chạy `run_luna_tests.py`.

---

## S1 – Slide chuyển phần
**Phần 2: Thay giám khảo GPT-4 bằng gpt-6-luna. Các thiên kiến paper tìm thấy có còn không?**
Quy trình: (1) lý do đổi judge → (2) kiểm tra thiên kiến của Luna bằng đúng các phép thử của paper →
(3) chỉ cải tiến những thiên kiến thực sự xuất hiện.

## S2 – Vì sao đổi judge
1. **Không tái lập được GPT-4 của paper.** Paper dùng `gpt-4-0314` và tự ghi chú kết quả có thể không tái lập
   với phiên bản GPT-4 sau (Phụ lục B). Phán quyết GPT-4 nhóm dùng là dữ liệu tác giả công bố (6/2023).
2. **Câu hỏi nghiên cứu mới:** thiên kiến của LLM-as-a-judge có còn ở judge đời mới không?
3. **Thực tế:** [[chi phí / khả năng truy cập API của nhóm]]

gpt-6-luna: [[nhà cung cấp, phiên bản, ngày chạy]]. Gọi qua OpenAI API với `reasoning_effort="none"`,
`max_completion_tokens=2048`; model không nhận tham số temperature (paper dùng temperature 0).

## S3 – Thiết lập thực nghiệm
- 80 câu MT-Bench × 2 lượt; 5 model được chấm: gpt-4, claude-v1, vicuna-13b-v1.3, guanaco-33b, alpaca-13b.
- Hai cách chấm của paper: single (1–10) và pairwise so với baseline gpt-3.5-turbo, mỗi cặp chấm 2 lần đổi thứ tự.
- Prompt gửi cho hai judge **giống hệt nhau** (đã kiểm tra trên cả 800 mục).
- GPT-4: phán quyết tác giả công bố (2023). Luna: nhóm tự chạy (11/2026) bằng pipeline FastChat gốc.
- Thống kê: khoảng tin cậy 95% bằng bootstrap lấy mẫu lại theo câu hỏi.

## S4 – Lưu ý khi đổi judge
| Vấn đề | Cách xử lý |
|---|---|
| Luna mạnh hơn nhiều so với model bị chấm → thiên kiến khó lộ trên dữ liệu tự nhiên | Dùng phép thử có kiểm soát của paper (cặp gần giống nhau, câu trả lời bị độn) |
| Luna có thể đã thấy MT-Bench khi huấn luyện | Nêu là hạn chế |
| Thang điểm khác nhau | Chỉ so thứ hạng và đồng thuận, không so điểm tuyệt đối |
| Không cố định được temperature | Kết quả có thể dao động giữa các lần chạy |
| 3% lượt chấm lỗi | Chạy lại, báo cáo tỉ lệ lỗi như cột Error ở Bảng 2 của paper |

## S5–S7 – So sánh hai judge (ảnh 01, 04, 03, 05, 09, 08, 02, 06)
Thông điệp chính:
- Luna chấm thấp hơn ~1.9 điểm và không dồn về 9–10 như GPT-4 (34% điểm 10 so với 7%).
- Lệch nhiều nhất ở câu hỏi mở; câu có đáp án tham chiếu thì gần như trùng nhau.
- Thứ hạng giữ nguyên trừ vicuna–guanaco (không có ý nghĩa thống kê).
- Hai judge đồng ý 78% khi cả hai chọn người thắng.

## S8 – Kiểm tra thiên kiến của Luna theo cách paper kiểm tra GPT-4 (bảng tổng hợp)
| Thiên kiến | Phép thử của paper | GPT-4 (paper) | GPT-4 (dữ liệu nhóm) | gpt-6-luna |
|---|---|---|---|---|
| Vị trí – cặp thật vs gpt-3.5 | Đổi thứ tự A/B | – | Nhất quán 81.0% | **85.6%** (ảnh 07, 11) |
| Vị trí – cặp gần giống nhau | Bảng 2 | 65% | – | **(CHỜ T2)** |
| Vị trí theo khoảng cách năng lực | Bảng 11 | 67.5% → 98.8% | 76% → 94% | 86–92%, gần như phẳng; llama **(CHỜ)** |
| Độ dài – câu trả lời thật | (mới, kiểu style control) | – | γ = +0.99 [0.36; 1.66], ưa câu dài | γ = −0.58 [−1.19; 0.04], không ưa câu dài (ảnh 12) |
| Độ dài – repetitive list | Bảng 3 | thất bại 8.7% | – | **(CHỜ T3)** |
| Chấm toán | Bảng 4 | thất bại 14/20 → 3/20 với reference | – | **(CHỜ T4)** |
| Độ tin cậy định dạng | cột Error Bảng 2 | 0% | 0% | 3.0% pairwise, 0.75% single (ảnh 10) |
| Tự thiên vị, đồng thuận với người | Hình 3, Bảng 5 | 85% (S2) | – | cần bộ phiếu chuyên gia |

Lưu ý khi nói: phần "dữ liệu thật" là cặp model vs gpt-3.5, dễ hơn phép thử của paper, nên không so trực tiếp
với con số 65%.

## S9 – Từ kết quả kiểm tra đến cải tiến
- Vị trí: làm P1/P2 nếu T2 cho thấy Luna nhất quán dưới ~80% trên cặp gần giống nhau.
- Độ dài: trên dữ liệu thật, Luna không ưa câu dài (GPT-4 có). Nếu T3 cũng sạch → V1/V2 không cần cho Luna;
  vẫn có thể kiểm chứng V1/V2 trên judge còn thiên kiến (GPT-4-style hoặc judge nhỏ 7–8B).
- Mọi cải tiến được đánh giá bằng đồng thuận với phiếu chuyên gia.

## S10 – Hạn chế và bước tiếp theo
- Chưa so với phiếu người → chưa nói được judge nào **đúng** hơn, chỉ nói được judge nào **khác**.
- 80 câu hỏi, 5 model; chênh lệch nhỏ là nhiễu.
- Giải thích hiệu ứng turn 2 của Luna là suy luận (pairwise phẳng), chưa kiểm chứng trực tiếp.
- Tiếp theo: chạy T2–T4, cặp llama, bộ phiếu chuyên gia (Bảng 5).

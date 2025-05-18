## Đề tài: Phân tích và đánh giá rủi ro tín dụng với German Credit Dataset

### Thông tin sinh viên

| Mã sinh viên | Họ tên           | Email                          |
|--------------|------------------|--------------------------------|
| 3122410282   | Nguyễn Tuyết Nhi | nhituyet20042008@gmail.com     |
| 3122410209   | Trương Thành Lâm | Lamtruong6442@gmail.com        |
| 3122410226   | H’ Như Lưk       | gaoluk1010@gmail.com           |


### Kế hoạch dự án (dự kiến)

| Giai đoạn                                       | Mô tả công việc                                                                                                                                                                  | Thời gian | Người phụ trách             |
|-------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|------------------------------|
| **Giai đoạn 1: Xác định vấn đề**                  | - Tìm hiểu tổng quan về rủi ro tín dụng. <br> - Khảo sát tài liệu liên quan. <br> - Định nghĩa vấn đề nghiên cứu.                                                                | Tuần 6    | H’ Như Lưk                   |
| **Giai đoạn 2: Khám phá dữ liệu**                  | - Tải và kiểm tra dataset. <br> - Thống kê mô tả dữ liệu. <br> - Xử lý dữ liệu bị thiếu, ngoại lệ.                                                                              | Tuần 7    | Trương Thành Lâm             |
| **Giai đoạn 3: Tiền xử lý & Trích xuất đặc trưng** | - Chuẩn hóa dữ liệu. <br> - Chuyển đổi dữ liệu danh mục. <br> - Chọn lọc đặc trưng quan trọng.                                                                                    | Tuần 8-9  | Nguyễn Tuyết Nhi             |
| **Giai đoạn 4: Xây dựng mô hình**                 | - Thử nghiệm các mô hình (Logistic Regression, Decision Tree, Random Forest, v.v.). <br> - Huấn luyện và tối ưu mô hình.                                                       | Tuần 10-11| Cả nhóm                      |
| **Giai đoạn 5: Đánh giá mô hình**                 | - So sánh hiệu suất mô hình. <br> - Phân tích độ quan trọng của đặc trưng. <br> - Đánh giá tác động của mô hình trong thực tế.                                                    | Tuần 12   | Cả nhóm                      |
| **Giai đoạn 6: Viết báo cáo & GitHub**            | - Tổng hợp kết quả. <br> - Viết báo cáo nghiên cứu. <br> - Cập nhật nội dung lên GitHub.                                                                                        | Tuần 13   | Cả nhóm                      |

## **Tổng quan đề tài**
### **1. Phân tích và đánh giá rủi ro tín dụng với German Credit Dataset**
Tín dụng là một yếu tố quan trọng trong nền kinh tế, giúp các cá nhân và doanh nghiệp tiếp cận nguồn vốn để phát triển. Tuy nhiên, các tổ chức tài chính phải đối mặt với rủi ro tín dụng – nguy cơ khách hàng không thể hoàn trả khoản vay. Để giảm thiểu rủi ro này, các ngân hàng và tổ chức tài chính sử dụng các mô hình phân tích dữ liệu để đánh giá khả năng trả nợ của khách hàng.

Dự án này sử dụng **German Credit Dataset** (german.csv), một tập dữ liệu phổ biến trong lĩnh vực phân tích rủi ro tín dụng. Mục tiêu của dự án là:
- Phân tích đặc điểm của các khách hàng vay vốn.
- Xây dựng mô hình dự đoán khả năng hoàn trả khoản vay.
- Đánh giá tầm quan trọng của các đặc trưng trong dự đoán rủi ro tín dụng.

Dự án được thực hiện theo các phương pháp nghiên cứu khoa học, bao gồm khảo sát tài liệu, tiền xử lý dữ liệu, áp dụng mô hình máy học và đánh giá kết quả.
#### Input: German Credit Dataset
#### Output: Mô hình phân loại rủi ro tín dụng

### 2. Các mô hình đánh giá rủi ro tín dụng
- **Random Forest**: Là một mô hình học ансамбль (ensemble) mạnh mẽ, hoạt động bằng cách xây dựng nhiều cây quyết định và tổng hợp kết quả dự đoán của chúng. Random Forest có khả năng xử lý dữ liệu phi tuyến tính và giảm thiểu hiện tượng quá khớp (overfitting).
- **Logistic Regression**: Là một thuật toán tuyến tính được sử dụng cho các bài toán phân loại nhị phân. Nó mô hình hóa xác suất của một trường hợp thuộc về một lớp cụ thể.
- **Decision Tree**: Là một mô hình cây cấu trúc, trong đó mỗi nút đại diện cho một thuộc tính, mỗi nhánh đại diện cho một quyết định và mỗi lá đại diện cho một kết quả phân loại. Decision Tree dễ hiểu và trực quan, nhưng có thể bị quá khớp.
- **K-Nearest Neighbors (KNN)**: Là một thuật toán dựa trên khoảng cách, phân loại một trường hợp mới dựa trên lớp của các trường hợp "gần nhất" với nó trong không gian đặc trưng. KNN đơn giản nhưng có thể tốn kém về mặt tính toán đối với dữ liệu lớn.
- **XGBoost**: Là một mô hình ансамбль dựa trên gradient boosting, được thiết kế để tối ưu hóa hiệu suất và tốc độ tính toán. XGBoost thường đạt được kết quả rất tốt trong các bài toán phân loại và hồi quy.
## **Đề cương đề tài**
https://docs.google.com/document/d/1mwllzgKaZaFJWuPMsWScaA3aB4CY0gVi24sI6a_11Jw/edit?usp=sharing

## **Báo cáo đề tài**
https://docs.google.com/document/d/1onQJ3sDFB7hqoVTcb14ZJ8Wtt5jucH7X_gqfGt3roEk/edit?tab=t.bxllhzhdx401


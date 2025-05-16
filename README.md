## **Tổng quan đề tài**
### **1. Phân tích và đánh giá rủi ro tín dụng với German Credit Dataset**
Tín dụng là một yếu tố quan trọng trong nền kinh tế, giúp các cá nhân và doanh nghiệp tiếp cận nguồn vốn để phát triển. Tuy nhiên, các tổ chức tài chính phải đối mặt với rủi ro tín dụng – nguy cơ khách hàng không thể hoàn trả khoản vay. Để giảm thiểu rủi ro này, các ngân hàng và tổ chức tài chính sử dụng các mô hình phân tích dữ liệu để đánh giá khả năng trả nợ của khách hàng.

Dự án này sử dụng **German Credit Dataset** (german.csv), một tập dữ liệu phổ biến trong lĩnh vực phân tích rủi ro tín dụng. Mục tiêu của dự án là:
- Phân tích đặc điểm của các khách hàng vay vốn.
- Xây dựng mô hình dự đoán khả năng hoàn trả khoản vay.
- Đánh giá tầm quan trọng của các đặc trưng trong dự đoán rủi ro tín dụng.

Dự án được thực hiện theo các phương pháp nghiên cứu khoa học, bao gồm khảo sát tài liệu, tiền xử lý dữ liệu, áp dụng mô hình máy học và đánh giá kết quả.
#### Input: German Credit Dataset
#### Output: Kết quả đánh giá rủi ro tín dụng

### **2. Mô hình đánh giá rủi ro tín dụng
- **Mô hình Rừng Ngẫu Nhiên** là một thuật toán học ансамбль (ensemble learning), có nghĩa là nó kết hợp dự đoán của nhiều mô hình học máy yếu hơn (trong trường hợp này là các cây quyết định) để tạo ra một mô hình mạnh mẽ và chính xác hơn.
- **Các thành phần chính của mô hình**:
+ Rừng (Forest): Thay vì một cây duy nhất, mô hình xây dựng một "rừng" gồm nhiều cây quyết định độc lập. Số lượng cây trong rừng thường rất lớn, có thể lên đến hàng trăm hoặc hàng nghìn.
+ Tính Ngẫu Nhiên (Randomness): Sự "ngẫu nhiên" thể hiện ở hai khía cạnh chính khi xây dựng mỗi cây:
  Lấy mẫu ngẫu nhiên có hoàn lại (Bootstrap Sampling): Mỗi cây được huấn luyện trên một tập dữ liệu con được tạo ra bằng cách lấy mẫu ngẫu nhiên có hoàn lại từ tập dữ liệu huấn luyện gốc. Điều này có nghĩa là một số mẫu có thể được chọn nhiều lần, trong khi một số mẫu khác có thể không được chọn. Mỗi cây nhìn nhận dữ liệu dưới một góc độ hơi khác nhau.
  Chọn thuộc tính ngẫu nhiên (Feature Randomness): Khi xây dựng mỗi cây, tại mỗi nút phân tách, mô hình chỉ xem xét một tập hợp con ngẫu nhiên các thuộc tính để tìm ra thuộc tính tốt nhất để phân chia dữ liệu. Điều này đảm bảo rằng các cây không quá giống nhau và tập trung vào các khía cạnh khác nhau của dữ liệu.
+ Bỏ phiếu đa số (Majority Voting): Khi cần đưa ra dự đoán cho một mẫu dữ liệu mới (ví dụ, đánh giá rủi ro tín dụng cho một người mới), mỗi cây trong rừng sẽ đưa ra một dự đoán ("tốt" hoặc "xấu"). Kết quả cuối cùng của Rừng Ngẫu Nhiên là kết quả được "bầu" nhiều nhất bởi các cây trong rừng.
- **Các ưu điểm của mô hình**:
+ Hiệu suất dự đoán cao: Rừng Ngẫu Nhiên thường đạt được độ chính xác dự đoán rất tốt trên nhiều loại bài toán, bao gồm cả phân loại rủi ro tín dụng. Việc kết hợp dự đoán của nhiều cây giúp giảm sai sót của từng cây riêng lẻ và tạo ra một dự đoán tổng thể ổn định và chính xác hơn.
+ Khả năng xử lý dữ liệu phức tạp: Tập German Credit Dataset có nhiều thuộc tính khác nhau, bao gồm cả thuộc tính số và thuộc tính phân loại. Rừng Ngẫu Nhiên có thể xử lý tốt cả hai loại thuộc tính này mà không đòi hỏi nhiều bước tiền xử lý phức tạp.
+ Khả năng chống chịu overfitting tốt: Overfitting (mô hình học quá tốt trên dữ liệu huấn luyện nhưng kém trên dữ liệu mới) là một vấn đề thường gặp trong đánh giá rủi ro tín dụng. Rừng Ngẫu Nhiên, nhờ cơ chế lấy mẫu ngẫu nhiên và chọn thuộc tính ngẫu nhiên, giúp giảm sự tương quan giữa các cây và làm cho mô hình tổng thể ít bị overfitting hơn so với một cây quyết định đơn lẻ.
+ Cung cấp thông tin về độ quan trọng của thuộc tính: Rừng Ngẫu Nhiên có thể ước tính mức độ quan trọng của từng thuộc tính trong việc đưa ra dự đoán rủi ro tín dụng. Điều này giúp các nhà phân tích hiểu được những yếu tố nào thực sự quan trọng trong việc đánh giá khả năng trả nợ (ví dụ: lịch sử tín dụng thường là yếu tố quan trọng nhất).
+ Ít nhạy cảm với các siêu tham số: Mặc dù Rừng Ngẫu Nhiên có một số siêu tham số cần điều chỉnh, hiệu suất của nó thường khá ổn định trong một phạm vi rộng của các giá trị siêu tham số, giúp việc điều chỉnh trở nên dễ dàng hơn so với một số thuật toán khác.
+ Khả năng xử lý dữ liệu bị thiếu (ở một mức độ nhất định): Mặc dù tiền xử lý vẫn quan trọng, Rừng Ngẫu Nhiên có thể hoạt động tương đối tốt ngay cả khi có một vài giá trị bị thiếu trong dữ liệu.

## **Đề cương đề tài**
https://docs.google.com/document/d/1mwllzgKaZaFJWuPMsWScaA3aB4CY0gVi24sI6a_11Jw/edit?usp=sharing

## **Báo cáo đề tài**
https://docs.google.com/document/d/1onQJ3sDFB7hqoVTcb14ZJ8Wtt5jucH7X_gqfGt3roEk/edit?tab=t.bxllhzhdx401


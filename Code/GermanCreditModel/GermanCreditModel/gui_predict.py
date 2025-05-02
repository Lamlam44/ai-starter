# gui_predict_form.py (Chỉ sửa phần nhập liệu cho cột Job)

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

# --- Nhập các thành phần từ predict_risk.py ---
# Đảm bảo predict_risk.py và các file .pkl nằm trong cùng thư mục.
try:
    # Import hàm dự đoán và danh sách thứ tự cột từ predict_risk.py
    # Lưu ý: Phiên bản này dựa trên predict_risk.py có tải model/scaler/encoders
    # và định nghĩa FEATURE_COLUMNS_ORDER.
    from predict_risk import predict_credit_risk, FEATURE_COLUMNS_ORDER

    # Kiểm tra nhanh xem predict_risk có tải được các file cần thiết không
    # (predict_risk.py sẽ báo lỗi hoặc đặt biến None nếu không tải được)
    # Ta kiểm tra một biến bất kỳ được tải trong predict_risk.py, ví dụ: model
    from predict_risk import model
    if model is None:
         raise ImportError("Các đối tượng .pkl chưa được tải thành công trong predict_risk.py.")

except ImportError as e:
    messagebox.showerror("Lỗi Khởi động", f"Không thể tải các thành phần dự đoán.\nChi tiết: {e}\n\nVui lòng đảm bảo:\n1. predict_risk.py nằm cùng thư mục.\n2. Đã chạy train_model.py thành công.\n3. Tất cả file .pkl đều có mặt.")
    sys.exit() # Thoát nếu không thể import hoặc tải model


# --- Các giá trị có thể có cho các cột phân loại KHÁC Job (Giống như trong phiên bản cũ bạn có thể đã dùng) ---
# Dictionary này chỉ để hiển thị dropdown cho các cột KHÔNG phải Job, không dùng từ .pkl
# Nếu bạn đã dùng phiên bản predict_risk.py mới, POSSIBLE_VALUES thực ra đã được tải từ đó.
# Tuy nhiên, để giữ thay đổi tối thiểu theo yêu cầu, ta vẫn định nghĩa lại ở đây.
POSSIBLE_VALUES_FOR_OTHER_CATEGORICALS = {
    'Sex': ['male', 'female'],
    'Housing': ['own', 'rent', 'free'],
    'Saving accounts': ['little', 'moderate', 'rich', 'quite rich', 'unknown'],
    'Checking account': ['little', 'moderate', 'rich', 'unknown'],
    'Purpose': ['car', 'radio/TV', 'furniture/equipment', 'business', 'education', 'repairs', 'domestic appliances', 'vacation/others'],
    'Telephone': ['yes', 'none'],
    'Foreign Worker': ['yes', 'no']
}

# --- Các giá trị và ghi chú cụ thể cho cột Job ---
JOB_POSSIBLE_VALUES = ['0', '1', '2', '3']
JOB_NOTES = ' (chọn: 0-unskilled non-res, 1-unskilled res, 2-skilled, 3-highly skilled)'


# --- Tạo giao diện ---
root = tk.Tk()
root.title("Dự đoán Rủi ro Tín dụng")

# Frame chứa các input
input_frame = ttk.Frame(root, padding="10")
input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Dictionary để lưu trữ các widget nhập liệu (biến StringVar)
input_widgets = {}

# Tạo Label và Widget cho từng đặc trưng theo thứ tự
row_index = 0
for col_name in FEATURE_COLUMNS_ORDER:
    # Label tên cột
    ttk.Label(input_frame, text=f"{col_name}:").grid(row=row_index, column=0, sticky=tk.W, pady=2, padx=5)

    widget = None # Biến để giữ widget nhập liệu cho cột hiện tại
    note_text = "" # Biến để giữ chú thích cho cột hiện tại

    # --- Xử lý RIÊNG cho cột Job ---
    if col_name == 'Job':
        var = tk.StringVar(root)
        if JOB_POSSIBLE_VALUES:
             var.set(JOB_POSSIBLE_VALUES[0]) # Đặt giá trị mặc định
        else:
             var.set("Lỗi giá trị Job")

        widget = ttk.OptionMenu(input_frame, var, var.get(), *JOB_POSSIBLE_VALUES)
        input_widgets[col_name] = var # Lưu biến StringVar
        note_text = JOB_NOTES # Lấy chú thích cụ thể cho Job

    # --- Xử lý cho các cột phân loại KHÁC Job (Nếu có trong danh sách cũ) ---
    elif col_name in POSSIBLE_VALUES_FOR_OTHER_CATEGORICALS:
        var = tk.StringVar(root)
        possible_values = POSSIBLE_VALUES_FOR_OTHER_CATEGORICALS[col_name]
        if possible_values:
             var.set(possible_values[0]) # Đặt giá trị mặc định
        else:
             var.set("Lỗi giá trị")

        widget = ttk.OptionMenu(input_frame, var, var.get(), *possible_values)
        input_widgets[col_name] = var # Lưu biến StringVar
        # Chú thích chung cho các cột phân loại khác
        note_text = f" (chọn: {', '.join(possible_values)})"


    # --- Xử lý cho các cột còn lại (Thường là cột số) ---
    else:
        var = tk.StringVar(root)
        widget = ttk.Entry(input_frame, textvariable=var, width=20)
        input_widgets[col_name] = var # Lưu biến StringVar
        # Chú thích chung cho cột số
        note_text = " (nhập số)" # Chú thích mặc định cho cột số


    # --- Đặt widget và chú thích vào grid ---
    if widget:
        widget.grid(row=row_index, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)

    if note_text: # Chỉ tạo label nếu có chú thích
        note_label = ttk.Label(input_frame, text=note_text, foreground="gray")
        note_label.grid(row=row_index, column=2, sticky=tk.W, padx=5)

    row_index += 1 # Tăng chỉ số dòng cho đặc trưng tiếp theo


# --- Hàm xử lý khi nhấn nút Dự đoán ---
def on_predict_button_click():
    input_data = {}
    try:
        # Thu thập dữ liệu từ các widget nhập liệu
        for col_name in FEATURE_COLUMNS_ORDER:
            # Kiểm tra xem widget cho cột này có tồn tại không
            if col_name not in input_widgets:
                 messagebox.showerror("Lỗi Thu thập", f"Không tìm thấy widget cho cột '{col_name}'.")
                 return

            value_str = input_widgets[col_name].get() # Lấy giá trị dạng string

            # *** Lưu ý: Không kiểm tra loại dữ liệu (số/phân loại) ở đây nữa ***
            # *** Hàm predict_credit_risk và preprocess_input trong predict_risk.py ***
            # *** sẽ lo việc chuyển đổi loại và mã hóa dựa trên thông tin đã lưu (.pkl) ***
            # *** Tuy nhiên, cần kiểm tra trường số có bị bỏ trống không ***

            # Kiểm tra trường số có bị bỏ trống không (dựa vào tên cột, không phải loại widget)
            # Cần biết cột nào là cột số. Ta không import NUMERIC_FEATURE_COLUMNS_USED ở đây để giữ tối thiểu thay đổi.
            # Cách đơn giản: dựa vào chú thích hoặc một list cứng các cột số nếu không muốn import thêm.
            # Dựa trên các chú thích phổ biến của dataset: Age, Duration, Installment rate, Present residence since, Number of existing credits, Number of people
            numeric_cols_approx = ['Age', 'Duration', 'Installment rate in percentage of disposable income',
                                 'Present residence since', 'Number of existing credits at this bank',
                                 'Number of people being liable to provide maintenance for']

            if col_name in numeric_cols_approx:
                if value_str.strip() == "":
                    messagebox.showwarning("Thiếu nhập liệu", f"Vui lòng nhập giá trị cho cột số '{col_name}'.")
                    return # Dừng nếu bỏ trống cột số
                # Không cố gắng chuyển đổi sang số ở đây, để predict_risk.py xử lý
                # predict_risk.py's preprocess_input expects string input and handles conversion

            input_data[col_name] = value_str # Lấy giá trị string cho cả số và phân loại

        # print("Dữ liệu thu thập được:", input_data) # Debug

        # Gọi hàm dự đoán từ predict_risk.py
        # Hàm này sẽ nhận dictionary với giá trị string và tự xử lý chuyển đổi/mã hóa/chuẩn hóa
        prediction_result_text = predict_credit_risk(input_data)

        # Hiển thị kết quả
        result_label.config(text=f"Kết quả dự đoán: {prediction_result_text}")

    except Exception as e:
        # Bắt các lỗi khác có thể xảy ra trong quá trình xử lý hoặc dự đoán
        messagebox.showerror("Lỗi dự đoán", f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")
        # print("Lỗi chi tiết:", e) # Debug


# --- Tạo nút Dự đoán ---
# row_index hiện tại là số dòng sau input cuối cùng + 1
predict_button = ttk.Button(root, text="Dự đoán Rủi ro Tín dụng", command=on_predict_button_click)
predict_button.grid(row=row_index, column=0, columnspan=3, pady=10) # Đặt dưới các input

# --- Label hiển thị kết quả ---
result_label = ttk.Label(root, text="Kết quả dự đoán: Chờ nhập liệu...", font=('TkDefaultFont', 10, 'bold'))
result_label.grid(row=row_index + 1, column=0, columnspan=3, pady=10) # Đặt dưới nút


# --- Chạy vòng lặp chính của giao diện ---
root.mainloop()
# gui_predict_form.py (Tự tải model và tiền xử lý từ saved_models)

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler # Import để có thể inverse transform target

# --- Định nghĩa đường dẫn đến thư mục chứa các file .pkl ---
MODEL_DIR = 'src/saved_models'

# --- Định nghĩa tên các file .pkl ---
MODEL_FILENAME = 'credit_risk_model.pkl'
SCALER_FILENAME = 'scaler.pkl'
ENCODERS_FILENAME = 'label_encoders.pkl'
TARGET_ENCODER_FILENAME = 'target_encoder.pkl'
FEATURE_COLUMNS_ORDER_FILENAME = 'feature_cols_order.pkl'
CATEGORICAL_FEATURE_COLUMNS_USED_FILENAME = 'categorical_features_used.pkl'
NUMERIC_FEATURE_COLUMNS_USED_FILENAME = 'numeric_features_used.pkl'
FILLNA_VALUE_FILENAME = 'fillna_value.pkl'

# --- Tạo đường dẫn đầy đủ đến các file .pkl ---
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
SCALER_PATH = os.path.join(MODEL_DIR, SCALER_FILENAME)
ENCODERS_PATH = os.path.join(MODEL_DIR, ENCODERS_FILENAME)
TARGET_ENCODER_PATH = os.path.join(MODEL_DIR, TARGET_ENCODER_FILENAME)
FEATURE_COLUMNS_ORDER_PATH = os.path.join(MODEL_DIR, FEATURE_COLUMNS_ORDER_FILENAME)
CATEGORICAL_FEATURE_COLUMNS_USED_PATH = os.path.join(MODEL_DIR, CATEGORICAL_FEATURE_COLUMNS_USED_FILENAME)
NUMERIC_FEATURE_COLUMNS_USED_PATH = os.path.join(MODEL_DIR, NUMERIC_FEATURE_COLUMNS_USED_FILENAME)
FILLNA_VALUE_PATH = os.path.join(MODEL_DIR, FILLNA_VALUE_FILENAME)

# --- Load các đối tượng đã lưu ---
model = None
scaler = None
label_encoders = {}
target_le = None
FEATURE_COLUMNS_ORDER = None
CATEGORICAL_FEATURE_COLUMNS_USED = None
NUMERIC_FEATURE_COLUMNS_USED = None
fillna_value = None

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    target_le = joblib.load(TARGET_ENCODER_PATH)
    FEATURE_COLUMNS_ORDER = joblib.load(FEATURE_COLUMNS_ORDER_PATH)
    CATEGORICAL_FEATURE_COLUMNS_USED = joblib.load(CATEGORICAL_FEATURE_COLUMNS_USED_PATH)
    NUMERIC_FEATURE_COLUMNS_USED = joblib.load(NUMERIC_FEATURE_COLUMNS_USED_PATH)
    fillna_value = joblib.load(FILLNA_VALUE_PATH)

    print("Đã tải mô hình và các đối tượng tiền xử lý thành công từ thư mục 'saved_models'.")

except FileNotFoundError as e:
    messagebox.showerror("Lỗi Khởi động", f"Không tìm thấy file .pkl tại đường dẫn: {e.filename}.\nĐảm bảo thư mục 'saved_models' tồn tại và chứa các file này.")
    sys.exit()
except Exception as e:
    messagebox.showerror("Lỗi Khởi động", f"Lỗi không xác định khi tải file .pkl: {e}")
    sys.exit()

# --- Các giá trị có thể có cho các cột phân loại (Lấy từ danh sách đã lưu) ---
POSSIBLE_VALUES_FOR_CATEGORICALS = {}
if label_encoders:
    for col, encoder in label_encoders.items():
        POSSIBLE_VALUES_FOR_CATEGORICALS[col] = list(encoder.classes_)

# --- Các giá trị và ghi chú cụ thể cho cột Job ---
JOB_POSSIBLE_VALUES = POSSIBLE_VALUES_FOR_CATEGORICALS.get('Job', ['0', '1', '2', '3'])
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

    # --- Xử lý cho các cột phân loại KHÁC Job ---
    elif col_name in POSSIBLE_VALUES_FOR_CATEGORICALS and col_name != 'Job':
        var = tk.StringVar(root)
        possible_values = POSSIBLE_VALUES_FOR_CATEGORICALS[col_name]
        if possible_values:
            var.set(possible_values[0]) # Đặt giá trị mặc định
        else:
            var.set(f"Lỗi giá trị {col_name}")

        widget = ttk.OptionMenu(input_frame, var, var.get(), *possible_values)
        input_widgets[col_name] = var # Lưu biến StringVar
        note_text = f" (chọn: {', '.join(possible_values)})"

    # --- Xử lý cho các cột còn lại (Thường là cột số) ---
    else:
        var = tk.StringVar(root)
        widget = ttk.Entry(input_frame, textvariable=var, width=20)
        input_widgets[col_name] = var # Lưu biến StringVar
        note_text = " (nhập số)" # Chú thích mặc định cho cột số

    # --- Đặt widget và chú thích vào grid ---
    if widget:
        widget.grid(row=row_index, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)

    if note_text: # Chỉ tạo label nếu có chú thích
        note_label = ttk.Label(input_frame, text=note_text, foreground="gray")
        note_label.grid(row=row_index, column=2, sticky=tk.W, padx=5)

    row_index += 1 # Tăng chỉ số dòng cho đặc trưng tiếp theo

# --- Hàm tiền xử lý dữ liệu đầu vào ---
def preprocess_input(input_data):
    try:
        input_df = pd.DataFrame([input_data])

        # 1. Fillna
        cols_to_fillna_existing = [col for col in ['Saving accounts', 'Checking account'] if col in input_df.columns]
        for col in cols_to_fillna_existing:
            input_df[col] = input_df[col].fillna(fillna_value)

        # 2. Mã hóa các cột phân loại
        for col in CATEGORICAL_FEATURE_COLUMNS_USED:
            if col in input_df.columns and col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])
            elif col in input_df.columns:
                print(f"Cảnh báo: Không tìm thấy label encoder cho cột '{col}'.")

        # 3. Chuẩn hóa các cột số
        numeric_cols_present = [col for col in NUMERIC_FEATURE_COLUMNS_USED if col in input_df.columns]
        if numeric_cols_present and scaler:
            input_df[numeric_cols_present] = scaler.transform(input_df[numeric_cols_present])
        elif numeric_cols_present:
            print("Cảnh báo: Không tìm thấy scaler.")

        # Đảm bảo thứ tự cột
        input_df = input_df[FEATURE_COLUMNS_ORDER]
        return input_df

    except Exception as e:
        messagebox.showerror("Lỗi Tiền Xử Lý", f"Đã xảy ra lỗi trong quá trình tiền xử lý: {e}")
        return None

# --- Hàm dự đoán ---
def predict_credit_risk(input_data):
    try:
        processed_input = preprocess_input(input_data)
        if processed_input is not None and model is not None:
            prediction = model.predict(processed_input)
            predicted_class = target_le.inverse_transform(prediction)[0] if target_le else str(prediction[0])
            return predicted_class
        else:
            return "Lỗi: Không thể thực hiện dự đoán."
    except Exception as e:
        messagebox.showerror("Lỗi Dự Đoán", f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")
        return "Lỗi dự đoán."

# --- Hàm xử lý khi nhấn nút Dự đoán ---
def on_predict_button_click():
    input_data = {}
    try:
        # Thu thập dữ liệu từ các widget nhập liệu
        for col_name in FEATURE_COLUMNS_ORDER:
            if col_name not in input_widgets:
                messagebox.showerror("Lỗi Thu thập", f"Không tìm thấy widget cho cột '{col_name}'.")
                return

            value_str = input_widgets[col_name].get()
            input_data[col_name] = value_str

        # Gọi hàm dự đoán
        prediction_result_text = predict_credit_risk(input_data)

        # Hiển thị kết quả
        result_label.config(text=f"Kết quả dự đoán: {prediction_result_text}")

    except Exception as e:
        messagebox.showerror("Lỗi Chung", f"Đã xảy ra lỗi: {e}")

# --- Tạo nút Dự đoán ---
predict_button = ttk.Button(root, text="Dự đoán Rủi ro Tín dụng", command=on_predict_button_click)
predict_button.grid(row=row_index, column=0, columnspan=3, pady=10)

# --- Label hiển thị kết quả ---
result_label = ttk.Label(root, text="Kết quả dự đoán: Chờ nhập liệu...", font=('TkDefaultFont', 10, 'bold'))
result_label.grid(row=row_index + 1, column=0, columnspan=3, pady=10)

# --- Chạy vòng lặp chính của giao diện ---
root.mainloop()
import tkinter as tk
from tkinter import ttk, messagebox, Toplevel, scrolledtext
import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from train_logistic_regression import train_and_evaluate_logistic_regression_with_credit_amount

# --- Định nghĩa đường dẫn ---
MODEL_DIR = 'src/LogisticRegression_with_CreditAmount/saved_models'
MODEL_FILENAME = 'logistic_regression_model_with_credit_amount.pkl'
SCALER_FILENAME = 'scaler_lr_with_credit_amount.pkl'
ENCODERS_FILENAME = 'label_encoders_lr_with_credit_amount.pkl'
TARGET_ENCODER_FILENAME = 'target_encoder_lr_with_credit_amount.pkl'
FEATURE_COLUMNS_ORDER_FILENAME = 'feature_cols_order_lr_with_credit_amount.pkl'
CATEGORICAL_FEATURE_COLUMNS_USED_FILENAME = 'categorical_features_used_lr_with_credit_amount.pkl'
NUMERIC_FEATURE_COLUMNS_USED_FILENAME = 'numeric_features_used_lr_with_credit_amount.pkl'
FILLNA_VALUE_FILENAME = 'fillna_value_lr_with_credit_amount.pkl'
DATA_PATH = 'src/german_credit_data (1).csv'

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
    print("Đã tải mô hình Logistic Regression (bao gồm Credit amount) và các đối tượng tiền xử lý thành công.")
except FileNotFoundError as e:
    messagebox.showerror("Lỗi Khởi động", f"Không tìm thấy file .pkl: {e.filename}.\nĐảm bảo thư mục 'src/LogisticRegression_with_CreditAmount/saved_models' tồn tại và chứa các file này.")
    sys.exit()
except Exception as e:
    messagebox.showerror("Lỗi Khởi động", f"Lỗi không xác định khi tải file .pkl: {e}")
    sys.exit()

# --- Các giá trị có thể có cho các cột phân loại ---
POSSIBLE_VALUES_FOR_CATEGORICALS = {}
if label_encoders:
    for col, encoder in label_encoders.items():
        POSSIBLE_VALUES_FOR_CATEGORICALS[col] = list(encoder.classes_)

JOB_POSSIBLE_VALUES = POSSIBLE_VALUES_FOR_CATEGORICALS.get('Job', ['0', '1', '2', '3'])
JOB_NOTES = ' (chọn: 0-unskilled non-res, 1-unskilled res, 2-skilled, 3-highly skilled)'

# --- Tạo giao diện ---
root = tk.Tk()
root.title("Dự đoán Rủi ro Tín dụng (Logistic Regression - có Credit amount)")

input_frame = ttk.Frame(root, padding="10")
input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

input_widgets = {}
row_index = 0
for col_name in FEATURE_COLUMNS_ORDER:
    ttk.Label(input_frame, text=f"{col_name}:").grid(row=row_index, column=0, sticky=tk.W, pady=2, padx=5)
    widget = None
    note_text = ""
    if col_name == 'Job':
        var = tk.StringVar(root)
        var.set(JOB_POSSIBLE_VALUES[0] if JOB_POSSIBLE_VALUES else "Lỗi giá trị Job")
        widget = ttk.OptionMenu(input_frame, var, var.get(), *JOB_POSSIBLE_VALUES)
        input_widgets[col_name] = var
        note_text = JOB_NOTES
    elif col_name in POSSIBLE_VALUES_FOR_CATEGORICALS and col_name != 'Job':
        var = tk.StringVar(root)
        possible_values = POSSIBLE_VALUES_FOR_CATEGORICALS[col_name]
        var.set(possible_values[0] if possible_values else f"Lỗi giá trị {col_name}")
        widget = ttk.OptionMenu(input_frame, var, var.get(), *possible_values)
        input_widgets[col_name] = var
        note_text = f" (chọn: {', '.join(possible_values)})"
    else:
        var = tk.StringVar(root)
        widget = ttk.Entry(input_frame, textvariable=var, width=20)
        input_widgets[col_name] = var
        note_text = " (nhập số)"
    if widget:
        widget.grid(row=row_index, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
    if note_text:
        note_label = ttk.Label(input_frame, text=note_text, foreground="gray")
        note_label.grid(row=row_index, column=2, sticky=tk.W, padx=5)
    row_index += 1

# --- Hàm tiền xử lý dữ liệu đầu vào ---
def preprocess_input(input_data):
    try:
        input_df = pd.DataFrame([input_data])
        cols_to_fillna_existing = [col for col in ['Saving accounts', 'Checking account'] if col in input_df.columns]
        for col in cols_to_fillna_existing:
            input_df[col] = input_df[col].fillna(fillna_value)
        for col in CATEGORICAL_FEATURE_COLUMNS_USED:
            if col in input_df.columns and col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])
            elif col in input_df.columns:
                print(f"Cảnh báo: Không tìm thấy label encoder cho cột '{col}'.")
        numeric_cols_present = [col for col in NUMERIC_FEATURE_COLUMNS_USED if col in input_df.columns]
        if numeric_cols_present and scaler:
            input_df[numeric_cols_present] = scaler.transform(input_df[numeric_cols_present])
        elif numeric_cols_present:
            print("Cảnh báo: Không tìm thấy scaler.")
        input_df = input_df[FEATURE_COLUMNS_ORDER]
        return input_df
    except Exception as e:
        messagebox.showerror("Lỗi Tiền Xử Lý", f"Đã xảy ra lỗi: {e}")
        return None

# --- Hàm dự đoán ---
def predict_credit_risk(input_data):
    try:
        processed_input = preprocess_input(input_data)
        if processed_input is not None and model is not None:
            prediction = model.predict(processed_input)
            predicted_class_encoded = prediction[0]
            predicted_class_original = target_le.inverse_transform([predicted_class_encoded])[0] if target_le else str(predicted_class_encoded)
            return predicted_class_original
        else:
            return "Lỗi: Không thể thực hiện dự đoán."
    except Exception as e:
        messagebox.showerror("Lỗi Dự Đoán", f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")
        return "Lỗi dự đoán."

# --- Hàm xử lý khi nhấn nút Dự đoán ---
def on_predict_button_click():
    input_data = {}
    try:
        for col_name in FEATURE_COLUMNS_ORDER:
            if col_name not in input_widgets:
                messagebox.showerror("Lỗi Thu thập", f"Không tìm thấy widget cho cột '{col_name}'.")
                return
            value_str = input_widgets[col_name].get()
            input_data[col_name] = value_str
        prediction_result_text = predict_credit_risk(input_data)
        result_label.config(text=f"Kết quả dự đoán: {prediction_result_text}")
    except Exception as e:
        messagebox.showerror("Lỗi Chung", f"Đã xảy ra lỗi: {e}")

# --- Hàm xử lý khi nhấn nút Huấn luyện Mô hình ---
def on_train_button_click():
    messagebox.showinfo("Bắt đầu Huấn luyện", "Quá trình huấn luyện mô hình Logistic Regression (bao gồm Credit amount) đang được tiến hành...")
    evaluation_metrics = train_and_evaluate_logistic_regression_with_credit_amount(DATA_PATH)
    if evaluation_metrics:
        accuracy = evaluation_metrics['accuracy']
        conf_matrix_str = np.array_str(evaluation_metrics['confusion_matrix'])
        class_report_str = evaluation_metrics['classification_report']

        # Tạo một cửa sổ mới để hiển thị kết quả đánh giá
        evaluation_window = Toplevel(root)
        evaluation_window.title("Kết quả Đánh giá Mô hình Logistic Regression (có Credit amount)")

        accuracy_label = ttk.Label(evaluation_window, text=f"Accuracy: {accuracy:.2f}")
        accuracy_label.pack(pady=5, padx=10)

        conf_matrix_label = ttk.Label(evaluation_window, text="Confusion Matrix:")
        conf_matrix_label.pack(pady=5, padx=10)
        conf_matrix_text = scrolledtext.ScrolledText(evaluation_window, height=5, width=40)
        conf_matrix_text.insert(tk.END, conf_matrix_str)
        conf_matrix_text.config(state=tk.DISABLED)
        conf_matrix_text.pack(pady=5, padx=10)

        class_report_label = ttk.Label(evaluation_window, text="Classification Report:")
        class_report_label.pack(pady=5, padx=10)
        class_report_text = scrolledtext.ScrolledText(evaluation_window, height=10, width=80)
        class_report_text.insert(tk.END, class_report_str)
        class_report_text.config(state=tk.DISABLED)
        class_report_text.pack(pady=5, padx=10)
    else:
        messagebox.showerror("Lỗi Huấn luyện", "Đã xảy ra lỗi trong quá trình huấn luyện mô hình Logistic Regression (bao gồm Credit amount).")

# --- Tạo nút Dự đoán ---
predict_button = ttk.Button(root, text="Dự đoán Rủi ro Tín dụng", command=on_predict_button_click)
predict_button.grid(row=row_index, column=0, columnspan=3, pady=10)

# --- Label hiển thị kết quả dự đoán ---
result_label = ttk.Label(root, text="Kết quả dự đoán: Chờ nhập liệu...", font=('TkDefaultFont', 10, 'bold'))
result_label.grid(row=row_index + 1, column=0, columnspan=3, pady=10)

# --- Tạo nút Huấn luyện Mô hình ---
train_button = ttk.Button(root, text="Huấn luyện lại Mô hình & Xem Đánh giá (có Credit amount)", command=on_train_button_click)
train_button.grid(row=row_index + 2, column=0, columnspan=3, pady=10)

# --- Chạy vòng lặp chính của giao diện ---
root.mainloop()
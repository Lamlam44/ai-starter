# app_xgboost.py

import tkinter as tk
from tkinter import ttk, messagebox, Toplevel, scrolledtext
import sys
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- Định nghĩa đường dẫn ---
MODEL_DIR = 'src/XGBoost/saved_models'
MODEL_FILENAME = 'xgboost_model.pkl'
LABEL_ENCODERS_FILENAME = 'label_encoders_xgboost.pkl'
TARGET_ENCODER_FILENAME = 'target_encoder_xgboost.pkl'
SCALER_FILENAME = 'scaler_xgboost.pkl'
FEATURE_COLUMNS_ORDER_FILENAME = 'feature_cols_order_xgboost.pkl'
CATEGORICAL_FEATURE_COLUMNS_FILENAME = 'categorical_features_xgboost.pkl'
NUMERICAL_FEATURE_COLUMNS_FILENAME = 'numerical_features_xgboost.pkl'
FILLNA_VALUE_FILENAME = 'fillna_value_xgboost.pkl'
DATA_PATH = 'src/german_credit_data (1).csv'

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
LABEL_ENCODERS_PATH = os.path.join(MODEL_DIR, LABEL_ENCODERS_FILENAME)
TARGET_ENCODER_PATH = os.path.join(MODEL_DIR, TARGET_ENCODER_FILENAME)
SCALER_PATH = os.path.join(MODEL_DIR, SCALER_FILENAME)
FEATURE_COLUMNS_ORDER_PATH = os.path.join(MODEL_DIR, FEATURE_COLUMNS_ORDER_FILENAME)
CATEGORICAL_FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, CATEGORICAL_FEATURE_COLUMNS_FILENAME)
NUMERICAL_FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, NUMERICAL_FEATURE_COLUMNS_FILENAME)
FILLNA_VALUE_PATH = os.path.join(MODEL_DIR, FILLNA_VALUE_FILENAME)

# --- Load các đối tượng đã lưu ---
model = None
label_encoders = {}
target_le = None
scaler = None
FEATURE_COLUMNS_ORDER = None
CATEGORICAL_FEATURE_COLUMNS = None
NUMERICAL_FEATURE_COLUMNS = None
fillna_value = None

try:
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    target_le = joblib.load(TARGET_ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    FEATURE_COLUMNS_ORDER = joblib.load(FEATURE_COLUMNS_ORDER_PATH)
    CATEGORICAL_FEATURE_COLUMNS = joblib.load(CATEGORICAL_FEATURE_COLUMNS_PATH)
    NUMERICAL_FEATURE_COLUMNS = joblib.load(NUMERICAL_FEATURE_COLUMNS_PATH)
    fillna_value = joblib.load(FILLNA_VALUE_PATH)
    print(f"Đã tải mô hình XGBoost và các đối tượng tiền xử lý thành công. Giá trị fillna: {fillna_value}")
except FileNotFoundError as e:
    messagebox.showerror("Lỗi Khởi động", f"Không tìm thấy file .pkl: {e.filename}.\nĐảm bảo thư mục 'saved_models' trong 'src/XGBoost' tồn tại và chứa các file này.")
    sys.exit()
except Exception as e:
    messagebox.showerror("Lỗi Khởi động", f"Lỗi không xác định khi tải file .pkl: {e}")
    sys.exit()

# --- Các giá trị có thể có cho các cột phân loại ---
POSSIBLE_VALUES_FOR_CATEGORICALS = {}
try:
    df = pd.read_csv(DATA_PATH)
    for col in ['Sex', 'Housing', 'Saving accounts', 'Purpose', 'Checking account']:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
            POSSIBLE_VALUES_FOR_CATEGORICALS[col] = [str(val) for val in df[col].unique().tolist()]
    POSSIBLE_VALUES_FOR_CATEGORICALS['Job'] = ['0', '1', '2', '3']
except FileNotFoundError:
    messagebox.showerror("Lỗi Dữ liệu", f"Không tìm thấy file dữ liệu tại: {DATA_PATH}")
    sys.exit()
except Exception as e:
    messagebox.showerror("Lỗi Dữ liệu", f"Lỗi khi đọc file dữ liệu: {e}")
    sys.exit()

JOB_POSSIBLE_VALUES = POSSIBLE_VALUES_FOR_CATEGORICALS.get('Job', ['0', '1', '2', '3'])
HOUSING_POSSIBLE_VALUES = POSSIBLE_VALUES_FOR_CATEGORICALS.get('Housing', ['own', 'rent', 'free', 'unknown'])
SAVING_ACCOUNTS_POSSIBLE_VALUES = POSSIBLE_VALUES_FOR_CATEGORICALS.get('Saving accounts', ['little', 'moderate', 'quite rich', 'rich', 'unknown'])
SEX_POSSIBLE_VALUES = POSSIBLE_VALUES_FOR_CATEGORICALS.get('Sex', ['male', 'female', 'unknown'])
PURPOSE_POSSIBLE_VALUES = POSSIBLE_VALUES_FOR_CATEGORICALS.get('Purpose', ['car', 'furniture/equipment', 'radio/TV', 'domestic appliances', 'repairs', 'education', 'business', 'vacation/others', 'unknown'])
CHECKING_ACCOUNT_POSSIBLE_VALUES = POSSIBLE_VALUES_FOR_CATEGORICALS.get('Checking account', ['little', 'moderate', 'rich', 'unknown'])

JOB_NOTES = ' (chọn: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)'
HOUSING_NOTES = ' (chọn: own, rent, free)'
SAVING_ACCOUNTS_NOTES = ' (chọn: little, moderate, quite rich, rich)'
SEX_NOTES = ' (chọn: male, female)'
PURPOSE_NOTES = ' (chọn: car, furniture/equipment, radio/TV, domestic appliances', 'repairs', 'education', 'business', 'vacation/others)'
CHECKING_ACCOUNT_NOTES = f" (chọn: {', '.join(CHECKING_ACCOUNT_POSSIBLE_VALUES)})"

# --- Tạo giao diện ---
root = tk.Tk()
root.title("Dự đoán Rủi ro Tín dụng (XGBoost)")

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
        widget = ttk.Combobox(input_frame, textvariable=var, values=JOB_POSSIBLE_VALUES, state='readonly')
        input_widgets[col_name] = var
        note_text = JOB_NOTES
    elif col_name == 'Housing':
        var = tk.StringVar(root)
        var.set(HOUSING_POSSIBLE_VALUES[0] if HOUSING_POSSIBLE_VALUES else "Lỗi giá trị Housing")
        widget = ttk.Combobox(input_frame, textvariable=var, values=HOUSING_POSSIBLE_VALUES, state='readonly')
        input_widgets[col_name] = var
        note_text = HOUSING_NOTES
    elif col_name == 'Saving accounts':
        var = tk.StringVar(root)
        var.set(SAVING_ACCOUNTS_POSSIBLE_VALUES[0] if SAVING_ACCOUNTS_POSSIBLE_VALUES else "Lỗi giá trị Saving accounts")
        widget = ttk.Combobox(input_frame, textvariable=var, values=SAVING_ACCOUNTS_POSSIBLE_VALUES, state='readonly')
        input_widgets[col_name] = var
        note_text = SAVING_ACCOUNTS_NOTES
    elif col_name == 'Sex':
        var = tk.StringVar(root)
        var.set(SEX_POSSIBLE_VALUES[0] if SEX_POSSIBLE_VALUES else "Lỗi giá trị Sex")
        widget = ttk.Combobox(input_frame, textvariable=var, values=SEX_POSSIBLE_VALUES, state='readonly')
        input_widgets[col_name] = var
        note_text = SEX_NOTES
    elif col_name == 'Purpose':
        var = tk.StringVar(root)
        var.set(PURPOSE_POSSIBLE_VALUES[0] if PURPOSE_POSSIBLE_VALUES else "Lỗi giá trị Purpose")
        widget = ttk.Combobox(input_frame, textvariable=var, values=PURPOSE_POSSIBLE_VALUES, state='readonly')
        input_widgets[col_name] = var
        note_text = PURPOSE_NOTES
    elif col_name == 'Checking account':
        var = tk.StringVar(root)
        var.set(CHECKING_ACCOUNT_POSSIBLE_VALUES[0] if CHECKING_ACCOUNT_POSSIBLE_VALUES else "Lỗi giá trị Checking account")
        widget = ttk.Combobox(input_frame, textvariable=var, values=CHECKING_ACCOUNT_POSSIBLE_VALUES, state='readonly')
        input_widgets[col_name] = var
        note_text = CHECKING_ACCOUNT_NOTES
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
        for col in CATEGORICAL_FEATURE_COLUMNS:
            if col in input_df.columns and col in label_encoders:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col])
                except ValueError as e:
                    messagebox.showerror("Lỗi Tiền Xử Lý", f"Giá trị không hợp lệ cho cột '{col}': {e}")
                    return None
            elif col in input_df.columns:
                print(f"Cảnh báo: Không tìm thấy label encoder cho cột '{col}'.")

        # Chuẩn hóa các thuộc tính số
        numerical_input = input_df[NUMERICAL_FEATURE_COLUMNS]
        if scaler is not None and not numerical_input.empty:
            input_df[NUMERICAL_FEATURE_COLUMNS] = scaler.transform(numerical_input)

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
            input_data[col_name] = input_widgets[col_name].get()
        prediction_result_text = predict_credit_risk(input_data)
        result_label.config(text=f"Kết quả dự đoán: {prediction_result_text}")
    except Exception as e:
        messagebox.showerror("Lỗi Chung", f"Đã xảy ra lỗi: {e}")

# --- Hàm xử lý khi nhấn nút Huấn luyện Mô hình ---
def on_train_button_click():
    from train_xgboost import train_and_evaluate_xgboost

    messagebox.showinfo("Bắt đầu Huấn luyện", "Quá trình huấn luyện mô hình XGBoost đang được tiến hành...")
    evaluation_metrics = train_and_evaluate_xgboost(DATA_PATH)
    if evaluation_metrics:
        accuracy = evaluation_metrics['accuracy']
        conf_matrix_str = np.array_str(evaluation_metrics['confusion_matrix'])
        class_report_str = evaluation_metrics['classification_report']

        # Tạo một cửa sổ mới để hiển thị kết quả đánh giá
        evaluation_window = Toplevel(root)
        evaluation_window.title("Kết quả Đánh giá Mô hình XGBoost")

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

        # Sau khi huấn luyện lại, thử tải lại mô hình và các đối tượng
        global model, label_encoders, target_le, scaler, FEATURE_COLUMNS_ORDER, CATEGORICAL_FEATURE_COLUMNS, NUMERICAL_FEATURE_COLUMNS, fillna_value
        try:
            model = joblib.load(MODEL_PATH)
            label_encoders = joblib.load(LABEL_ENCODERS_PATH)
            target_le = joblib.load(TARGET_ENCODER_PATH)
            scaler = joblib.load(SCALER_PATH)
            FEATURE_COLUMNS_ORDER = joblib.load(FEATURE_COLUMNS_ORDER_PATH)
            CATEGORICAL_FEATURE_COLUMNS = joblib.load(CATEGORICAL_FEATURE_COLUMNS_PATH)
            NUMERICAL_FEATURE_COLUMNS = joblib.load(NUMERICAL_FEATURE_COLUMNS_PATH)
            fillna_value = joblib.load(FILLNA_VALUE_PATH)
            print(f"Đã tải lại mô hình XGBoost và các đối tượng tiền xử lý sau khi huấn luyện. Giá trị fillna: {fillna_value}")
        except FileNotFoundError as e:
            messagebox.showerror("Lỗi Tải Lại Mô hình", f"Không tìm thấy file .pkl sau huấn luyện: {e.filename}.")
        except Exception as e:
            messagebox.showerror("Lỗi Tải Lại Mô hình", f"Lỗi khi tải lại file .pkl sau huấn luyện: {e}")

    else:
        messagebox.showerror("Lỗi Huấn luyện", "Đã xảy ra lỗi trong quá trình huấn luyện mô hình XGBoost.")

# --- Tạo nút Dự đoán ---
predict_button = ttk.Button(root, text="Dự đoán Rủi ro Tín dụng", command=on_predict_button_click)
predict_button.grid(row=row_index, column=0, columnspan=3, pady=10)

# --- Label hiển thị kết quả dự đoán ---
result_label = ttk.Label(root, text="Kết quả dự đoán: Chờ nhập liệu...", font=('TkDefaultFont', 10, 'bold'))
result_label.grid(row=row_index + 1, column=0, columnspan=3, pady=10)

# --- Tạo nút Huấn luyện Mô hình ---
train_button = ttk.Button(root, text="Huấn luyện lại Mô hình & Xem Đánh giá", command=on_train_button_click)
train_button.grid(row=row_index + 2, column=0, columnspan=3, pady=10)

# --- Chạy vòng lặp chính của giao diện ---
root.mainloop()
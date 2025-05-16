# predict_risk.py

# 1. Import Libraries
import pandas as pd
import numpy as np
import joblib # Thư viện để lưu và tải mô hình/đối tượng
import sys
import os

# Cần import các class này để joblib có thể tải các đối tượng
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


# --- Tên file mô hình và các đối tượng tiền xử lý đã lưu ---
MODEL_FILENAME = 'credit_risk_model.pkl'
SCALER_FILENAME = 'scaler.pkl'
ENCODERS_FILENAME = 'label_encoders.pkl' # Chứa encoders cho các cột đặc trưng phân loại
TARGET_ENCODER_FILENAME = 'target_encoder.pkl' # Lưu encoder cho cột mục tiêu
FEATURE_COLUMNS_ORDER_FILENAME = 'feature_cols_order.pkl' # Lưu thứ tự cột
CATEGORICAL_FEATURE_COLUMNS_USED_FILENAME = 'categorical_features_used.pkl' # Tên file lưu tên cột phân loại đã dùng
NUMERIC_FEATURE_COLUMNS_USED_FILENAME = 'numeric_features_used.pkl' # Tên file lưu tên cột số đã dùng
FILLNA_VALUE_FILENAME = 'fillna_value.pkl' # Tên file lưu giá trị fillna


# --- Tải Mô hình và các Đối tượng tiền xử lý ---
# Việc tải này sẽ chạy khi file được import hoặc chạy trực tiếp
model = None
scaler = None
label_encoders = None
target_le = None
FEATURE_COLUMNS_ORDER = None
CATEGORICAL_FEATURE_COLUMNS_USED = None
NUMERIC_FEATURE_COLUMNS_USED = None
FILLNA_VALUE = None


try:
    model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
    label_encoders = joblib.load(ENCODERS_FILENAME)
    target_le = joblib.load(TARGET_ENCODER_FILENAME)
    FEATURE_COLUMNS_ORDER = joblib.load(FEATURE_COLUMNS_ORDER_FILENAME)
    CATEGORICAL_FEATURE_COLUMNS_USED = joblib.load(CATEGORICAL_FEATURE_COLUMNS_USED_FILENAME)
    NUMERIC_FEATURE_COLUMNS_USED = joblib.load(NUMERIC_FEATURE_COLUMNS_USED_FILENAME)
    FILLNA_VALUE = joblib.load(FILLNA_VALUE_FILENAME)

    print("Đã tải mô hình và các đối tượng tiền xử lý thành công.")

except FileNotFoundError as e:
    print(f"Lỗi: Không tìm thấy file cần thiết: {e}")
    print("Vui lòng đảm bảo các file (.pkl) nằm trong cùng thư mục với predict_risk.py")
    print("Và bạn đã chạy train_model.py bản cập nhật trước đó để tạo ra chúng.")
    # Không thoát hẳn process nếu chỉ import, chỉ báo lỗi và các biến sẽ là None


# --- Hàm tiền xử lý dữ liệu đầu vào mới ---
def preprocess_input(data: dict, scaler, label_encoders, feature_cols_order: list,
                       categorical_cols_used: list, numeric_cols_used: list, fillna_value) -> np.ndarray | None:
    """
    Tiền xử lý dữ liệu đầu vào mới cho mô hình, áp dụng các biến đổi tương tự lúc train.
    Args:
        data (dict): Dữ liệu của một cá nhân dưới dạng dictionary {tên_cột: giá_trị}.
                     Các tên cột phải khớp với CÁC CỘT ĐẶC TRƯNG dùng trong X lúc huấn luyện.
        scaler: Đối tượng StandardScaler đã fit.
        label_encoders: Dictionary chứa các LabelEncoder đã fit cho từng cột phân loại đặc trưng.
        feature_cols_order (list): Thứ tự chính xác tên các cột đặc trưng mà mô hình đã huấn luyện.
        categorical_cols_used (list): Tên các cột phân loại đã thực sự dùng làm đặc trưng.
        numeric_cols_used (list): Tên các cột số đã thực sự dùng làm đặc trưng.
        fillna_value: Giá trị được sử dụng để điền vào các giá trị thiếu lúc train.
    Returns:
        np.ndarray | None: Dữ liệu đã được tiền xử lý, sẵn sàng cho mô hình dự đoán, hoặc None nếu lỗi.
    """
    if scaler is None or label_encoders is None or feature_cols_order is None or categorical_cols_used is None or numeric_cols_used is None:
         print("Lỗi tiền xử lý: Các đối tượng scaler, encoders, hoặc thông tin cột chưa được tải.")
         return None

    # Chuyển dictionary thành DataFrame với thứ tự cột đảm bảo khớp
    # Tạo DataFrame với tất cả các cột có thể có trong input (cả số và phân loại dùng làm đặc trưng)
    # Sau đó chọn lọc và sắp xếp lại theo feature_cols_order
    all_input_cols_potential = list(data.keys())
    df_input = pd.DataFrame([data], columns=all_input_cols_potential) # Tạo DF từ dict

    # --- Áp dụng tiền xử lý giống như trong train_model.py ---

    # Xử lý giá trị thiếu (fillna) - Chỉ cho các cột đã xử lý lúc train
    cols_to_fillna = ['Saving accounts', 'Checking account'] # Các cột đã fillna lúc train
    for col in cols_to_fillna:
        if col in df_input.columns: # Kiểm tra cột có trong input không
             # Điền giá trị thiếu bằng giá trị đã lưu từ lúc train
             df_input[col] = df_input[col].fillna(fillna_value)


    # Mã hóa biến phân loại sử dụng các encoders đã tải
    for col in categorical_cols_used: # Chỉ lặp qua các cột phân loại đã thực sự dùng làm đặc trưng
        if col in df_input.columns and col in label_encoders: # Kiểm tra cột có trong input và có encoder
             le = label_encoders[col]
             # Xử lý giá trị chưa từng thấy lúc train:
             # Nếu giá trị input KHÔNG nằm trong các lớp mà encoder đã học
             if df_input[col].iloc[0] not in le.classes_:
                  print(f"Cảnh báo tiền xử lý: Giá trị '{df_input[col].iloc[0]}' trong cột '{col}' chưa thấy lúc huấn luyện.")
                  # Cách xử lý: Thay thế bằng giá trị mã hóa của lớp "unknown" nếu có
                  if fillna_value in le.classes_:
                       safe_encoded_value = le.transform([fillna_value])[0]
                       print(f"Thay thế bằng giá trị mã hóa của '{fillna_value}' ({safe_encoded_value}).")
                       df_input[col] = safe_encoded_value
                  else:
                       # Nếu không có 'unknown' hoặc fillna_value khác, thay bằng giá trị mã hóa của lớp đầu tiên (index 0)
                       print(f"Thay thế bằng giá trị mã hóa của lớp đầu tiên ({le.classes_[0]} -> 0).")
                       df_input[col] = 0 # Giá trị mã hóa của lớp đầu tiên (thường là 0)
             else:
                  # Nếu giá trị nằm trong các lớp đã thấy, áp dụng transform bình thường
                  df_input[col] = le.transform(df_input[col]) # <<< Đã sửa lỗi: dùng df_input[col]

        elif col in CATEGORICAL_FEATURE_COLUMNS_USED and col not in df_input.columns:
             print(f"Lỗi tiền xử lý: Thiếu cột đặc trưng phân loại '{col}' trong dữ liệu đầu vào.")
             return None # Trả về None nếu thiếu cột đặc trưng


    # --- Kiểm tra và sắp xếp lại cột đặc trưng cuối cùng ---
    # Cần đảm bảo df_input chứa CHÍNH XÁC các cột trong feature_cols_order và theo đúng thứ tự
    # Đảm bảo tất cả các cột trong feature_cols_order đều có trong df_input
    missing_cols = [col for col in feature_cols_order if col not in df_input.columns]
    if missing_cols:
         print(f"Lỗi tiền xử lý: Dữ liệu đầu vào thiếu các cột đặc trưng bắt buộc: {missing_cols}")
         return None

    # Bỏ đi các cột thừa trong df_input không có trong feature_cols_order
    extra_cols = [col for col in df_input.columns if col not in feature_cols_order]
    if extra_cols:
         print(f"Cảnh báo tiền xử lý: Dữ liệu đầu vào có các cột thừa ({extra_cols}) sẽ bị bỏ qua.")
         df_input = df_input.drop(columns=extra_cols)

    # Sắp xếp cột theo đúng thứ tự huấn luyện cuối cùng
    df_input = df_input[feature_cols_order]

    # Chuẩn hóa đặc trưng số sử dụng scaler đã tải
    # Chỉ chuẩn hóa các cột số đã xác định lúc train
    # Cần đảm bảo các cột số này tồn tại trong df_input sau khi sắp xếp và mã hóa
    numeric_cols_in_input = [col for col in numeric_cols_used if col in df_input.columns]
    if numeric_cols_in_input: # Chỉ chuẩn hóa nếu có cột số
        # scaler.transform mong đợi input là 2D array hoặc DataFrame
        df_input[numeric_cols_in_input] = scaler.transform(df_input[numeric_cols_in_input])
    # else: print("Không có cột số nào để chuẩn hóa trong input.") # Debug


    return df_input.values # Trả về dưới dạng numpy array


# --- Hàm Dự đoán Rủi ro Tín dụng ---
def predict_credit_risk(data: dict) -> str:
    """
    Dự đoán rủi ro tín dụng cho một cá nhân.
    Args:
        data (dict): Dữ liệu của một cá nhân dưới dạng dictionary {tên_cột: giá_trị}.
                     Các tên cột phải khớp với CÁC CỘT ĐẶC TRƯNG dùng trong X lúc huấn luyện.
    Returns:
        str: Kết quả dự đoán rủi ro tín dụng ('Good' hoặc 'Bad') hoặc thông báo lỗi.
    """
    # Kiểm tra xem mô hình và scaler đã được tải thành công chưa
    if model is None or scaler is None or label_encoders is None or target_le is None or FEATURE_COLUMNS_ORDER is None or CATEGORICAL_FEATURE_COLUMNS_USED is None or NUMERIC_FEATURE_COLUMNS_USED is None or FILLNA_VALUE is None:
        return "Lỗi: Mô hình hoặc các đối tượng tiền xử lý chưa được tải. Vui lòng kiểm tra file .pkl."

    # Tiền xử lý dữ liệu đầu vào
    processed_data = preprocess_input(data, scaler, label_encoders, FEATURE_COLUMNS_ORDER,
                                       CATEGORICAL_FEATURE_COLUMNS_USED, NUMERIC_FEATURE_COLUMNS_USED, FILLNA_VALUE)

    if processed_data is None:
        return "Không thể dự đoán do lỗi tiền xử lý dữ liệu đầu vào."

    # Dự đoán (kết quả là 0 hoặc 1)
    # model.predict mong đợi input là 2D array/matrix, ngay cả khi chỉ có 1 mẫu
    prediction_encoded = model.predict(processed_data)

    # Giải mã kết quả dự đoán từ 0/1 về lại nhãn gốc ('Bad'/'Good')
    try:
        prediction_label = target_le.inverse_transform(prediction_encoded)
        # Trả về nhãn gốc
        return str(prediction_label[0]) # Lấy giá trị string từ mảng numpy
    except Exception as e:
         print(f"Lỗi giải mã kết quả dự đoán ({prediction_encoded[0]}): {e}")
         # Trả về giá trị mã hóa nếu giải mã lỗi
         return f"Kết quả mã hóa: {prediction_encoded[0]} (Lỗi giải mã)"


# --- Ví dụ sử dụng (chỉ chạy khi file predict_risk.py được chạy trực tiếp) ---
if __name__ == "__main__":
     print("\n--- Chạy predict_risk.py trực tiếp ---")
     # Kiểm tra xem mô hình và scaler đã được tải thành công chưa trước khi chạy ví dụ
     if model is None:
          print("Không thể chạy ví dụ vì mô hình hoặc các file .pkl chưa được tải.")
     else:
         print("Chạy ví dụ dự đoán...")
         # Tạo dữ liệu input cho một cá nhân mới
         # Đảm bảo các cột và kiểu dữ liệu khớp với CÁC CỘT ĐẶC TRƯNG dùng trong X lúc huấn luyện (FEATURE_COLUMNS_ORDER)
         # Các cột cần có trong dictionary input (tên cột gốc):
         # 'Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account',
         # 'Duration', 'Purpose', 'Installment rate in percentage of disposable income',
         # 'Present residence since', 'Number of existing credits at this bank',
         # 'Number of people being liable to provide maintenance for', 'Telephone', 'Foreign Worker'

         sample_person_features = {
             'Age': 30,
             'Sex': 'male',
             'Job': '2', # Giá trị phải là 0, 1, 2, 3 (hoặc string tương ứng nếu gốc là string)
             'Housing': 'own',
             'Saving accounts': 'little',
             'Checking account': 'moderate',
             'Duration': 24, # Tháng
             'Purpose': 'car',
             'Installment rate in percentage of disposable income': 3,
             'Present residence since': 2,
             'Number of existing credits at this bank': 1,
             'Number of people being liable to provide maintenance for': 1,
             'Telephone': 'none', # Giá trị gốc 'yes' hoặc 'none'
             'Foreign Worker': 'yes' # Giá trị gốc 'yes' hoặc 'no'
         }

         prediction_result = predict_credit_risk(sample_person_features)
         print(f"\nDữ liệu đầu vào ví dụ 1: {sample_person_features}")
         print(f"Kết quả dự đoán rủi ro tín dụng: {prediction_result}")

         # Thêm một ví dụ khác
         sample_person_features_2 = {
             'Age': 55,
             'Sex': 'female',
             'Job': '1', # Giá trị phải là 0, 1, 2, 3
             'Housing': 'rent',
             'Saving accounts': 'unknown', # Giá trị thiếu được điền lúc train
             'Checking account': 'little',
             'Duration': 48,
             'Purpose': 'furniture/equipment',
             'Installment rate in percentage of disposable income': 4,
             'Present residence since': 4,
             'Number of existing credits at this bank': 2,
             'Number of people being liable to provide maintenance for': 1,
             'Telephone': 'yes',
             'Foreign Worker': 'yes'
         }

         prediction_result_2 = predict_credit_risk(sample_person_features_2)
         print(f"\n--- Ví dụ Dự đoán 2 ---")
         print(f"Dữ liệu đầu vào ví dụ 2: {sample_person_features_2}")
         print(f"Kết quả dự đoán rủi ro tín dụng: {prediction_result_2}")

         # Ví dụ với giá trị chưa thấy (sẽ bị xử lý) hoặc thiếu cột
         sample_person_features_3 = {
             'Age': 25,
             'Sex': 'other', # Giá trị mới chưa thấy
             'Job': '3',
             'Housing': 'own',
             'Saving accounts': 'very rich', # Giá trị mới chưa thấy
             'Checking account': 'super rich', # Giá trị mới chưa thấy
             'Duration': 12,
             'Purpose': 'new car', # Giá trị mới chưa thấy
             'Installment rate in percentage of disposable income': 2,
             'Present residence since': 1,
             'Number of existing credits at this bank': 1,
             'Number of people being liable to provide maintenance for': 1,
             'Telephone': 'none',
             'Foreign Worker': 'no',
             # Thiếu cột nào đó nếu cố tình bỏ đi
         }
         print(f"\n--- Ví dụ Dự đoán 3 (với giá trị mới) ---")
         print(f"Dữ liệu đầu vào ví dụ 3: {sample_person_features_3}")
         prediction_result_3 = predict_credit_risk(sample_person_features_3)
         print(f"Kết quả dự đoán rủi ro tín dụng: {prediction_result_3}")
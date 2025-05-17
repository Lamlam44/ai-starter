# train_model.py

# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib # Thư viện để lưu và tải mô hình/đối tượng
import sys # <<< Đã thêm import sys
import os # <<< Thêm thư viện os để làm việc với hệ điều hành

# 2. Load Data
# Thay đổi đường dẫn này đến file german_credit_data.csv trên máy của bạn
DATA_PATH = 'src/german_credit_data (1).csv'
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Đã tải dữ liệu thành công từ: {DATA_PATH}")
    print("\n--- Tên các cột trong file CSV ---")
    print(df.columns.tolist()) # In ra tên tất cả các cột để kiểm tra
    print("----------------------------------")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file dữ liệu tại {DATA_PATH}. Vui lòng kiểm tra lại đường dẫn DATA_PATH.")
    sys.exit(1) # Thoát nếu không tìm thấy file

# --- Định nghĩa tên cột mục tiêu và các cột đặc trưng ---
TARGET_COLUMN_NAME = 'Risk' # Tên cột rủi ro tín dụng thực tế (ví dụ: 'Risk')
# Các cột phân loại sẽ được mã hóa và dùng làm đặc trưng. Cần liệt kê TẤT CẢ cột phân loại bạn muốn dùng.
CATEGORICAL_FEATURE_COLUMNS = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Telephone', 'Foreign Worker'] # Thêm các cột phân loại khác nếu có và cần dùng

# Các cột cần bỏ đi khỏi features X. Luôn bỏ cột Credit amount và cột mục tiêu.
# Cột không tên (Unnamed: 0) cũng cần bỏ nếu có.
COLUMNS_TO_DROP_FROM_X = ['Credit amount', TARGET_COLUMN_NAME]

# Kiểm tra sự tồn tại của cột không tên và thêm vào danh sách drop nếu có
# Cột này thường không có header trong CSV gốc và pandas đặt tên Unnamed: 0
unnamed_col = None
for col in df.columns:
    if col.startswith('Unnamed:'): # Kiểm tra tên cột bắt đầu bằng 'Unnamed:'
        unnamed_col = col
        break # Chỉ lấy cột đầu tiên không tên

if unnamed_col:
    COLUMNS_TO_DROP_FROM_X.append(unnamed_col)
    print(f"Đã phát hiện và sẽ bỏ cột '{unnamed_col}'.")
else:
     print("Không tìm thấy cột bắt đầu bằng 'Unnamed:'.")


# Kiểm tra sự tồn tại của các cột quan trọng
# Bao gồm cột mục tiêu, các cột phân loại đã liệt kê, và các cột khác còn lại sau khi drop mà không phải là số
# (để chắc chắn không thiếu cột nào mà bạn muốn dùng làm đặc trưng)
# cols_after_drop_check = [col for col in df.columns if col not in COLUMNS_TO_DROP_FROM_X and col != TARGET_COLUMN_NAME]
# required_cols = [TARGET_COLUMN_NAME] + [col for col in CATEGORICAL_FEATURE_COLUMNS if col in df.columns] + [col for col in cols_after_drop_check if col not in CATEGORICAL_FEATURE_COLUMNS]

# Cách kiểm tra đơn giản hơn: Đảm bảo cột mục tiêu và các cột phân loại được liệt kê đều có trong df
if TARGET_COLUMN_NAME not in df.columns:
     print(f"\nLỗi: Không tìm thấy cột mục tiêu '{TARGET_COLUMN_NAME}' trong file CSV. Vui lòng kiểm tra lại tên cột.")
     sys.exit(1)

for col in CATEGORICAL_FEATURE_COLUMNS:
    if col not in df.columns:
         print(f"\nCảnh báo: Không tìm thấy cột phân loại '{col}' trong file CSV. Cột này sẽ không được sử dụng làm đặc trưng.")
         # Loại bỏ cột này khỏi danh sách để xử lý tiếp
         CATEGORICAL_FEATURE_COLUMNS.remove(col)


# 3. Data Preprocessing
print("\nBắt đầu tiền xử lý dữ liệu...")

# Xử lý giá trị thiếu (fillna) - Chỉ xử lý cho các cột được liệt kê và có thể có giá trị thiếu
# Các cột thường có NaN trong dataset này là 'Saving accounts', 'Checking account'.
# Nếu có cột khác, thêm vào đây.
cols_to_fillna = ['Saving accounts', 'Checking account']
# Lọc chỉ lấy các cột thực sự tồn tại trong df
cols_to_fillna_existing = [col for col in cols_to_fillna if col in df.columns]

fillna_value = 'unknown' # Giá trị sẽ dùng để điền NaN

for col in cols_to_fillna_existing:
    df[col] = df[col].fillna(fillna_value)
    # print(f"Đã điền NaN cho cột '{col}' bằng '{fillna_value}'.") # Debug

# *** Mã hóa CỘT MỤC TIÊU ***
# Giả định 'Bad' thường được mã hóa thành 0 và 'Good' thành 1 bởi LabelEncoder theo thứ tự bảng chữ cái.
# target_le.classes_ sẽ cho biết thứ tự đó sau fit.
target_le = LabelEncoder()
df[TARGET_COLUMN_NAME] = target_le.fit_transform(df[TARGET_COLUMN_NAME])
print(f"Đã mã hóa cột mục tiêu '{TARGET_COLUMN_NAME}'. Các lớp gốc: {target_le.classes_}, mã hóa thành: {target_le.transform(target_le.classes_)}")


# *** Mã hóa CÁC CỘT ĐẶC TRƯNG PHÂN LOẠI ***
label_encoders = {} # Lưu encoders cho các cột đặc trưng phân loại
# Chỉ lặp qua các cột phân loại thực sự tồn tại và được sử dụng
CATEGORICAL_FEATURE_COLUMNS_USED = [col for col in CATEGORICAL_FEATURE_COLUMNS if col in df.columns]

for col in CATEGORICAL_FEATURE_COLUMNS_USED:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le # Lưu encoder đặc trưng
    # print(f"Đã mã hóa cột đặc trưng '{col}'. Lớp gốc: {le.classes_}")


# Xử lý giá trị vô cùng nếu có (ít khả năng xảy ra sau fillna và encode)
df = df.replace([np.inf, -np.inf], np.nan)
# Cân nhắc xử lý NaN còn sót lại nếu có (ví dụ: cột số có NaN)
# print(f"Số lượng NaN còn sót lại: {df.isnull().sum().sum()}") # Debug
# df.dropna(inplace=True) # Nếu muốn xóa các dòng còn NaN
# Hoặc điền NaN cho các cột số nếu có:
# for col in df.select_dtypes(include=np.number).columns:
#     if df[col].isnull().any():
#         df[col] = df[col].fillna(df[col].mean()) # Hoặc median, mode, 0, ...


print("Hoàn tất tiền xử lý dữ liệu.")

# 4. Định nghĩa Đặc trưng (X) và Biến Mục tiêu (y)

# Bỏ các cột không cần thiết khỏi tập đặc trưng X
# Đảm bảo các cột trong COLUMNS_TO_DROP_FROM_X tồn tại trong df trước khi drop
cols_existing_to_drop = [col for col in COLUMNS_TO_DROP_FROM_X if col in df.columns]
X = df.drop(columns=cols_existing_to_drop)


# Biến mục tiêu y là cột mục tiêu đã được mã hóa (0/1)
y = df[TARGET_COLUMN_NAME]

# Xác định các cột số cuối cùng trong X sau khi drop và mã hóa phân loại
numeric_cols_in_X = X.select_dtypes(include=np.number).columns.tolist()

print(f"Số lượng đặc trưng (X) sau khi drop và mã hóa: {X.shape[1]}")
print(f"Tên các đặc trưng cuối cùng được sử dụng:\n{X.columns.tolist()}")
print(f"Các cột phân loại đã dùng: {CATEGORICAL_FEATURE_COLUMNS_USED}")
print(f"Các cột số đã dùng: {numeric_cols_in_X}")
print(f"Số lượng mẫu: {X.shape[0]}")
print(f"Cột mục tiêu (y): {TARGET_COLUMN_NAME} (đã mã hóa)")

# Lưu lại thứ tự tên cột đặc trưng để sử dụng trong file dự đoán
FEATURE_COLUMNS_ORDER = X.columns.tolist()


# 5. Chia tập dữ liệu
# Sử dụng stratify=y để giữ tỷ lệ các lớp mục tiêu trong tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# 6. Chuẩn hóa đặc trưng số
# X_train và X_test đã được tạo ở bước 5.
# numeric_cols_in_X đã được xác định ở bước 4.

scaler = StandardScaler()
# Chỉ áp dụng scaler lên các cột số trong X_train và X_test
X_train[numeric_cols_in_X] = scaler.fit_transform(X_train[numeric_cols_in_X])
X_test[numeric_cols_in_X] = scaler.transform(X_test[numeric_cols_in_X])

print("Đã chia tập dữ liệu và chuẩn hóa đặc trưng.")

# 7. Huấn luyện Mô hình
print("Bắt đầu huấn luyện mô hình Random Forest...")
# <<< Dòng này sử dụng X_train và y_train (có gạch dưới) >>>
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Huấn luyện mô hình hoàn tất.")

# 8. Đánh giá Mô hình trên tập Test
print("\n--- Đánh giá mô hình trên tập Test ---")
# <<< Các dòng này sử dụng X_test và y_test (có gạch dưới) >>>
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
print("--------------------------------------")


# 9. Lưu Mô hình và các Đối tượng tiền xử lý
OUTPUT_DIR = 'src/RandomForest/saved_models' # Tên thư mục bạn muốn tạo
os.makedirs(OUTPUT_DIR, exist_ok=True) # Tạo thư mục nếu nó chưa tồn tại

MODEL_FILENAME = os.path.join(OUTPUT_DIR, 'credit_risk_model.pkl')
SCALER_FILENAME = os.path.join(OUTPUT_DIR, 'scaler.pkl')
ENCODERS_FILENAME = os.path.join(OUTPUT_DIR, 'label_encoders.pkl') # Chứa encoders cho các cột đặc trưng phân loại
TARGET_ENCODER_FILENAME = os.path.join(OUTPUT_DIR, 'target_encoder.pkl') # Lưu encoder cho cột mục tiêu
FEATURE_COLUMNS_ORDER_FILENAME = os.path.join(OUTPUT_DIR, 'feature_cols_order.pkl') # Lưu thứ tự cột
CATEGORICAL_FEATURE_COLUMNS_USED_FILENAME = os.path.join(OUTPUT_DIR, 'categorical_features_used.pkl') # Lưu tên các cột phân loại đã dùng
NUMERIC_FEATURE_COLUMNS_USED_FILENAME = os.path.join(OUTPUT_DIR, 'numeric_features_used.pkl') # Lưu tên các cột số đã dùng
FILLNA_VALUE_FILENAME = os.path.join(OUTPUT_DIR, 'fillna_value.pkl') # Lưu giá trị fillna (nếu cố định)


joblib.dump(model, MODEL_FILENAME)
joblib.dump(scaler, SCALER_FILENAME)
joblib.dump(label_encoders, ENCODERS_FILENAME)
joblib.dump(target_le, TARGET_ENCODER_FILENAME)
joblib.dump(FEATURE_COLUMNS_ORDER, FEATURE_COLUMNS_ORDER_FILENAME)
joblib.dump(CATEGORICAL_FEATURE_COLUMNS_USED, CATEGORICAL_FEATURE_COLUMNS_USED_FILENAME)
joblib.dump(numeric_cols_in_X, NUMERIC_FEATURE_COLUMNS_USED_FILENAME) # Lưu tên cột số đã dùng
joblib.dump(fillna_value, FILLNA_VALUE_FILENAME) # Lưu giá trị fillna


print(f"\nĐã lưu mô hình vào {MODEL_FILENAME}")
print(f"Đã lưu scaler vào {SCALER_FILENAME}")
print(f"Đã lưu label encoders đặc trưng vào {ENCODERS_FILENAME}")
print(f"Đã lưu target encoder vào {TARGET_ENCODER_FILENAME}")
print(f"Đã lưu thứ tự cột đặc trưng vào {FEATURE_COLUMNS_ORDER_FILENAME}")
print(f"Đã lưu danh sách cột phân loại đã dùng vào {CATEGORICAL_FEATURE_COLUMNS_USED_FILENAME}")
print(f"Đã lưu danh sách cột số đã dùng vào {NUMERIC_FEATURE_COLUMNS_USED_FILENAME}")
print(f"Đã lưu giá trị fillna vào {FILLNA_VALUE_FILENAME}")


print("\nQuá trình huấn luyện và lưu mô hình hoàn tất.")
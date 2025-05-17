# train_xgboost.py

# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

def train_and_evaluate_xgboost(DATA_PATH):
    """
    Huấn luyện và đánh giá mô hình XGBoost, trả về các chỉ số đánh giá.

    Args:
        DATA_PATH (str): Đường dẫn đến file CSV của tập dữ liệu.

    Returns:
        dict or None: Dictionary chứa các chỉ số đánh giá (accuracy, confusion matrix,
                        classification report) hoặc None nếu có lỗi.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Đã tải dữ liệu thành công từ: {DATA_PATH}")

        TARGET_COLUMN_NAME = 'Risk'
        CATEGORICAL_FEATURE_COLUMNS = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
        NUMERICAL_FEATURE_COLUMNS = ['Age', 'Credit amount', 'Duration']
        FEATURE_COLUMNS = NUMERICAL_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS

        # Xử lý cột unnamed (số thứ tự)
        unnamed_col = None
        for col in df.columns:
            if col.startswith('Unnamed:'):
                unnamed_col = col
                break
        if unnamed_col in FEATURE_COLUMNS:
            FEATURE_COLUMNS.remove(unnamed_col)
            print(f"Đã phát hiện và sẽ bỏ cột '{unnamed_col}' khỏi danh sách thuộc tính.")
        elif unnamed_col:
            print(f"Đã phát hiện cột '{unnamed_col}' nhưng không nằm trong danh sách thuộc tính.")
        else:
            print("Không tìm thấy cột bắt đầu bằng 'Unnamed:'.")

        if TARGET_COLUMN_NAME not in df.columns:
            print(f"\nLỗi: Không tìm thấy cột mục tiêu '{TARGET_COLUMN_NAME}'.")
            return None

        # Thay thế NA bằng 'unknown' cho các thuộc tính phân loại
        for col in CATEGORICAL_FEATURE_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')

        # Mã hóa cột mục tiêu
        target_le = LabelEncoder()
        df[TARGET_COLUMN_NAME] = target_le.fit_transform(df[TARGET_COLUMN_NAME])
        print(f"Đã mã hóa cột mục tiêu '{TARGET_COLUMN_NAME}'. Các lớp gốc: {target_le.classes_}, mã hóa thành: {target_le.transform(target_le.classes_)}")

        # Mã hóa các thuộc tính phân loại
        label_encoders = {}
        for col in CATEGORICAL_FEATURE_COLUMNS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN_NAME]) # Loại bỏ các hàng có NA trong thuộc tính hoặc target sau xử lý

        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN_NAME]

        # Chuẩn hóa các thuộc tính số
        scaler = StandardScaler()
        X[NUMERICAL_FEATURE_COLUMNS] = scaler.fit_transform(X[NUMERICAL_FEATURE_COLUMNS])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        print("Bắt đầu huấn luyện mô hình XGBoost...")
        model = XGBClassifier(random_state=42) # Bạn có thể tinh chỉnh các siêu tham số
        model.fit(X_train, y_train)
        print("Huấn luyện mô hình hoàn tất.")

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=target_le.classes_)

        print("\n--- Đánh giá mô hình trên tập Test ---")
        print(f"Accuracy: {accuracy:.2f}")
        print("Confusion Matrix:\n{conf_matrix}")
        print("Classification Report:\n{class_report}")
        print("--------------------------------------")

        OUTPUT_DIR = 'src/XGBoost/saved_models'
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(OUTPUT_DIR, 'xgboost_model.pkl'))
        joblib.dump(label_encoders, os.path.join(OUTPUT_DIR, 'label_encoders_xgboost.pkl'))
        joblib.dump(target_le, os.path.join(OUTPUT_DIR, 'target_encoder_xgboost.pkl'))
        joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler_xgboost.pkl'))
        joblib.dump(FEATURE_COLUMNS, os.path.join(OUTPUT_DIR, 'feature_cols_order_xgboost.pkl'))
        joblib.dump(CATEGORICAL_FEATURE_COLUMNS, os.path.join(OUTPUT_DIR, 'categorical_features_xgboost.pkl'))
        joblib.dump(NUMERICAL_FEATURE_COLUMNS, os.path.join(OUTPUT_DIR, 'numerical_features_xgboost.pkl'))
        joblib.dump('unknown', os.path.join(OUTPUT_DIR, 'fillna_value_xgboost.pkl')) # Lưu 'unknown' làm giá trị fillna

        LOG_DIR = 'logs'
        os.makedirs(LOG_DIR, exist_ok=True)
        LOG_FILENAME = os.path.join(LOG_DIR, 'training_log_xgboost.txt')

        with open(LOG_FILENAME, 'w', encoding='utf-8') as f:
            f.write("--- Báo cáo Huấn luyện Mô hình XGBoost ---\n\n")
            f.write(f"Accuracy: {accuracy:.2f}\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array_str(conf_matrix) + "\n")
            f.write("Classification Report:\n")
            f.write(class_report + "\n")

        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {DATA_PATH}.")
        return None
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình huấn luyện: {e}")
        return None

if __name__ == "__main__":
    DATA_PATH = 'src/german_credit_data (1).csv'
    evaluation_metrics = train_and_evaluate_xgboost(DATA_PATH)
    if evaluation_metrics:
        print("\n--- Các chỉ số đánh giá trả về ---")
        print(f"Accuracy: {evaluation_metrics['accuracy']:.2f}")
        print("Confusion Matrix:\n", evaluation_metrics['confusion_matrix'])
        print("Classification Report:\n", evaluation_metrics['classification_report'])
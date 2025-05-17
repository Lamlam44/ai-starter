# train_logistic_regression.py

# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

def train_and_evaluate_logistic_regression(DATA_PATH):
    """
    Huấn luyện và đánh giá mô hình Logistic Regression, trả về các chỉ số đánh giá.

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
        CATEGORICAL_FEATURE_COLUMNS = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Telephone', 'Foreign Worker']
        COLUMNS_TO_DROP_FROM_X = ['Credit amount', TARGET_COLUMN_NAME]

        unnamed_col = None
        for col in df.columns:
            if col.startswith('Unnamed:'):
                unnamed_col = col
                break
        if unnamed_col:
            COLUMNS_TO_DROP_FROM_X.append(unnamed_col)
            print(f"Đã phát hiện và sẽ bỏ cột '{unnamed_col}'.")
        else:
            print("Không tìm thấy cột bắt đầu bằng 'Unnamed:'.")

        if TARGET_COLUMN_NAME not in df.columns:
            print(f"\nLỗi: Không tìm thấy cột mục tiêu '{TARGET_COLUMN_NAME}'.")
            return None

        CATEGORICAL_FEATURE_COLUMNS_USED = [col for col in CATEGORICAL_FEATURE_COLUMNS if col in df.columns]
        for col in list(CATEGORICAL_FEATURE_COLUMNS):
            if col not in CATEGORICAL_FEATURE_COLUMNS_USED:
                print(f"\nCảnh báo: Không tìm thấy cột phân loại '{col}'.")
                CATEGORICAL_FEATURE_COLUMNS.remove(col)

        cols_to_fillna = ['Saving accounts', 'Checking account']
        cols_to_fillna_existing = [col for col in cols_to_fillna if col in df.columns]
        fillna_value = 'unknown'
        for col in cols_to_fillna_existing:
            df[col] = df[col].fillna(fillna_value)

        target_le = LabelEncoder()
        df[TARGET_COLUMN_NAME] = target_le.fit_transform(df[TARGET_COLUMN_NAME])
        print(f"Đã mã hóa cột mục tiêu '{TARGET_COLUMN_NAME}'. Các lớp gốc: {target_le.classes_}, mã hóa thành: {target_le.transform(target_le.classes_)}")

        label_encoders = {}
        for col in CATEGORICAL_FEATURE_COLUMNS_USED:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        df = df.replace([np.inf, -np.inf], np.nan)

        cols_existing_to_drop = [col for col in COLUMNS_TO_DROP_FROM_X if col in df.columns]
        X = df.drop(columns=cols_existing_to_drop)
        y = df[TARGET_COLUMN_NAME]
        numeric_cols_in_X = X.select_dtypes(include=np.number).columns.tolist()
        FEATURE_COLUMNS_ORDER = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train[numeric_cols_in_X] = scaler.fit_transform(X_train[numeric_cols_in_X])
        X_test[numeric_cols_in_X] = scaler.transform(X_test[numeric_cols_in_X])

        print("Bắt đầu huấn luyện mô hình Logistic Regression...")
        model = LogisticRegression(random_state=42, solver='liblinear') # Chọn solver phù hợp
        model.fit(X_train, y_train)
        print("Huấn luyện mô hình hoàn tất.")

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        print("\n--- Đánh giá mô hình trên tập Test ---")
        print(f"Accuracy: {accuracy:.2f}")
        print("Confusion Matrix:\n{conf_matrix}")
        print("Classification Report:\n{class_report}")
        print("--------------------------------------")

        OUTPUT_DIR = 'src/LogisticRegression/saved_models'
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(OUTPUT_DIR, 'logistic_regression_model.pkl'))
        joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler_lr.pkl'))
        joblib.dump(label_encoders, os.path.join(OUTPUT_DIR, 'label_encoders_lr.pkl'))
        joblib.dump(target_le, os.path.join(OUTPUT_DIR, 'target_encoder_lr.pkl'))
        joblib.dump(FEATURE_COLUMNS_ORDER, os.path.join(OUTPUT_DIR, 'feature_cols_order_lr.pkl'))
        joblib.dump(CATEGORICAL_FEATURE_COLUMNS_USED, os.path.join(OUTPUT_DIR, 'categorical_features_used_lr.pkl'))
        joblib.dump(numeric_cols_in_X, os.path.join(OUTPUT_DIR, 'numeric_features_used_lr.pkl'))
        joblib.dump(fillna_value, os.path.join(OUTPUT_DIR, 'fillna_value_lr.pkl'))

        LOG_DIR = 'logs'
        os.makedirs(LOG_DIR, exist_ok=True)
        LOG_FILENAME = os.path.join(LOG_DIR, 'training_log_lr.txt')

        with open(LOG_FILENAME, 'w', encoding='utf-8') as f:
            f.write("--- Báo cáo Huấn luyện Mô hình Logistic Regression ---\n\n")
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
    evaluation_metrics = train_and_evaluate_logistic_regression(DATA_PATH)
    if evaluation_metrics:
        print("\n--- Các chỉ số đánh giá trả về ---")
        print(f"Accuracy: {evaluation_metrics['accuracy']:.2f}")
        print("Confusion Matrix:\n", evaluation_metrics['confusion_matrix'])
        print("Classification Report:\n", evaluation_metrics['classification_report'])
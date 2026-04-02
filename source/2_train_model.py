import pandas as pd
import numpy as np
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

print("--- BƯỚC 2: HUẤN LUYỆN MÔ HÌNH TRÊN TẬP TRAIN ---")
train_df = pd.read_csv("train_data.csv")
target = 'extreme'

# Xóa các cột gây rò rỉ dữ liệu (Nhiệt độ, Gió, Mưa hiện tại)
drop_cols = [
    target, 'rain', 'time', 'weather_code', 'weather_main', 'weather_description', 'weather_icon',
    'temperature', 'temp_min', 'temp_max', 'feels_like', 'temp_range', 
    'wind_speed', 'wind_gust', 'rainfall', 'precipitation',            
    'temp_level', 'humidity_level', 'pressure_level', 'wind_level'     
]

X_train = train_df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])
y_train = train_df[target]

# XGBoost yêu cầu nhãn (y) phải là số (0, 1, 2, 3) thay vì chữ (normal, heatwave...)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
joblib.dump(le, 'label_encoder.pkl') # Lưu bộ mã hóa để dùng lúc Test

# Điền khuyết và Cân bằng dữ liệu (SMOTE)
X_train = X_train.ffill().bfill().fillna(0)
print("Đang cân bằng dữ liệu với SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train_encoded)

# Huấn luyện mô hình
print("Đang huấn luyện Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
rf_model.fit(X_train_sm, y_train_sm)

print("Đang huấn luyện XGBoost (Công nghệ mới)...")
xgb_model = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, 
                          random_state=42, eval_metric='mlogloss', n_jobs=-1)
xgb_model.fit(X_train_sm, y_train_sm)

# Lưu mô hình
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(xgb_model, 'xgb_model.pkl')
print("=> Đã lưu mô hình: 'rf_model.pkl' và 'xgb_model.pkl'")
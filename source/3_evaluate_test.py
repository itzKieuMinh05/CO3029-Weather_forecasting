import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

print("--- BƯỚC 3: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ĐỘC LẬP ---")
test_df = pd.read_csv("test_data.csv")
target = 'extreme'

drop_cols = [
    target, 'rain', 'time', 'weather_code', 'weather_main', 'weather_description', 'weather_icon',
    'temperature', 'temp_min', 'temp_max', 'feels_like', 'temp_range', 
    'wind_speed', 'wind_gust', 'rainfall', 'precipitation',            
    'temp_level', 'humidity_level', 'pressure_level', 'wind_level'     
]

X_test = test_df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])
y_test_raw = test_df[target]
X_test = X_test.ffill().bfill().fillna(0)

# Load Model và Label Encoder
le = joblib.load('label_encoder.pkl')
rf_model = joblib.load('rf_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')

y_test = le.transform(y_test_raw)

# Hàm đánh giá
def get_metrics(model, name):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)
    acc = accuracy_score(y_test, pred)
    rec = recall_score(y_test, pred, average='macro')
    prec = precision_score(y_test, pred, average='macro', zero_division=0)
    
    classes = model.classes_
    if len(classes) == 2:
        auc = roc_auc_score(y_test, prob[:, 1])
    else:
        y_test_bin = label_binarize(y_test, classes=classes)
        auc = roc_auc_score(y_test_bin, prob, average='macro', multi_class='ovr')
    return [name, f"{auc:.4f}", f"{acc:.4f}", f"{rec:.4f}", f"{prec:.4f}"]

# Bảng so sánh
results = [
    get_metrics(rf_model, "Random Forest (Baseline)"),
    get_metrics(xgb_model, "XGBoost (Công nghệ mới)")
]

results_df = pd.DataFrame(results, columns=['Model', 'AUC (OVR)', 'Accuracy', 'Recall', 'Precision'])
print("\n[BẢNG SO SÁNH TRÊN TẬP TEST]")
print(results_df.to_markdown(index=False))

# Vẽ Confusion Matrix cho mô hình tốt nhất (Giả sử là XGBoost)
best_model = xgb_model
y_pred = best_model.predict(X_test)
classes_names = le.inverse_transform(best_model.classes_)

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd', xticklabels=classes_names, yticklabels=classes_names)
plt.title('Ma trận nhầm lẫn - Cảnh báo Thời tiết (XGBoost)')
plt.ylabel('Thực tế')
plt.xlabel('Dự đoán')
plt.savefig('test_confusion_matrix.png')
print("\n=> Đã xuất ảnh 'test_confusion_matrix.png'!")

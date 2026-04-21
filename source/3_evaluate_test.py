import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

print("--- BƯỚC 3: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ĐỘC LẬP ---")
test_df = pd.read_csv("test_data.csv")
target = 'extreme'

drop_cols = [
    target, 'rain', 'time', 'weather_code', 'weather_main', 'weather_description', 'weather_icon',
    'temperature', 'temp_min', 'temp_max', 'feels_like', 'temp_range',
    'wind_speed', 'wind_gust', 'rainfall', 'precipitation',
    'temp_level', 'humidity_level', 'pressure_level', 'wind_level',
    'temp_lag_1', 'humidity_lag_1', 'pressure_lag_1'
]

X_test = test_df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])
X_test = X_test.ffill().bfill().fillna(0)

y_test_extreme_raw = test_df[target]
y_test_rain = test_df['rain']

# Load model và label encoder
le = joblib.load('label_encoder.pkl')
rf_model = joblib.load('rf_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')
rf_rain = joblib.load('rf_rain_model.pkl')
xgb_rain = joblib.load('xgb_rain_model.pkl')

y_test_extreme = le.transform(y_test_extreme_raw)


def get_metrics(model, X_eval, y_true, name, is_multiclass=False):
    pred = model.predict(X_eval)
    prob = model.predict_proba(X_eval)

    acc = accuracy_score(y_true, pred)
    rec = recall_score(y_true, pred, average='macro')
    prec = precision_score(y_true, pred, average='macro', zero_division=0)

    if prob.ndim == 2 and prob.shape[1] == 2:
        auc = roc_auc_score(y_true, prob[:, 1])
    else:
        auc = roc_auc_score(y_true, prob, average='macro', multi_class='ovr')

    return [name, f"{auc:.4f}", f"{acc:.4f}", f"{rec:.4f}", f"{prec:.4f}"]


# Đánh giá extreme
results_extreme = [
    get_metrics(rf_model, X_test, y_test_extreme, "Random Forest - Extreme", is_multiclass=True),
    get_metrics(xgb_model, X_test, y_test_extreme, "XGBoost - Extreme", is_multiclass=True),
]

results_extreme_df = pd.DataFrame(results_extreme, columns=['Model', 'AUC (OVR)', 'Accuracy', 'Recall', 'Precision'])
print("\n[BẢNG SO SÁNH EXTREME TRÊN TẬP TEST]")
print(results_extreme_df.to_markdown(index=False))


# Đánh giá rain
results_rain = [
    get_metrics(rf_rain, X_test, y_test_rain, "Random Forest - Rain"),
    get_metrics(xgb_rain, X_test, y_test_rain, "XGBoost - Rain"),
]

results_rain_df = pd.DataFrame(results_rain, columns=['Model', 'AUC', 'Accuracy', 'Recall', 'Precision'])
print("\n[BẢNG SO SÁNH RAIN TRÊN TẬP TEST]")
print(results_rain_df.to_markdown(index=False))


# Confusion matrix cho extreme
best_extreme_model = xgb_model
y_pred_extreme = best_extreme_model.predict(X_test)
extreme_class_names = le.inverse_transform(best_extreme_model.classes_)

plt.figure(figsize=(8, 6))
cm_extreme = confusion_matrix(y_test_extreme, y_pred_extreme, labels=best_extreme_model.classes_)
sns.heatmap(cm_extreme, annot=True, fmt='d', cmap='OrRd', xticklabels=extreme_class_names, yticklabels=extreme_class_names)
plt.title('Ma trận nhầm lẫn - Cảnh báo Thời tiết (XGBoost - Extreme)')
plt.ylabel('Thực tế')
plt.xlabel('Dự đoán')
plt.tight_layout()
plt.savefig('test_confusion_matrix_extreme.png')
plt.close()


# Confusion matrix cho rain
best_rain_model = xgb_rain
y_pred_rain = best_rain_model.predict(X_test)
rain_class_names = ['No Rain', 'Rain']

plt.figure(figsize=(8, 6))
cm_rain = confusion_matrix(y_test_rain, y_pred_rain, labels=best_rain_model.classes_)
sns.heatmap(cm_rain, annot=True, fmt='d', cmap='Blues', xticklabels=rain_class_names, yticklabels=rain_class_names)
plt.title('Ma trận nhầm lẫn - Dự đoán mưa (XGBoost - Rain)')
plt.ylabel('Thực tế')
plt.xlabel('Dự đoán')
plt.tight_layout()
plt.savefig('test_confusion_matrix_rain.png')
plt.close()

print("\n=> Đã xuất ảnh 'test_confusion_matrix_extreme.png' và 'test_confusion_matrix_rain.png'!")

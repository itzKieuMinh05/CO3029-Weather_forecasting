# 🌦️ WeatherVN Dashboard — Hướng dẫn cài đặt

## Cấu trúc thư mục

```
weather_dashboard/
├── app.py                  ← File chính (dashboard)
├── requirements.txt        ← Thư viện cần cài
├── rf_model.pkl            ← ✅ Cần đặt vào đây
├── xgb_model.pkl           ← ✅ Cần đặt vào đây
├── label_encoder.pkl       ← ✅ Cần đặt vào đây
├── test_data.csv           ← ✅ Cần đặt vào đây (hoặc weather_huyen.csv)
└── weather_clustered.csv   ← (Tùy chọn) kết quả clustering
```

## Cài đặt

```bash
# 1. Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 2. Cài thư viện
pip install -r requirements.txt

# 3. Đặt các file pkl và csv vào thư mục weather_dashboard/

# 4. Chạy dashboard
streamlit run app.py
```

## Các tab trong Dashboard

| Tab | Nội dung |
|-----|----------|
| 📊 Tổng quan | KPI cards, phân bố extreme/rain theo region & tháng |
| 🌧️ Dự báo | Nhập tay thông số → RF + XGBoost dự đoán + xác suất |
| 🔍 Đánh giá | AUC, Accuracy, F1, Confusion Matrix, Feature Importance |
| 🗺️ Clustering | PCA scatter, Elbow chart, thống kê cluster |

## Lưu ý

- Sidebar cho phép đổi tên file CSV linh hoạt (test_data.csv, weather_huyen.csv, ...)
- Tab Dự báo tự động align features với model (không bị lỗi thiếu cột)
- Tab Clustering tự chạy lại KMeans khi thay đổi k

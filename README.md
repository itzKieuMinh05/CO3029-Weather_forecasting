# 🌦️ WeatherVN Dashboard — CO3029 Data Mining Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-EC6C2A?logo=xgboost&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

**Dự án môn học CO3029 — Data Mining | Nhóm 03 — Lớp L03**

Hệ thống phân tích và dự báo thời tiết Việt Nam sử dụng Random Forest, XGBoost, KMeans Clustering và Apriori Association Rules, được trình bày qua dashboard Streamlit tương tác.

</div>

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Tính năng](#-tính-năng)
- [Kiến trúc dự án](#-kiến-trúc-dự-án)
- [Cài đặt](#-cài-đặt)
- [Chạy ứng dụng](#-chạy-ứng-dụng)
- [Pipeline dữ liệu & huấn luyện](#-pipeline-dữ-liệu--huấn-luyện)
- [Mô tả các Tab Dashboard](#-mô-tả-các-tab-dashboard)
- [Models & Thuật toán](#-models--thuật-toán)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [File cần thiết](#-file-cần-thiết)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)

---

## 🌟 Giới thiệu

**WeatherVN Dashboard** là một ứng dụng web tương tác phân tích dữ liệu thời tiết tại các tỉnh/thành phố trên toàn quốc Việt Nam. Dự án áp dụng các kỹ thuật khai phá dữ liệu để:

- **Phân loại thời tiết cực đoan** (nắng nóng, bão, mưa lớn, bình thường) bằng Random Forest và XGBoost.
- **Dự báo khả năng mưa** theo thời gian thực từ các thông số khí tượng đầu vào.
- **Phân cụm tỉnh/thành** theo đặc điểm khí hậu bằng KMeans + PCA.
- **Khám phá luật kết hợp** (Apriori) giữa các điều kiện thời tiết và sự kiện cực đoan.

Dữ liệu bao gồm các thành phố thuộc **3 vùng miền**: Bắc, Trung, Nam — với các đặc trưng như nhiệt độ, độ ẩm, áp suất khí quyển, tốc độ gió, lượng mây và tầm nhìn.

---

## ✨ Tính năng

| Tính năng | Mô tả |
|---|---|
| 📊 **Tổng quan dữ liệu** | KPI cards, phân bố mưa theo tháng, heatmap nhiệt độ, correlation matrix |
| 🌧️ **Dự báo thời tiết** | Nhập thủ công các thông số → Dự báo cực đoan + xác suất mưa (RF + XGB ensemble) |
| 🔍 **Đánh giá mô hình** | So sánh AUC, Accuracy, F1, Confusion Matrix, Feature Importance |
| 🗺️ **Clustering** | PCA scatter, Elbow + Silhouette chart, Radar profile từng cụm |
| 🧩 **Apriori Rules** | Dashboard luật kết hợp, lọc theo region/target/lift/confidence |
| ⚡ **Preset scenarios** | Mẫu thời tiết có sẵn: Ngày mưa, Nắng nóng, Bão, Nắng đẹp |

---

## 🏗️ Kiến trúc dự án

```
┌─────────────────────────────────────────────────────────┐
│                    WeatherVN Dashboard                  │
│                      (Streamlit)                        │
├──────────┬──────────┬─────────┬────────────┬────────────┤
│ Overview │Prediction│Evaluate │ Clustering │  Apriori   │
│ (📊)     │  (🌧️)   │  (🔍)  │   (🗺️)   │   (🧩)    │
└──────────┴──────────┴─────────┴────────────┴────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│                      utils.py                           │
│   CSS · Model Loader · CSV Loader · HTML Helpers        │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────┐   ┌────────────────┐   ┌────────────┐
│ source/          │   │ notebook/      │   │  Models    │
│ preprocessing.py │   │ apriori.ipynb  │   │  (*.pkl)   │
│ 1_split_data.py  │   │ clustering.py  │   │            │
│ 2_train_model.py │   │ classific..py  │   │            │
│ 3_evaluate_..py  │   │ visualization.py│  │            │
└──────────────────┘   └────────────────┘   └────────────┘
```

---

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/itzKieuMinh05/CO3029-Weather_forecasting.git
cd CO3029-Weather_forecasting
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Đặt file dữ liệu & model

Xem chi tiết tại phần **[File cần thiết](#-file-cần-thiết)** bên dưới.

---

## ▶️ Chạy ứng dụng

```bash
streamlit run app.py
```

Mở trình duyệt tại: **http://localhost:8501**

---

## 🔄 Pipeline dữ liệu & huấn luyện

Nếu muốn tự huấn luyện lại mô hình từ dữ liệu thô, thực hiện theo thứ tự:

### Bước 0 — Tiền xử lý dữ liệu

```bash
python source/preprocessing.py
```

**Chức năng:**
- Load toàn bộ file CSV từ `../data/weather-vn-*.csv`
- Trích xuất các đặc trưng thời gian: `hour`, `day`, `month`, `weekday`
- Gán vùng miền (`north` / `central` / `south`) theo tên tỉnh/thành
- Xử lý missing values (median cho số, mode cho chuỗi)
- Tính `temp_range` theo ngày và tỉnh
- Chuyển `wind_direction` → `wind_dir_sin`, `wind_dir_cos`
- Tạo nhãn `rain` (0/1) từ cột `rainfall` hoặc `precipitation`
- Tạo nhãn `extreme`: `heatwave` / `heavy_rain` / `storm` / `normal`
- Xuất: `weather_vn_cleaned.csv`

### Bước 1 — Chia tập Train/Test

```bash
python source/1_split_data.py
```

Chia 80/20 với `stratify` theo cả `extreme` và `rain`.
- Xuất: `train_data.csv`, `test_data.csv`

### Bước 2 — Huấn luyện mô hình

```bash
python source/2_train_model.py
```

**Mô hình được huấn luyện:**
- `RandomForestClassifier` (100 cây, max_depth=12, class_weight='balanced')
- `XGBClassifier` (150 cây, max_depth=6, learning_rate=0.05)
- Cả hai đều dùng **SMOTE** để xử lý imbalanced classes
- Thêm model riêng cho bài toán phân loại `rain` (binary)
- Xuất: `rf_model.pkl`, `xgb_model.pkl`, `rf_rain_model.pkl`, `xgb_rain_model.pkl`, `label_encoder.pkl`

### Bước 3 — Đánh giá trên tập Test

```bash
python source/3_evaluate_test.py
```

In bảng so sánh AUC, Accuracy, Recall, Precision và vẽ confusion matrix cho cả hai bài toán (extreme & rain).

### Phân tích Apriori (Notebook)

```bash
jupyter notebook notebook/apriori.ipynb
```

Xuất file `rules_output/rules_{region}.csv` (north / central / south) để dashboard Apriori có thể đọc.

---

## 📱 Mô tả các Tab Dashboard

### 📊 Tổng quan (`pages/overview.py`)

- **6 KPI cards**: Tổng bản ghi, tỷ lệ mưa, % cực đoan, số thành phố, nhiệt độ TB, gió TB
- **Biểu đồ Stacked Bar**: Phân bổ loại thời tiết (extreme) theo vùng miền
- **Line + Bar chart**: Xác suất mưa theo từng tháng
- **Heatmap**: Nhiệt độ trung bình theo Region × Tháng
- **Histogram**: Phân bố tốc độ gió theo vùng miền
- **Bar + Line chart**: Pattern mưa theo giờ trong ngày
- **Correlation matrix**: Tương quan giữa các biến số
- **Data table**: Xem và lọc 200 bản ghi mẫu theo thành phố
- **Sidebar**: Lọc theo region (Bắc/Trung/Nam), chọn file CSV

### 🌧️ Dự báo (`pages/prediction.py`)

- **Nhập thông số**: Nhiệt độ, Độ ẩm, Áp suất, Mây che phủ (4 thông số chính)
- **Tinh chỉnh nâng cao**: Tầm nhìn, hướng gió, giờ/ngày/tháng, ban ngày/đêm
- **Preset scenarios**: Ngày mưa / Nắng nóng / Nắng đẹp / Bão / Tùy chỉnh
- **Kết quả dự báo**:
  - Phân loại thời tiết cực đoan (ensemble RF + XGB)
  - Nguy cơ nắng nóng (%)
  - Xác suất mưa (%) với so sánh RF vs XGB
  - Progress bars và xác suất từng lớp

### 🔍 Đánh giá (`pages/evaluate.py`)

- **6 KPI cards**: AUC, Accuracy, F1 của cả hai mô hình
- **Grouped Bar chart**: So sánh AUC, Accuracy, Recall, Precision, F1
- **Winner card**: Highlight mô hình tốt hơn kèm chênh lệch AUC
- **Confusion Matrix**: RF (Blues) vs XGBoost (Oranges)
- **Feature Importance**: Top 15 đặc trưng quan trọng nhất của từng mô hình
- **Sidebar**: Chọn file test

### 🗺️ Clustering (`pages/clustering.py`)

- **KMeans + Auto-k**: Tự động tìm k tối ưu theo Silhouette score
- **PCA Scatter plot**: Visualize các cụm với label tên tỉnh/thành
- **Cluster centroids**: Bảng giá trị trung bình từng cụm
- **Region breakdown**: Phân bổ vùng miền trong từng cụm (stacked bar)
- **Radar chart**: Profile đặc trưng khí hậu từng cụm
- **Extreme heatmap**: Tỷ lệ thời tiết cực đoan per cluster
- **Elbow + Silhouette charts**: Hỗ trợ chọn k
- **Sidebar**: Tự/thủ công chọn k, lọc region, % mẫu hiển thị

### 🧩 Apriori (`pages/apriori.py`)

- Load kết quả từ `rules_output/rules_{region}.csv`
- **6 KPI cards**: Số rules, Confidence TB, Lift TB, Lift cao nhất, Tiền đề unique, % rules mưa
- **Bar charts**: Phân bố rules theo region và theo loại consequent
- **Scatter plot**: Support vs Confidence (màu = Lift)
- **Horizontal bar**: Top N rules theo Lift
- **Bảng chi tiết**: Toàn bộ rules với Antecedent, Consequent, Support, Confidence, Lift
- **Sidebar**: Lọc region, target, lift/confidence tối thiểu, top N

---

## 🤖 Models & Thuật toán

### Bài toán phân loại thời tiết cực đoan (Multiclass)

| Model | Hyperparameters | Xử lý imbalanced |
|---|---|---|
| Random Forest | n_estimators=100, max_depth=12, class_weight='balanced' | SMOTE |
| XGBoost | n_estimators=150, max_depth=6, learning_rate=0.05, eval_metric='mlogloss' | SMOTE |

**Nhãn**: `normal` · `heatwave` · `storm` · `heavy_rain`

### Bài toán phân loại mưa (Binary)

| Model | Hyperparameters |
|---|---|
| Random Forest | n_estimators=100, max_depth=12, class_weight='balanced' |
| XGBoost | n_estimators=150, max_depth=6, learning_rate=0.05, eval_metric='logloss' |

**Nhãn**: `0` (không mưa) · `1` (có mưa)

### Clustering

- **Thuật toán**: KMeans (n_init=10, random_state=42)
- **Chuẩn hóa**: StandardScaler
- **Tìm k**: Silhouette Score (tự động) hoặc Elbow method (trực quan)
- **Visualize**: PCA 2 chiều + Radar chart

### Apriori Association Rules

- Chạy theo từng region (north / central / south)
- Lọc luật theo `support`, `confidence`, `lift`
- Consequent target: `rain_yes`, `heatwave`, `storm`, `heavy_rain`

---

## 📁 Cấu trúc thư mục

```
CO3029-Weather_forecasting/
│
├── app.py                          ← Trang chủ Streamlit (hero + navigation)
├── utils.py                        ← Helpers: CSS, model loader, HTML components
├── requirements.txt                ← Danh sách thư viện Python
│
├── pages/
│   ├── overview.py                 ← Tab Tổng quan
│   ├── prediction.py               ← Tab Dự báo
│   ├── evaluate.py                 ← Tab Đánh giá mô hình
│   ├── clustering.py               ← Tab Phân tích cụm
│   └── apriori.py                  ← Tab Apriori Rules
│
├── source/
│   ├── preprocessing.py            ← Tiền xử lý & tạo nhãn
│   ├── 1_split_data.py             ← Chia train/test
│   ├── 2_train_model.py            ← Huấn luyện mô hình
│   └── 3_evaluate_test.py          ← Đánh giá trên tập test
│
├── notebook/
│   ├── apriori.ipynb               ← Notebook Apriori (sinh rules_output/)
│   ├── data_exploration.ipynb      ← EDA notebook
│   ├── classification.py           ← Script phân loại
│   ├── clustering.py               ← Script phân cụm
│   └── visualization.py            ← Script visualize
│
├── .streamlit/
│   └── config.toml                 ← Cấu hình giao diện Streamlit
│
│   ── [Cần thêm thủ công] ──
├── rf_model.pkl                    ← Model Random Forest (extreme)
├── xgb_model.pkl                   ← Model XGBoost (extreme)
├── rf_rain_model.pkl               ← Model Random Forest (rain)
├── xgb_rain_model.pkl              ← Model XGBoost (rain)
├── label_encoder.pkl               ← LabelEncoder cho nhãn extreme
├── kmeans_model.pkl                ← Model KMeans (tùy chọn)
├── scaler.pkl                      ← StandardScaler (tùy chọn)
├── weather_vn_cleaned.csv          ← Dữ liệu đã xử lý
├── train_data.csv                  ← Tập huấn luyện (80%)
├── test_data.csv                   ← Tập kiểm tra (20%)
└── rules_output/
    ├── rules_north.csv             ← Luật Apriori vùng Bắc
    ├── rules_central.csv           ← Luật Apriori vùng Trung
    └── rules_south.csv             ← Luật Apriori vùng Nam
```

---

## 📦 File cần thiết

App có thể khởi động mà không cần đủ file, tuy nhiên một số tab sẽ báo lỗi nếu thiếu. Dưới đây là danh sách đầy đủ:

| File | Bắt buộc | Sinh từ | Mô tả |
|---|---|---|---|
| `weather_vn_cleaned.csv` | ✅ | `preprocessing.py` | Dữ liệu đã xử lý |
| `train_data.csv` | ✅ | `1_split_data.py` | Tập huấn luyện |
| `test_data.csv` | ✅ | `1_split_data.py` | Tập kiểm tra |
| `rf_model.pkl` | ✅ | `2_train_model.py` | Model RF (extreme) |
| `xgb_model.pkl` | ✅ | `2_train_model.py` | Model XGB (extreme) |
| `rf_rain_model.pkl` | ✅ | `2_train_model.py` | Model RF (rain) |
| `xgb_rain_model.pkl` | ✅ | `2_train_model.py` | Model XGB (rain) |
| `label_encoder.pkl` | ✅ | `2_train_model.py` | LabelEncoder |
| `kmeans_model.pkl` | ⬜ | Tùy chọn | KMeans (nếu có) |
| `scaler.pkl` | ⬜ | Tùy chọn | StandardScaler |
| `rules_output/rules_*.csv` | ⬜ | `apriori.ipynb` | Tab Apriori |

> **Lưu ý:** App tự động tìm file trong thư mục gốc và thư mục `data/`. Sidebar sẽ hiển thị trạng thái từng model (● xanh = OK, ○ đỏ = thiếu).

---

## 🖥️ Yêu cầu hệ thống

- **Python**: 3.10 trở lên
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB)
- **Hệ điều hành**: Windows / Linux / macOS

### Thư viện chính

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
joblib
```


## 📄 Tài liệu tham khảo

- [Báo cáo nhóm (PDF)](./Report_BTL_Data_Mining_L03_Group03%20(2).pdf)
- [Slide thuyết trình (PDF)](./Slide_BTL_Data_Mining_L03_Group03.pdf)
- [Streamlit Documentation](https://docs.streamlit.io)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io)

---

<div align="center">
<sub>Made with ❤️ for CO3029 Data Mining · HCMUT · 2024–2025</sub>
</div>

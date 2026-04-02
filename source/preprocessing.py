import pandas as pd
import numpy as np
import glob

files = glob.glob("../data/weather-vn-*.csv")
dfs = []

for f in files:
    print("Loading:", f)
    df = pd.read_csv(f)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
print("Total rows:", len(df))

# 1. Drop duplicates
df = df.drop_duplicates()

# 2. Parse time
df["time"] = pd.to_datetime(df["time"])
df["hour"] = df["time"].dt.hour
df["day"] = df["time"].dt.day
df["month"] = df["time"].dt.month
df["weekday"] = df["time"].dt.weekday

# 3. Missing values
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=["object", "string"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 4. Feature Engineering: Range nhiệt độ & Hướng gió
if "temp_max" in df.columns and "temp_min" in df.columns:
    df["temp_range"] = df["temp_max"] - df["temp_min"]

if "wind_direction" in df.columns:
    df["wind_dir_sin"] = np.sin(np.deg2rad(df["wind_direction"])).round(4)
    df["wind_dir_cos"] = np.cos(np.deg2rad(df["wind_direction"])).round(4)

df = df.drop(columns=["weather_desc"], errors="ignore")

if "rainfall" in df.columns:
    df["rain"] = df["rainfall"].apply(lambda x: 1 if x > 0 else 0)
elif "precipitation" in df.columns:
    df["rain"] = df["precipitation"].apply(lambda x: 1 if x > 0 else 0)

# 5. Create EXTREME WEATHER column (Hạ ngưỡng để có đủ 4 class)
def extreme_weather(row):
    if row.get("temp_max", 0) > 35:
        return "heatwave"
    elif row.get("rainfall", 0) > 15: # Hạ ngưỡng lượng mưa (ví dụ >15mm)
        return "heavy_rain"
    elif row.get("wind_speed", 0) > 15: # Hạ ngưỡng gió (ví dụ >15m/s)
        return "storm"
    else:
        return "normal"

df["extreme"] = df.apply(extreme_weather, axis=1)

# 6. Discretization (for Apriori)
if "temp_max" in df.columns:
    df["temp_level"] = pd.cut(
        df["temp_max"],
        bins=[-100, 20, 30, 100],
        labels=["temp_low", "temp_medium", "temp_high"]
    )

if "humidity" in df.columns:
    df["humidity_level"] = pd.cut(
        df["humidity"],
        bins=[0, 60, 80, 100],
        labels=["humidity_low", "humidity_medium", "humidity_high"]
    )

if "pressure" in df.columns:
    df["pressure_level"] = pd.cut(
        df["pressure"],
        bins=[0, 1000, 1015, 2000],
        labels=["pressure_low", "pressure_normal", "pressure_high"]
    )

if "wind_speed" in df.columns:
    df["wind_level"] = pd.cut(
        df["wind_speed"],
        bins=[0, 10, 25, 100],
        labels=["wind_low", "wind_medium", "wind_high"]
    )

# 7. Lag features (previous hour) - SỬA LỖI LOGIC TẠI ĐÂY
# Phải sort theo thành phố rồi mới sort theo thời gian
df = df.sort_values(by=["city", "time"])

if "temp_max" in df.columns:
    # Group by city rồi mới shift để không bị lẫn lộn dữ liệu giữa các tỉnh
    df["temp_lag_1"] = df.groupby("city")["temp_max"].shift(1)

if "humidity" in df.columns:
    df["humidity_lag_1"] = df.groupby("city")["humidity"].shift(1)

if "pressure" in df.columns:
    df["pressure_lag_1"] = df.groupby("city")["pressure"].shift(1)

# 8. Drop location columns (Chỉ xóa SAU KHI đã tạo xong Lag features)
df = df.drop(columns=["province", "city"], errors="ignore")

# 9. Save
df.to_csv("weather_vn_cleaned.csv", index=False)

print("Saved: weather_vn_cleaned.csv")
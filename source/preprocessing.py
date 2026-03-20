import pandas as pd
import numpy as np
import glob

files = glob.glob("data/weather-vn-*.csv")
dfs = []

for f in files:
    print("Loading:", f)
    df = pd.read_csv(f)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
print("Total rows:", len(df))

# drop duplicates
df = df.drop_duplicates()


# parse time
df["time"] = pd.to_datetime(df["time"])

df["hour"] = df["time"].dt.hour
df["day"] = df["time"].dt.day
df["month"] = df["time"].dt.month
df["weekday"] = df["time"].dt.weekday


# drop location columns

df = df.drop(columns=["province", "city"], errors="ignore")
# Missing values

num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=["object","string"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# range nhiệt độ
if "temp_max" in df.columns and "temp_min" in df.columns:
    df["temp_range"] = df["temp_max"] - df["temp_min"]

# wind direction -> vector
if "wind_direction" in df.columns:
    df["wind_dir_sin"] = np.sin(np.deg2rad(df["wind_direction"])).round(4)
    df["wind_dir_cos"] = np.cos(np.deg2rad(df["wind_direction"])).round(4)


# drop weather_desc
df = df.drop(columns=["weather_desc"], errors="ignore")
# save
df.to_csv("weather_vn_cleaned.csv", index=False)

print("Saved: weather_vn_cleaned.csv")
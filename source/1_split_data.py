import pandas as pd
from sklearn.model_selection import train_test_split

print("--- BƯỚC 1: ĐỌC VÀ CHIA TẬP DỮ LIỆU ---")
df = pd.read_csv("weather_vn_cleaned.csv")

# Chia 80% Train - 20% Test (stratify giúp giữ nguyên tỷ lệ các loại thời tiết cực đoan)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['extreme'])

# Lưu ra 2 file vật lý
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print(f"Đã lưu thành công:")
print(f"- train_data.csv: {len(train_df)} dòng")
print(f"- test_data.csv: {len(test_df)} dòng")
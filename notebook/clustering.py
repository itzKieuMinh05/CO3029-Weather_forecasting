import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ======================
# 1. LOAD DATA
# ======================


def load_data(path="weather_huyen.csv"):
    print("Loading data...")
    df = pd.read_csv(path)
    df = ensure_region_column(df)
    print("Shape:", df.shape)
    return df


def ensure_region_column(df):
    # Uu tien dung cot da duoc tao tu preprocessing.
    if "region" in df.columns:
        return df

    for col in REGION_SOURCE_COLUMNS[1:]:
        if col in df.columns:
            df["region"] = df[col]
            return df

    # Fallback: giu schema nhat quan de cac buoc sau khong bi loi cot.
    df["region"] = "unknown"
    return df


# ======================
# 2. SELECT FEATURES
# ======================
def select_features(df):
    features = [
        # "temp_max",
        # "temp_min",
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "temp_range"
    ]

    # giữ những cột tồn tại
    features = [f for f in features if f in df.columns]

    if not features:
        raise ValueError("Khong tim thay feature so hop le de clustering.")

    print("Using features:", features)

    X = df[features].dropna()
    return X, features


# ======================
# 3. SCALING
# ======================
def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# ======================
# 4. FIND BEST K (ELBOW)
# ======================
def find_optimal_k(X_scaled):
    inertia = []
    k_values = range(1, 10)

    batch_size = min(2048, max(256, len(X_scaled) // 20))

    print(f"Running Elbow Method with MiniBatchKMeans (batch_size={batch_size})...")

    for k in k_values:
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
            batch_size=batch_size,
            max_iter=200,
            reassignment_ratio=0.01,
        )
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure()
    plt.plot(k_values, inertia, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method (MiniBatchKMeans)")
    plt.show()


# ======================
# 5. TRAIN KMEANS
# ======================
def train_kmeans(X_scaled, k=3):
    batch_size = min(2048, max(256, len(X_scaled) // 20))
    print(f"Training MiniBatchKMeans with k={k}, batch_size={batch_size}...")
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
        batch_size=batch_size,
        max_iter=300,
        reassignment_ratio=0.01,
    )
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels


# ======================
# 6. VISUALIZATION (PCA)
# ======================
def visualize_clusters(X_scaled, labels, regions):
    import numpy as np

    n = len(X_scaled)

    # ======================
    # SAMPLE 25%
    # ======================
    sample_size = int(n * 0.25)
    idx = np.random.permutation(n)[:sample_size]

    X_sample = X_scaled[idx]
    labels_sample = labels[idx]
    regions_sample = pd.Series(regions).iloc[idx].reset_index(drop=True)

    print(f"Visualizing on {sample_size:,} samples (25%)")

    # ======================
    # PCA FIT TRÊN FULL DATA (chuẩn hơn)
    # ======================
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_sample)

    var = pca.explained_variance_ratio_
    print(f"PCA variance: {var[0]:.1%}, {var[1]:.1%} (total={var.sum():.1%})")

    # ======================
    # TÁCH THEO REGION → 3 BIỂU ĐỒ
    # ======================
    region_series = regions_sample.fillna("khac").astype(str)
    unique_regions = sorted(region_series.unique())

    for region in unique_regions:
        mask = region_series == region

        X_r = X_pca[mask]
        labels_r = labels_sample[mask]

        plt.figure(figsize=(6, 5))

        sc = plt.scatter(
            X_r[:, 0],
            X_r[:, 1],
            c=labels_r,
            cmap="tab10",
            s=10,
            alpha=0.6,
        )

        plt.xlabel(f"PC1 ({var[0]:.1%})")
        plt.ylabel(f"PC2 ({var[1]:.1%})")
        plt.title(f"PCA - Region: {region}")

        plt.legend(*sc.legend_elements(), title="Cluster", loc="upper right")

        plt.tight_layout()
        plt.show()
    
    # ======================
    # PLOT 2: THEO REGION
    # ======================
    plt.figure(figsize=(7, 6))

    region_series = regions_sample.fillna("khac").astype(str)
    unique_regions = sorted(region_series.unique())

    palette = sns.color_palette("Set2", n_colors=len(unique_regions))
    color_map = {r: palette[i] for i, r in enumerate(unique_regions)}


    for region in unique_regions:
        mask = region_series == region

        plt.figure(figsize=(7, 6))

        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            color=color_map[region],
            s=10,
            alpha=0.6,
        )

        plt.xlabel(f"PC1 ({var[0]:.1%})")
        plt.ylabel(f"PC2 ({var[1]:.1%})")
        plt.title(f"PCA theo Region: {region}")

        # legend đơn giản (1 màu thôi)
        handle = plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            color=color_map[region],
            label=region
        )
        plt.legend(handles=[handle], title="Region", loc="upper right")

        plt.tight_layout()
        plt.show()
# ======================
# 7. ANALYZE CLUSTERS
# ======================
def analyze_clusters(df, X, labels, features):
    df_clustered = df.loc[X.index].copy()
    df_clustered["cluster"] = labels

    print("\nCluster Summary:")
    summary = df_clustered.groupby("cluster")[features].mean()
    print(summary)

    if "region" in df_clustered.columns:
        print("\nCluster by region (count):")
        print(df_clustered.groupby(["cluster", "region"]).size().unstack(fill_value=0))
    
    if "rain" in df_clustered.columns:
        print("\nRain rate per cluster:")
        print(df_clustered.groupby("cluster")["rain"].mean())
    
    if "extreme" in df_clustered.columns:
        print("\nExtreme weather distribution:")
        print(df_clustered.groupby("cluster")["extreme"].value_counts(normalize=True))

    return df_clustered


# ======================
# 8. SAVE RESULT
# ======================
def save_result(df_clustered, path="weather_clustered.csv"):
    df_clustered.to_csv(path, index=False)
    print(f"Saved clustered data to {path}")


# ======================
# 9. MAIN
# ======================
def main():
    # load
    df = load_data()

    # feature selection
    X, features = select_features(df)

    # scale
    X_scaled = scale_data(X)

    # tìm k
    find_optimal_k(X_scaled)

    k = 3

    # train
    _, labels = train_kmeans(X_scaled, k)

    # visualize
    visualize_clusters(X_scaled, labels, df.loc[X.index, "region"])

    # analyze
    df_clustered = analyze_clusters(df, X, labels, features)

    # save
    save_result(df_clustered)


if __name__ == "__main__":
    main()
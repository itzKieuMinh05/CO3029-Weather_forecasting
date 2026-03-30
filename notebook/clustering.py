import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ======================
# 1. LOAD DATA
# ======================
REGION_SOURCE_COLUMNS = ["region", "mien", "miền"]


def load_data(path="../source/weather_vn_cleaned.csv"):
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
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    scatter_cluster = axes[0].scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap="tab10",
        s=12,
        alpha=0.7,
    )
    axes[0].set_xlabel("PCA 1")
    axes[0].set_ylabel("PCA 2")
    axes[0].set_title("PCA to mau theo cluster")
    legend_cluster = axes[0].legend(
        *scatter_cluster.legend_elements(),
        title="Cluster",
        loc="upper right",
    )
    axes[0].add_artist(legend_cluster)

    region_series = pd.Series(regions).fillna("khac").astype(str)
    unique_regions = sorted(region_series.unique())
    palette = sns.color_palette("Set2", n_colors=max(len(unique_regions), 3))
    color_map = {region: palette[i] for i, region in enumerate(unique_regions)}
    region_colors = region_series.map(color_map)

    axes[1].scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=list(region_colors),
        s=12,
        alpha=0.7,
    )
    axes[1].set_xlabel("PCA 1")
    axes[1].set_ylabel("PCA 2")
    axes[1].set_title("PCA to mau theo region")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color_map[r], label=r)
        for r in unique_regions
    ]
    axes[1].legend(handles=handles, title="Region", loc="upper right")

    plt.suptitle("So sanh phan bo Cluster va Region tren khong gian PCA")
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
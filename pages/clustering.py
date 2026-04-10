import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Clustering — WeatherVN", page_icon="🗺️", layout="wide", initial_sidebar_state="expanded")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

def _sb():
    st.markdown('<div style="padding:12px 4px 8px;font-size:11px;font-weight:700;color:#374151;text-transform:uppercase;letter-spacing:.8px;font-family:\'DM Mono\',monospace;">⚙️ Cài đặt</div>', unsafe_allow_html=True)
    df_ = st.selectbox("📂 File dữ liệu", ["weather_vn_cleaned.csv","test_data.csv","train_data.csv"])
    kv  = st.slider("Số clusters (k)", 2, 8, 3)
    sp  = st.slider("% mẫu hiển thị", 5, 50, 25)
    ro  = st.selectbox("Lọc Region", ["Tất cả","north","central","south"])
    se  = st.checkbox("Hiện Elbow chart", value=True)
    st.session_state['cl_file'] = df_
    st.session_state['cl_k']    = kv
    st.session_state['cl_sp']   = sp
    st.session_state['cl_ro']   = ro
    st.session_state['cl_se']   = se

sidebar_header(_sb)
data_file  = st.session_state.get('cl_file', 'weather_vn_cleaned.csv')
k_val      = st.session_state.get('cl_k',    3)
sample_pct = st.session_state.get('cl_sp',   25)
region_opt = st.session_state.get('cl_ro',   'Tất cả')
show_elbow = st.session_state.get('cl_se',   True)


page_header("🗺️","linear-gradient(135deg,#FAF5FF,#EDE9FE)",
            "Phân tích Clustering",
            "KMeans · PCA · Phân bổ thời tiết theo vùng miền")

models  = load_models()
df_full = load_csv(data_file)
if df_full is None and data_file != "test_data.csv":
    df_full = load_csv("test_data.csv")

if df_full is None:
    st.warning("⚠️ Không tìm thấy file dữ liệu."); st.stop()

# FIX: Only keep CLUSTER_FEATURES that actually exist in the dataframe
feats = [f for f in CLUSTER_FEATURES if f in df_full.columns]
if not feats:
    st.error(f"Không tìm thấy bất kỳ cột nào trong: {CLUSTER_FEATURES}"); st.stop()

X_raw = df_full[feats].dropna()

if len(X_raw) == 0:
    st.error("Không có dữ liệu sau khi loại bỏ NaN."); st.stop()

# Scale
scaler = models.get('scaler')
if scaler is not None:
    try:
        # FIX: scaler may have been fitted on more/fewer features — use safe transform
        if hasattr(scaler, 'feature_names_in_'):
            common = [f for f in feats if f in scaler.feature_names_in_]
            if len(common) == len(list(scaler.feature_names_in_)):
                X_sc = scaler.transform(X_raw[common])
            else:
                X_sc = StandardScaler().fit_transform(X_raw)
        else:
            X_sc = scaler.transform(X_raw)
    except Exception:
        X_sc = StandardScaler().fit_transform(X_raw)
else:
    X_sc = StandardScaler().fit_transform(X_raw)

# KMeans
with st.spinner("⏳ Đang phân cụm..."):
    labels = None
    km_saved = models.get('kmeans')
    if km_saved is not None and hasattr(km_saved, 'n_clusters') and km_saved.n_clusters == k_val:
        try:
            labels = km_saved.predict(X_sc)
        except Exception:
            labels = None
    if labels is None:
        bs = min(2048, max(256, len(X_sc) // 20))
        km = MiniBatchKMeans(n_clusters=k_val, random_state=42, n_init=10,
                             batch_size=bs, max_iter=300)
        labels = km.fit_predict(X_sc)
        km_saved = km

    # FIX: PCA n_components must be <= n_features
    n_components = min(2, X_sc.shape[1])
    pca   = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_sc)
    var   = pca.explained_variance_ratio_

    # Pad to 2D if only 1 component
    if X_pca.shape[1] == 1:
        X_pca = np.hstack([X_pca, np.zeros((len(X_pca), 1))])
        var   = np.append(var, 0.0)

df_cl = df_full.loc[X_raw.index].copy()
df_cl['cluster'] = labels

# Region filter
if region_opt != "Tất cả" and 'region' in df_cl.columns:
    df_view = df_cl[df_cl['region'] == region_opt]
else:
    df_view = df_cl

n = len(df_view)
if n == 0:
    st.warning(f"Không có dữ liệu cho region: {region_opt}"); st.stop()

idx_s = np.random.RandomState(42).permutation(n)[:max(1, int(n * sample_pct / 100))]

# FIX: Correctly compute mask for PCA view
view_indices = df_view.index
mask = df_cl.index.isin(view_indices)

X_pca_v  = X_pca[mask]
labels_v  = labels[mask]
X_pca_s  = X_pca_v[idx_s]
labels_s  = labels_v[idx_s]

PAL     = plt.cm.tab10(np.linspace(0, 0.9, k_val))
PAL_HEX = [f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}' for c in PAL]

# ── PCA scatter ───────────────────────────────────────
section(f"PCA Scatter — {len(X_pca_s):,} điểm ({sample_pct}% mẫu)")
pca_c, stat_c = st.columns([3, 2])

with pca_c:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig, ax = fig_ax(7, 5.5)
    unique_clusters = sorted(np.unique(labels_s).tolist())
    for ci in unique_clusters:
        m = labels_s == ci
        if m.sum() > 0:
            ax.scatter(X_pca_s[m, 0], X_pca_s[m, 1],
                       color=PAL[ci], s=10, alpha=0.55, label=f'Cluster {ci}', zorder=3)
    # Centroids
    centroid_clusters = sorted(np.unique(labels).tolist())
    cents = np.array([X_sc[labels == ci].mean(0) for ci in centroid_clusters])
    cents_pca = pca.transform(cents)
    ax.scatter(cents_pca[:, 0], cents_pca[:, 1], marker='*', s=280,
               color='#1a1d2e', edgecolors='white', linewidths=1.5,
               zorder=10, label='Centroid')
    ax.set_xlabel(f"PC1 ({var[0]:.1%})", color=MPL_LABEL, fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]:.1%})", color=MPL_LABEL, fontsize=11)
    r_tag = f" — {region_opt}" if region_opt != "Tất cả" else ""
    ax.set_title(f"PCA{r_tag}  |  Explained variance: {var.sum():.1%}",
                 color=MPL_TITLE, fontsize=12, fontweight='700', pad=8)
    ax.legend(facecolor='white', edgecolor='#E5E7EB', fontsize=9, markerscale=1.5)
    st.pyplot(fig, use_container_width=True); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

with stat_c:
    summary = df_view.groupby('cluster')[feats].mean().round(2)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**📊 Thống kê TB theo Cluster**")
    st.dataframe(summary, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
    st.markdown("**📦 Số mẫu mỗi Cluster**")
    counts = df_view['cluster'].value_counts().sort_index()
    for ci, cnt in counts.items():
        pct = cnt / len(df_view) * 100
        color_hex = PAL_HEX[ci % len(PAL_HEX)]
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:5px 0;
                    border-bottom:1px solid #F4F6FB;">
            <div style="width:10px;height:10px;border-radius:50%;
                        background:{color_hex};flex-shrink:0;"></div>
            <span style="font-size:13px;color:#1a1d2e;font-weight:600;">Cluster {ci}</span>
            <span style="font-size:12px;color:#9ca3af;margin-left:auto;">{cnt:,} ({pct:.1f}%)</span>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Region breakdown ──────────────────────────────────
if 'region' in df_cl.columns:
    st.markdown("<br>", unsafe_allow_html=True)
    section("Phân bổ Region × Cluster")
    rc1, rc2 = st.columns(2)
    reg_colors = ['#4F6EF7','#10B981','#F97316']

    with rc1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Số lượng theo Cluster & Region**")
        cross = pd.crosstab(df_view['cluster'], df_view['region'])
        fig, ax = fig_ax(6, 3.5)
        cross.plot(kind='bar', ax=ax, color=reg_colors[:len(cross.columns)],
                   alpha=0.88, width=0.65, edgecolor='none', zorder=3)
        ax.set_xlabel("Cluster", color=MPL_LABEL, fontsize=10)
        ax.set_ylabel("Số lượng", color=MPL_LABEL, fontsize=10)
        ax.set_title("Phân bổ Region", color=MPL_TITLE, fontsize=11, fontweight='700', pad=8)
        ax.legend(facecolor='white', edgecolor='#E5E7EB', fontsize=9)
        ax.tick_params(axis='x', rotation=0)
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with rc2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Tỷ lệ (%) theo Cluster & Region**")
        cn = pd.crosstab(df_view['cluster'], df_view['region'], normalize='index') * 100
        fig, ax = fig_ax(6, 3.5)
        bottom = np.zeros(len(cn))
        for i, reg in enumerate(cn.columns):
            bars = ax.bar(cn.index, cn[reg], bottom=bottom,
                          label=reg, color=reg_colors[i % 3], alpha=0.88, width=0.55, zorder=3)
            for rect, v in zip(bars, cn[reg]):
                if v > 6:
                    ax.text(rect.get_x()+rect.get_width()/2,
                            rect.get_y()+rect.get_height()/2,
                            f'{v:.0f}%', ha='center', va='center',
                            fontsize=8, color='white', fontweight='700')
            bottom += cn[reg].values
        ax.set_title("Tỷ lệ Region (%)", color=MPL_TITLE, fontsize=11, fontweight='700', pad=8)
        ax.set_xlabel("Cluster", color=MPL_LABEL, fontsize=10)
        ax.legend(facecolor='white', edgecolor='#E5E7EB', fontsize=9)
        ax.tick_params(axis='x', rotation=0)
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

# ── Radar chart ───────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section("Radar — Profile Cluster", "ĐẶC TRƯNG TRUNG BÌNH MỖI CỤM")
s_norm = (summary - summary.min()) / (summary.max() - summary.min() + 1e-9)
cats   = list(s_norm.columns)
N      = len(cats)

if N >= 3:
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
    n_profiles = len(summary)
    fig, axes = plt.subplots(1, n_profiles, figsize=(4.2*n_profiles, 4.2),
                              facecolor=MPL_BG, subplot_kw=dict(polar=True))
    if n_profiles == 1:
        axes = [axes]
    for ci, ax in zip(summary.index.tolist(), axes):
        ax.set_facecolor('#FAFBFF')
        profile = s_norm.loc[ci].tolist()
        vals = profile + [profile[0]]
        c    = PAL[ci]
        ax.fill(angles, vals, alpha=0.2, color=c)
        ax.plot(angles, vals, color=c, lw=2.5)
        ax.scatter(angles[:-1], vals[:-1], color=c, s=50, zorder=5)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cats, size=8, color='#4b5563')
        ax.set_yticklabels([]); ax.grid(color='#E5E7EB', lw=0.8)
        ax.set_title(f'Cluster {ci}', color=MPL_TITLE, fontsize=11, fontweight='700', pad=12)
        for spine in ax.spines.values(): spine.set_color('#E5E7EB')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()
else:
    st.info("Cần ít nhất 3 features để vẽ radar chart.")

# ── Extreme heatmap per cluster ───────────────────────
if 'extreme' in df_cl.columns:
    st.markdown("<br>", unsafe_allow_html=True)
    section("Extreme Weather per Cluster (%)")
    col_ext, _ = st.columns([2, 1])
    with col_ext:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        ext_pct = (df_view.groupby('cluster')['extreme']
                   .value_counts(normalize=True).mul(100).round(2)
                   .unstack(fill_value=0))
        fig, ax = fig_ax(7, 3.2)
        sns.heatmap(ext_pct, annot=True, fmt='.1f', cmap='YlOrRd',
                    ax=ax, linewidths=0.3, linecolor='white',
                    annot_kws={'size': 10, 'weight': '600'},
                    cbar_kws={'label': '%', 'shrink': 0.8})
        ax.set_title("Tỷ lệ loại thời tiết theo Cluster (%)",
                     color=MPL_TITLE, fontsize=11, fontweight='700', pad=8)
        ax.tick_params(labelsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

# ── Elbow ─────────────────────────────────────────────
if show_elbow:
    st.markdown("<br>", unsafe_allow_html=True)
    section("Elbow Method", "CHỌN K TỐI ƯU")
    with st.spinner("Đang tính inertia..."):
        inertias, kr = [], range(1, 10)
        bs = min(2048, max(256, len(X_sc) // 20))
        for ki in kr:
            km_ = MiniBatchKMeans(n_clusters=ki, random_state=42,
                                  n_init=5, batch_size=bs, max_iter=100)
            km_.fit(X_sc)
            inertias.append(km_.inertia_)

    col_elbow, _ = st.columns([2, 1])
    with col_elbow:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig, ax = fig_ax(7, 3.5)
        ax.plot(list(kr), inertias, marker='o', color=ACCENT, lw=2.5, ms=8, zorder=5)
        ax.fill_between(list(kr), inertias, alpha=0.07, color=ACCENT)
        ax.scatter([k_val], [inertias[k_val-1]], color='#F97316',
                   s=120, zorder=10, label=f'k={k_val} hiện tại')
        ax.set_xlabel("k", color=MPL_LABEL, fontsize=11)
        ax.set_ylabel("Inertia", color=MPL_LABEL, fontsize=11)
        ax.set_title("Elbow Method", color=MPL_TITLE, fontsize=12, fontweight='700', pad=8)
        ax.legend(facecolor='white', edgecolor='#E5E7EB')
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
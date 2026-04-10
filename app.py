import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import GLOBAL_CSS, load_models, load_csv, root_path

st.set_page_config(
    page_title="WeatherVN",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Sidebar logo ──────────────────────────────────────
from utils import sidebar_header, load_models as _lm
def _app_sidebar():
    models = _lm()
    st.markdown("""
    <div style="margin:12px 0 4px;padding:14px 16px;background:#F8FAFF;
                border-radius:12px;border:1px solid #EEF2FF;">
        <div style="font-size:10px;color:#9ca3af;font-family:'DM Mono',monospace;
                    letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;">
            Model Status
        </div>""", unsafe_allow_html=True)
    for key, label, icon in [("rf","Random Forest","🌲"),("xgb","XGBoost","⚡"),
                               ("le","Label Encoder","🏷️"),("kmeans","KMeans","🗺️"),
                               ("scaler","Scaler","📐")]:
        ok = key in models
        col_txt = "#10B981" if ok else "#EF4444"
        dot = "●" if ok else "○"
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:3px 0;font-size:12px;">
            <span style="color:#4b5563;">{icon} {label}</span>
            <span style="color:{col_txt};font-weight:700;">{dot}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

sidebar_header(_app_sidebar)


# ── Hero section ──────────────────────────────────────
col_hero, col_stats = st.columns([3, 2])

with col_hero:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#4F6EF7 0%,#6C8EFF 50%,#818CF8 100%);
                border-radius:24px; padding:40px 36px; color:white; position:relative;
                overflow:hidden; margin-bottom:20px; min-height:220px;">
        <div style="position:absolute;top:-30px;right:-30px;width:200px;height:200px;
                    border-radius:50%;background:rgba(255,255,255,0.08);"></div>
        <div style="position:absolute;bottom:-50px;right:60px;width:150px;height:150px;
                    border-radius:50%;background:rgba(255,255,255,0.05);"></div>
        <div style="font-size:13px;opacity:.7;letter-spacing:2px;
                    font-family:'DM Mono',monospace;text-transform:uppercase;
                    margin-bottom:12px;">Việt Nam Weather AI</div>
        <div style="font-size:38px;font-weight:800;line-height:1.15;margin-bottom:14px;">
            Dự báo Thời tiết<br>Thông minh
        </div>
        <div style="font-size:14px;opacity:.75;line-height:1.6;max-width:380px;">
            Phân tích mưa &amp; thời tiết cực đoan theo vùng miền Bắc, Trung, Nam
            bằng Random Forest và XGBoost.
        </div>
        <div style="margin-top:20px;display:flex;gap:10px;flex-wrap:wrap;">
            <span style="background:rgba(255,255,255,.18);border-radius:20px;
                         padding:5px 14px;font-size:12px;font-weight:600;">
                🌲 Random Forest
            </span>
            <span style="background:rgba(255,255,255,.18);border-radius:20px;
                         padding:5px 14px;font-size:12px;font-weight:600;">
                ⚡ XGBoost
            </span>
            <span style="background:rgba(255,255,255,.18);border-radius:20px;
                         padding:5px 14px;font-size:12px;font-weight:600;">
                🗺️ KMeans Clustering
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:15px;font-weight:700;color:#1a1d2e;margin-bottom:14px;">Chọn chức năng</div>', unsafe_allow_html=True)

    nc1, nc2 = st.columns(2)
    with nc1:
        if st.button("📊 Tổng quan", use_container_width=True,
                     help="Phân bố dữ liệu & thống kê thời tiết"):
            st.switch_page("pages/overview.py")

        if st.button("🌧️ Dự báo", use_container_width=True,
                     help="Nhập thông số → Kết quả dự báo"):
            st.switch_page("pages/prediction.py")

    with nc2:
        if st.button("🔍 Đánh giá", use_container_width=True,
                     help="So sánh mô hình Random Forest vs XGBoost"):
            st.switch_page("pages/evaluate.py")

        if st.button("🗺️ Clustering", use_container_width=True,
                     help="PCA & phân tích cụm"):
            st.switch_page("pages/clustering.py")

        if st.button("🧩 Apriori", use_container_width=True,
                     help="Phân tích luật kết hợp từ notebook Apriori"):
            st.switch_page("pages/apriori.py")

with col_stats:
    st.markdown("""
    <div style="font-size:15px;font-weight:700;color:#1a1d2e;margin-bottom:14px;">
        Thông tin Dataset
    </div>""", unsafe_allow_html=True)

    # FIX: Try multiple CSV files with fallback
    df = None
    for fname in ["weather_vn_cleaned.csv", "test_data.csv", "train_data.csv"]:
        df = load_csv(fname)
        if df is not None:
            break

    if df is not None:
        total  = len(df)
        rain   = df['rain'].mean()*100      if 'rain'    in df.columns else 0
        ext    = (df['extreme'] != 'normal').mean()*100 if 'extreme' in df.columns else 0
        cities = df['city'].nunique()       if 'city'    in df.columns else "—"
        stats = [
            ("#4F6EF7","#EEF2FF",f"{total:,}",  "Tổng bản ghi","quan sát"),
            ("#10B981","#ECFDF5",f"{rain:.1f}%","Tỷ lệ mưa",   "trung bình"),
            ("#F97316","#FFF7ED",f"{ext:.1f}%", "Cực đoan",    "heatwave/storm"),
            ("#9333EA","#FAF5FF",str(cities),   "Thành phố",   "trong dataset"),
        ]
    else:
        stats = [
            ("#4F6EF7","#EEF2FF","—","Tổng bản ghi","chưa load data"),
            ("#10B981","#ECFDF5","—","Tỷ lệ mưa",   "chưa load data"),
            ("#F97316","#FFF7ED","—","Cực đoan",    "chưa load data"),
            ("#9333EA","#FAF5FF","—","Thành phố",   "chưa load data"),
        ]

    for color, bg, val, label, sub in stats:
        st.markdown(f"""
        <div style="background:{bg};border-radius:14px;padding:18px 20px;
                    margin-bottom:12px;border-left:4px solid {color};">
            <div style="font-size:11px;color:{color};font-family:'DM Mono',monospace;
                        letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;">
                {label}
            </div>
            <div style="font-size:28px;font-weight:800;color:#1a1d2e;line-height:1;">
                {val}
            </div>
            <div style="font-size:11px;color:#9ca3af;margin-top:4px;">{sub}</div>
        </div>""", unsafe_allow_html=True)

# ── Files needed notice ───────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="background:#FFFBEB;border-radius:14px;padding:18px 22px;
            border:1px solid #FDE68A;display:flex;align-items:center;gap:12px;">
    <span style="font-size:20px;">📁</span>
    <div>
        <div style="font-size:13px;font-weight:700;color:#92400E;">
            File cần đặt cùng thư mục với app.py (hoặc trong thư mục data/)
        </div>
        <div style="font-size:12px;color:#B45309;margin-top:3px;font-family:'DM Mono',monospace;">
            rf_model.pkl &nbsp;·&nbsp; xgb_model.pkl &nbsp;·&nbsp; label_encoder.pkl
            &nbsp;·&nbsp; kmeans_model.pkl &nbsp;·&nbsp; scaler.pkl
            &nbsp;·&nbsp; weather_vn_cleaned.csv &nbsp;·&nbsp; test_data.csv
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
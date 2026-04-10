"""
utils.py — shared helpers for WeatherVN Dashboard
Place this file in the ROOT folder (same level as app.py)
"""
import streamlit as st
import joblib, os, pandas as pd, numpy as np

# ── Path resolution ───────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

def root_path(filename):
    return os.path.join(ROOT, filename)

def find_file(filename):
    candidates = [
        root_path(filename),
        root_path(os.path.join("data", filename)),
        filename,
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# ── Global CSS ────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, .stApp, .stMarkdown, .stText, .stButton, .stSelectbox, .stNumberInput, .stDataFrame {
    font-family: 'Manrope', sans-serif !important;
}

.stApp { background: #F4F6FB !important; color: #1a1d2e !important; }

/* Hide default chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

.stApp [data-testid="stSidebar"],
.stApp [data-testid="stSidebar"] > div,
.stApp [data-testid="stSidebarContent"] {
    background: #FFFFFF !important;
}

[data-testid="stSidebar"] {
    border-right: 1px solid #EEF0F6 !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.04) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }
[data-testid="stSidebarContent"] { padding-top: 0 !important; }
[data-testid="stSidebarNav"] { padding-top: 8px !important; }
[data-testid="stSidebarNav"] ul { padding: 0 10px !important; }
[data-testid="stSidebarNav"] a {
    border-radius: 10px !important;
    padding: 9px 14px !important;
    margin: 2px 0 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #6b7280 !important;
    transition: all .18s !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}
[data-testid="stSidebarNav"] a:hover {
    background: #F0F4FF !important;
    color: #4F6EF7 !important;
}
[data-testid="stSidebarNav"] [aria-current="page"] {
    background: linear-gradient(135deg, #EEF2FF, #E0E7FF) !important;
    color: #4F6EF7 !important;
    font-weight: 700 !important;
}
[data-testid="stSidebarCollapseButton"] {
    color: #4F6EF7 !important;
}

/* Cards */
.card {
    background: #FFFFFF; border-radius: 18px;
    padding: 22px 24px;
    box-shadow: 0 2px 12px rgba(30,40,80,0.06);
    border: 1px solid #F0F2F8;
    transition: box-shadow .2s, transform .2s;
    margin-bottom: 4px;
}
.card:hover { box-shadow: 0 6px 24px rgba(79,110,247,0.10); transform: translateY(-2px); }

/* KPI */
.kpi {
    background: #FFFFFF; border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 2px 10px rgba(30,40,80,0.06);
    border: 1px solid #F0F2F8; transition: all .2s;
}
.kpi:hover { box-shadow: 0 6px 20px rgba(79,110,247,0.10); transform: translateY(-2px); }
.kpi-label {
    font-family: 'DM Mono', monospace; font-size: 10px;
    font-weight: 500; color: #9ca3af;
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 10px;
}
.kpi-value { font-size: 30px; font-weight: 800; color: #1a1d2e; line-height: 1; }
.kpi-badge { display:inline-block; margin-top:8px; font-size:11px; font-weight:600; padding:3px 8px; border-radius:20px; }
.badge-up   { background:#ECFDF5; color:#10B981; }
.badge-down { background:#FEF2F2; color:#EF4444; }
.badge-info { background:#EEF2FF; color:#4F6EF7; }

/* Section */
.sec-title { font-size:17px; font-weight:700; color:#1a1d2e; margin:4px 0 16px 0; }
.sec-sub { font-size:12px; color:#9ca3af; font-family:'DM Mono',monospace; letter-spacing:.5px; }

/* Page header */
.page-header {
    background:#FFFFFF; border-radius:18px; padding:24px 28px;
    margin-bottom:22px; box-shadow:0 2px 12px rgba(30,40,80,0.05);
    border:1px solid #F0F2F8; display:flex; align-items:center; gap:16px;
}
.page-icon { width:50px; height:50px; border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:24px; }
.page-title { font-size:22px; font-weight:800; color:#1a1d2e; line-height:1; }
.page-desc  { font-size:13px; color:#9ca3af; margin-top:4px; }

/* Prediction */
.pred-result { border-radius:16px; padding:24px; text-align:center; transition:all .2s; }
.pred-normal   { background:#F0FDF4; border:2px solid #86EFAC; }
.pred-heatwave { background:#FFF7ED; border:2px solid #FED7AA; }
.pred-storm    { background:#FAF5FF; border:2px solid #D8B4FE; }

/* Input group */
.input-group {
    background:#F8FAFF; border-radius:12px;
    padding:16px 18px; margin-bottom:10px;
    border:1px solid #EEF2FF;
}
.input-group-title {
    font-size:12px; font-weight:700; color:#4F6EF7;
    margin-bottom:12px; display:flex; align-items:center; gap:6px;
    text-transform:uppercase; letter-spacing:.5px;
    font-family:'DM Mono',monospace;
}

/* Buttons */
.stButton > button {
    background:linear-gradient(135deg,#4F6EF7,#6C8EFF) !important;
    color:#fff !important; border:none !important;
    border-radius:12px !important;
    font-family:'Manrope',sans-serif !important;
    font-weight:700 !important; font-size:14px !important;
    padding:10px 28px !important;
    box-shadow:0 4px 14px rgba(79,110,247,0.35) !important;
    transition:all .2s !important;
}
.stButton > button:hover {
    background:linear-gradient(135deg,#3d5be8,#5a7aff) !important;
    box-shadow:0 6px 20px rgba(79,110,247,0.45) !important;
    transform:translateY(-1px) !important;
}

/* number_input — làm nổi bật hơn */
[data-testid="stNumberInput"] input {
    font-size:16px !important; font-weight:700 !important;
    color:#1a1d2e !important; border-radius:10px !important;
    border:1.5px solid #E5E7EB !important;
    background:#FFFFFF !important;
    text-align:center !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color:#4F6EF7 !important;
    box-shadow:0 0 0 3px rgba(79,110,247,0.12) !important;
}
[data-testid="stNumberInput"] button {
    background:#F0F4FF !important; border-color:#E5E7EB !important;
    color:#4F6EF7 !important;
}

/* selectbox */
[data-baseweb="select"] { border-radius:10px !important; border-color:#E5E7EB !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background:#F4F6FB; border-radius:12px; padding:4px; gap:2px; border:none; }
.stTabs [data-baseweb="tab"] { background:transparent; border-radius:9px; color:#6b7280 !important; font-weight:600; font-size:13px; padding:7px 18px; }
.stTabs [aria-selected="true"] { background:#FFFFFF !important; color:#4F6EF7 !important; box-shadow:0 2px 8px rgba(30,40,80,0.08) !important; }

.stAlert { border-radius:12px !important; }
hr { border-color:#F0F2F8 !important; }

/* insight box */
.insight-box {
    background:#F0F4FF; border-left:4px solid #4F6EF7;
    border-radius:0 10px 10px 0; padding:10px 14px;
    font-size:12px; color:#374151; margin-top:8px;
}
.metric-pill {
    display:inline-flex; align-items:center; gap:6px;
    background:#EEF2FF; border-radius:20px;
    padding:4px 12px; font-size:11px; font-weight:600; color:#4F6EF7; margin:2px;
}

.sb-brand {
    margin: 12px 0 10px;
    background: linear-gradient(135deg,#4F6EF7,#6C8EFF);
    border-radius: 14px;
    color: #FFFFFF;
    padding: 14px 14px 12px;
}
.sb-brand-title {
    font-size: 15px;
    font-weight: 800;
    line-height: 1.2;
}
.sb-brand-sub {
    font-size: 11px;
    opacity: 0.9;
    margin-top: 4px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.6px;
}
</style>
"""

# ── Matplotlib theme ──────────────────────────────────
MPL_BG    = "#FFFFFF"
MPL_AX_BG = "#FAFBFF"
MPL_SPINE = "#E5E7EB"
MPL_TICK  = "#9ca3af"
MPL_TITLE = "#1a1d2e"
MPL_LABEL = "#6b7280"
ACCENT    = "#4F6EF7"
PALETTE   = ["#4F6EF7","#10B981","#F97316","#9333EA","#EF4444","#06B6D4","#F59E0B","#EC4899"]

def style_ax(ax, fig=None):
    if fig: fig.patch.set_facecolor(MPL_BG)
    ax.set_facecolor(MPL_AX_BG)
    for sp in ax.spines.values():
        sp.set_color(MPL_SPINE); sp.set_linewidth(0.8)
    ax.tick_params(colors=MPL_TICK, labelsize=9)
    ax.grid(color='#F0F2F8', linewidth=0.7, linestyle='-', axis='y')
    ax.set_axisbelow(True)

def fig_ax(w=7, h=4):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w, h), facecolor=MPL_BG)
    style_ax(ax, fig)
    return fig, ax

# ── HTML helpers ──────────────────────────────────────
def kpi_card(label, value, badge="", badge_type="info"):
    badge_html = f'<div class="kpi-badge badge-{badge_type}">{badge}</div>' if badge else ""
    return f"""<div class="kpi">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {badge_html}
    </div>"""

def page_header(icon, color, title, desc):
    st.markdown(f"""
    <div class="page-header">
        <div class="page-icon" style="background:{color};">{icon}</div>
        <div>
            <div class="page-title">{title}</div>
            <div class="page-desc">{desc}</div>
        </div>
    </div>""", unsafe_allow_html=True)

def section(title, sub=""):
    sub_html = f'<div class="sec-sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="sec-title">{title}</div>{sub_html}', unsafe_allow_html=True)

# ── Sidebar header — gọi 1 lần duy nhất ở mỗi page ───
def sidebar_header(extra_content_fn=None):
    with st.sidebar:
        st.markdown(
            """
            <div class="sb-brand">
                <div class="sb-brand-title">WeatherVN Dashboard</div>
                <div class="sb-brand-sub">AI WEATHER · RF · XGB · KMEANS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if extra_content_fn:
            extra_content_fn()

# ── Loaders ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    out = {}
    for key, fname in [("rf","rf_model.pkl"),("xgb","xgb_model.pkl"),
                        ("rf_rain","rf_rain_model.pkl"),("xgb_rain","xgb_rain_model.pkl"),
                        ("le","label_encoder.pkl"),("kmeans","kmeans_model.pkl"),
                        ("scaler","scaler.pkl")]:
        p = find_file(fname)
        if p:
            try:
                out[key] = joblib.load(p)
            except Exception:
                continue
    return out

@st.cache_data(show_spinner=False)
def load_csv(name):
    p = find_file(name)
    if p:
        try: return pd.read_csv(p)
        except: return None
    return None

# ── Constants ─────────────────────────────────────────
DROP_COLS = [
    'extreme','rain','time','weather_code','weather_main','weather_description',
    'weather_icon','temperature','temp_min','temp_max','feels_like','temp_range',
    'wind_speed','wind_gust','rainfall','precipitation',
    'temp_level','humidity_level','pressure_level','wind_level','city','region'
]
CLUSTER_FEATURES = ['temperature','humidity','pressure','wind_speed']
import html
import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import GLOBAL_CSS, page_header, sidebar_header


st.set_page_config(
    page_title="Notebook Apriori - WeatherVN",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.markdown(
    """
<style>
.nb-shell {
    background:#ffffff;
    border:1px solid #EEF2FF;
    border-radius:16px;
    padding:18px;
    box-shadow:0 2px 10px rgba(30,40,80,0.05);
}
.nb-title {
    font-size:16px;
    font-weight:800;
    color:#1a1d2e;
    margin-bottom:6px;
}
.nb-sub {
    font-size:12px;
    color:#64748b;
    line-height:1.6;
}
.nb-code {
    background:#0f172a;
    color:#e2e8f0;
    border-radius:14px;
    padding:14px 16px;
    font-family:'DM Mono', monospace;
    font-size:12px;
    line-height:1.6;
    white-space:pre-wrap;
    overflow-x:auto;
}
.nb-quote {
    font-size:13px;
    color:#334155;
    background:#fafafa;
    border:1px solid #e5e7eb;
    border-radius:12px;
    padding:10px 12px;
    line-height:1.6;
}
.nb-note {
    background:#F8FAFF;
    border:1px solid #E0E7FF;
    border-left:4px solid #4F6EF7;
    border-radius:12px;
    padding:12px 14px;
    font-size:13px;
    color:#334155;
    line-height:1.6;
}
.nb-step-no {
    min-width:30px;
    height:30px;
    border-radius:9px;
    background:linear-gradient(135deg,#4F6EF7,#6C8EFF);
    color:#fff;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    font-weight:800;
    font-size:13px;
    margin-right:10px;
}
</style>
""",
    unsafe_allow_html=True,
)

NOTEBOOK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notebook",
    "apriori.ipynb",
)


def sidebar_content():
    st.markdown(
        """
        <div style="padding:12px 4px 8px;font-size:11px;font-weight:700;color:#374151;
                    text-transform:uppercase;letter-spacing:.8px;font-family:'DM Mono',monospace;">
            Notebook Apriori
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Trang này hiển thị trực tiếp nội dung notebook/apriori.ipynb.")


sidebar_header(sidebar_content)

page_header(
    "🧩",
    "linear-gradient(135deg,#EEF2FF,#E0E7FF)",
    "Notebook Apriori",
    "Trình chiếu trực tiếp nội dung notebook/apriori.ipynb theo từng mục.",
)

if not os.path.exists(NOTEBOOK_PATH):
    st.error("Không tìm thấy notebook/apriori.ipynb.")
    st.stop()

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    notebook = json.load(f)

cells = notebook.get("cells", [])
sections = []
current = None

for cell in cells:
    text = cell.get("source", [])
    if isinstance(text, list):
        text = "".join(text)
    text = text.strip()
    if not text:
        continue

    if cell.get("cell_type") == "markdown" and text.startswith("## "):
        current = {"title": text.replace("## ", "", 1).strip(), "items": []}
        sections.append(current)
        continue

    if current is None:
        current = {"title": "Mở đầu", "items": []}
        sections.append(current)

    current["items"].append({
        "type": cell.get("cell_type", ""),
        "text": text,
    })

st.markdown(
    """
    <div class="nb-note">
        Đây là bản trình chiếu trực tiếp từ <b>notebook/apriori.ipynb</b>.
        Không dùng các file trong <b>checkpoint</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

for idx, sec in enumerate(sections, start=1):
    st.markdown('<div class="nb-shell">', unsafe_allow_html=True)
    st.markdown(f'<div class="nb-title">{html.escape(sec["title"])}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="nb-sub">Mục {idx} trong notebook</div>', unsafe_allow_html=True)

    for jdx, item in enumerate(sec["items"], start=1):
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if item["type"] == "markdown":
            quote_html = html.escape(item["text"]).replace("\n", "<br>")
            st.markdown(
                f'<div class="nb-quote">{quote_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div><span class="nb-step-no">{jdx}</span><span class="nb-sub">Code cell</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="nb-code">{html.escape(item["text"])}</div>',
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="nb-note">
        Nếu bạn muốn, mình có thể tiếp tục làm chế độ xem giống notebook hơn nữa: thu gọn/mở rộng từng cell, hoặc render cả output hình ảnh từ notebook.
    </div>
    """,
    unsafe_allow_html=True,
)

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import get_theme

st.set_page_config(
    page_title="股票量化监测系统",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "light"

_PAGES_DIR = os.path.join(os.path.dirname(__file__), "pages")

pages = st.navigation([
    st.Page(os.path.join(_PAGES_DIR, "stock_dashboard.py"), title="自选股总览", icon=":material/dashboard:", default=True),
    st.Page(os.path.join(_PAGES_DIR, "kline_detail.py"), title="K线详情", icon=":material/candlestick_chart:"),
    st.Page(os.path.join(_PAGES_DIR, "prediction.py"), title="AI预测", icon=":material/psychology:"),
    st.Page(os.path.join(_PAGES_DIR, "news_sentiment.py"), title="新闻舆情", icon=":material/newspaper:"),
    st.Page(os.path.join(_PAGES_DIR, "settings.py"), title="系统设置", icon=":material/settings:"),
])

st.sidebar.markdown("""
<div class="mac-sidebar-header">
    量化监测系统
</div>
""", unsafe_allow_html=True)

current_mode = st.session_state["theme_mode"]
is_dark = current_mode == "dark"

if st.sidebar.toggle("暗色模式", value=is_dark, key="theme_toggle"):
    new_mode = "dark"
else:
    new_mode = "light"

if new_mode != current_mode:
    st.session_state["theme_mode"] = new_mode
    try:
        theme = get_theme(new_mode)
        st._config.set_option("theme.base", new_mode)
        st._config.set_option("theme.backgroundColor", theme["bg_primary"])
        st._config.set_option("theme.secondaryBackgroundColor", theme["bg_secondary"])
        st._config.set_option("theme.textColor", theme["text_primary"])
        st._config.set_option("theme.primaryColor", theme["accent_blue"])
    except Exception:
        pass
    st.rerun()


def inject_global_css(theme: dict, mode: str) -> None:
    is_dark = mode == "dark"
    sidebar_hover_bg = "rgba(255,255,255,0.06)" if is_dark else "rgba(0,0,0,0.04)"
    sidebar_active_bg = "rgba(255,255,255,0.12)" if is_dark else "rgba(0,0,0,0.08)"
    sidebar_divider = "rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.06)"
    sidebar_text_active = "#FFFFFF" if is_dark else "#1D1D1F"

    css = f"""
    <style>
    .stApp {{
        background-color: {theme["bg_primary"]};
        font-family: {theme["font_family"]};
    }}

    /* ── macOS 风格侧边栏 ── */
    [data-testid="stSidebar"] {{
        background-color: {theme["bg_secondary"]};
        border-right: 1px solid {sidebar_divider};
    }}

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
        padding: 0 12px;
    }}

    .mac-sidebar-header {{
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
        font-size: 15px;
        font-weight: 600;
        color: {theme["text_primary"]};
        letter-spacing: 0.3px;
        padding: 24px 8px 16px 8px;
        margin-bottom: 12px;
        border-bottom: 1px solid {sidebar_divider};
    }}

    /* 导航项 */
    [data-testid="stSidebarNavLink"] {{
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 2px;
        transition: all 0.2s ease-in-out;
        color: {theme["text_secondary"]};
        font-weight: 500;
        font-size: 14px;
        display: flex;
        align-items: center;
    }}

    [data-testid="stSidebarNavLink"]:hover {{
        background-color: {sidebar_hover_bg};
        color: {theme["text_primary"]};
    }}

    [data-testid="stSidebarNavLink"][aria-current="page"] {{
        background-color: {sidebar_active_bg};
        color: {sidebar_text_active};
        font-weight: 600;
    }}

    /* 图标颜色调整 */
    [data-testid="stSidebarNavLink"] svg {{
        margin-right: 10px;
        opacity: 0.85;
        transition: color 0.2s;
    }}

    [data-testid="stSidebarNavLink"][aria-current="page"] svg {{
        opacity: 1;
        color: {theme["accent_blue"]};
    }}

    /* 暗色模式 Toggle 开关调整 */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
        font-size: 14px;
        font-weight: 500;
        color: {theme["text_secondary"]};
    }}

    /* ── 现有组件样式保留 ── */
    [data-testid="stMetric"] {{
        background: {theme["card_bg"]};
        border: 1px solid {theme["card_border"]};
        border-radius: 18px;
        padding: 16px 20px;
        box-shadow: {theme["card_shadow"]};
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }}

    .stButton > button {{
        border-radius: 8px;
        background-color: {theme["accent_blue"]};
        color: #FFFFFF;
        border: none;
        font-family: {theme["font_family"]};
        font-weight: 500;
        transition: opacity 0.2s;
    }}

    .stButton > button:hover {{
        opacity: 0.85;
    }}

    [data-testid="stDataFrame"] {{
        border-radius: 12px;
        overflow: hidden;
    }}

    button[data-testid="stTab"] {{
        font-family: {theme["font_family"]};
        font-weight: 500;
        color: {theme["text_secondary"]};
        border-bottom: 2px solid transparent;
    }}

    button[data-testid="stTab"][aria-selected="true"] {{
        color: {theme["accent_blue"]};
        border-bottom-color: {theme["accent_blue"]};
    }}

    h1, h2, h3 {{
        font-family: {theme["font_family"]};
        color: {theme["text_primary"]};
    }}

    h1 > a, h2 > a, h3 > a,
    [data-testid="stHeadingWithActionElements"] a[href] {{
        display: none !important;
    }}

    [data-testid="stForm"] {{
        border-radius: 18px;
        border: 1px solid {theme["card_border"]};
        background: {theme["card_bg"]};
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }}

    .glass-card {{
        background: {theme["glass_bg"]};
        border: 1px solid {theme["glass_border"]};
        border-radius: 18px;
        padding: 24px;
        box-shadow: {theme["card_shadow"]};
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }}

    /* ── 侧边栏分组标题（Apple 风格） ── */
    .sidebar-section-title {{
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: {theme["text_tertiary"]};
        padding: 8px 4px 6px 4px;
        margin-bottom: 4px;
    }}

    /* ── 侧边栏控件 Apple 精调 ── */
    [data-testid="stSidebar"] [data-testid="stSelectbox"],
    [data-testid="stSidebar"] [data-testid="stMultiSelect"],
    [data-testid="stSidebar"] [data-testid="stDateInput"],
    [data-testid="stSidebar"] [data-testid="stCheckbox"] {{
        margin-bottom: 4px;
    }}

    [data-testid="stSidebar"] [data-testid="stSelectbox"] label p,
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] label p,
    [data-testid="stSidebar"] [data-testid="stDateInput"] label p,
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label p {{
        font-size: 13px;
        font-weight: 500;
        color: {theme["text_secondary"]};
        font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
    }}

    [data-testid="stSidebar"] [data-baseweb="select"] {{
        font-size: 13px;
    }}

    [data-testid="stSidebar"] [data-baseweb="select"] > div {{
        border-radius: 8px;
        border-color: {theme["card_border"]};
        background-color: {theme["bg_primary"]};
        min-height: 36px;
        transition: border-color 0.2s ease;
    }}

    [data-testid="stSidebar"] [data-baseweb="select"] > div:hover {{
        border-color: {theme["accent_blue"]};
    }}

    [data-testid="stSidebar"] [data-baseweb="input"] {{
        font-size: 13px;
        border-radius: 8px;
    }}

    [data-testid="stSidebar"] [data-baseweb="input"] > div {{
        border-radius: 8px;
        border-color: {theme["card_border"]};
        background-color: {theme["bg_primary"]};
        min-height: 36px;
    }}

    [data-testid="stSidebar"] [data-testid="stDateInput"] input {{
        font-size: 13px;
        font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
        padding: 6px 10px;
        width: 100%;
        box-sizing: border-box;
    }}

    [data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="input"] > div {{
        min-height: 36px;
        border-radius: 8px;
        border-color: {theme["card_border"]};
        background-color: {theme["bg_primary"]};
    }}

    [data-testid="stSidebar"] .stCheckbox label {{
        font-size: 13px;
        gap: 8px;
        align-items: center;
    }}

    [data-testid="stSidebar"] .stCheckbox label span[data-baseweb="checkbox"] {{
        width: 18px;
        height: 18px;
        border-radius: 5px;
    }}

    /* 侧边栏内 columns 间距紧凑 */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{
        gap: 8px;
    }}

    /* ── 隐藏 Streamlit 自带的 Deploy 按钮和 Made with Streamlit 页脚 ── */
    [data-testid="stToolbar"] {{
        display: none !important;
    }}
    footer {{
        display: none !important;
    }}
    #MainMenu {{
        display: none !important;
    }}

    /* ── components.html iframe 透明 + 淡入 ── */
    iframe[title="streamlit_components.v1.components.html"] {{
        background: transparent !important;
        opacity: 0;
        animation: iframeFadeIn 150ms ease-out 50ms forwards;
    }}
    .stCustomComponentV1 iframe,
    [data-testid="stCustomComponentV1"] iframe {{
        background: transparent !important;
        opacity: 0;
        animation: iframeFadeIn 150ms ease-out 50ms forwards;
    }}
    @keyframes iframeFadeIn {{
        to {{ opacity: 1; }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


theme = get_theme(st.session_state["theme_mode"])
inject_global_css(theme, st.session_state["theme_mode"])

pages.run()

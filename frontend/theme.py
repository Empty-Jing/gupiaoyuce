# 纯 Python 主题模块（不导入 Streamlit）— A股配色：涨红跌绿

_FONT_FAMILY = (
    "-apple-system, BlinkMacSystemFont, 'SF Pro Display', "
    "'Helvetica Neue', 'PingFang SC', 'Microsoft YaHei', sans-serif"
)

LIGHT_THEME: dict = {
    # ── 页面背景 ──
    "bg_primary": "#FFFFFF",
    "bg_secondary": "#F5F5F7",

    # ── 文字 ──
    "text_primary": "#1D1D1F",
    "text_secondary": "#86868B",
    "text_tertiary": "#AEAEB2",

    # ── A股涨跌 ──
    "rise_color": "#FF3B30",
    "fall_color": "#34C759",

    # ── 强调色 ──
    "accent_blue": "#007AFF",

    # ── 卡片 ──
    "card_bg": "rgba(255,255,255,0.72)",
    "card_border": "rgba(0,0,0,0.06)",
    "card_shadow": "0 4px 24px rgba(0,0,0,0.08)",

    # ── 毛玻璃 ──
    "glass_bg": "rgba(255,255,255,0.72)",
    "glass_border": "rgba(0,0,0,0.06)",

    # ── Plotly 图表 ──
    "plotly_paper_bg": "#FFFFFF",
    "plotly_plot_bg": "#FFFFFF",
    "plotly_font_color": "#1D1D1F",
    "plotly_gridcolor": "rgba(0,0,0,0.06)",
    "plotly_spike_color": "#86868B",

    # ── MA 线颜色（4色）──
    "ma_colors": ["#007AFF", "#FF9500", "#AF52DE", "#FF3B30"],

    # ── BOLL 线颜色 ──
    "boll_colors": {
        "upper": "#86868B",
        "mid": "#FF9500",
        "lower": "#86868B",
    },

    # ── MACD 颜色 ──
    "macd_colors": {
        "line": "#007AFF",
        "signal": "#FF9500",
        "hist_pos": "#FF3B30",
        "hist_neg": "#34C759",
    },

    # ── KDJ 颜色 ──
    "kdj_colors": {
        "k": "#1D1D1F",
        "d": "#007AFF",
        "j": "#AF52DE",
    },

    # ── RSI ──
    "rsi_hline_color": "#86868B",

    # ── 信号标记 ──
    "signal_buy_color": "#FF3B30",
    "signal_sell_color": "#34C759",

    # ── OHLC 面板（亮色翻转：白底深字）──
    "ohlc_panel_bg": "rgba(255,255,255,0.92)",
    "ohlc_panel_text": "#1D1D1F",
    "ohlc_panel_title": "#1D1D1F",
    "ohlc_panel_high": "#FF3B30",
    "ohlc_panel_low": "#34C759",

    # ── 情绪 badge ──
    "badge_positive": "#FF3B30",
    "badge_negative": "#34C759",
    "badge_neutral": "#86868B",
    "badge_text_color": "#FFFFFF",

    # ── Gauge 图 ──
    "gauge_bg": "#FFFFFF",
    "gauge_border": "#86868B",
    "gauge_tick_color": "#1D1D1F",
    "gauge_threshold_color": "#FF3B30",
    "gauge_steps": [
        "rgba(52,199,89,0.3)",
        "rgba(255,149,0,0.3)",
        "rgba(255,59,48,0.3)",
    ],

    # ── Pie 图 ──
    "pie_colors": ["#007AFF", "#FF9500", "#34C759"],

    # ── 情绪趋势 ──
    "sentiment_line_color": "#007AFF",
    "sentiment_fill_color": "rgba(0,122,255,0.2)",

    # ── 字体 ──
    "font_family": _FONT_FAMILY,
}

DARK_THEME: dict = {
    # ── 页面背景 ──
    "bg_primary": "#000000",
    "bg_secondary": "#1C1C1E",

    # ── 文字 ──
    "text_primary": "#F5F5F7",
    "text_secondary": "#86868B",
    "text_tertiary": "#636366",

    # ── A股涨跌 ──
    "rise_color": "#FF453A",
    "fall_color": "#30D158",

    # ── 强调色 ──
    "accent_blue": "#0A84FF",

    # ── 卡片 ──
    "card_bg": "rgba(28,28,30,0.72)",
    "card_border": "rgba(255,255,255,0.1)",
    "card_shadow": "0 4px 24px rgba(0,0,0,0.3)",

    # ── 毛玻璃 ──
    "glass_bg": "rgba(28,28,30,0.72)",
    "glass_border": "rgba(255,255,255,0.1)",

    # ── Plotly 图表 ──
    "plotly_paper_bg": "#000000",
    "plotly_plot_bg": "#1C1C1E",
    "plotly_font_color": "#F5F5F7",
    "plotly_gridcolor": "rgba(255,255,255,0.08)",
    "plotly_spike_color": "#86868B",

    # ── MA 线颜色（4色）──
    "ma_colors": ["#0A84FF", "#FF9F0A", "#BF5AF2", "#FF453A"],

    # ── BOLL 线颜色 ──
    "boll_colors": {
        "upper": "#86868B",
        "mid": "#FF9F0A",
        "lower": "#86868B",
    },

    # ── MACD 颜色 ──
    "macd_colors": {
        "line": "#0A84FF",
        "signal": "#FF9F0A",
        "hist_pos": "#FF453A",
        "hist_neg": "#30D158",
    },

    # ── KDJ 颜色（K 线用 #F5F5F7 替代 black，否则暗色不可见）──
    "kdj_colors": {
        "k": "#F5F5F7",
        "d": "#0A84FF",
        "j": "#BF5AF2",
    },

    # ── RSI ──
    "rsi_hline_color": "#86868B",

    # ── 信号标记 ──
    "signal_buy_color": "#FF453A",
    "signal_sell_color": "#30D158",

    # ── OHLC 面板（暗色：深底亮字）──
    "ohlc_panel_bg": "rgba(28,28,30,0.92)",
    "ohlc_panel_text": "#F5F5F7",
    "ohlc_panel_title": "#F5F5F7",
    "ohlc_panel_high": "#FF453A",
    "ohlc_panel_low": "#30D158",

    # ── 情绪 badge ──
    "badge_positive": "#FF453A",
    "badge_negative": "#30D158",
    "badge_neutral": "#86868B",
    "badge_text_color": "#FFFFFF",

    # ── Gauge 图 ──
    "gauge_bg": "#1C1C1E",
    "gauge_border": "#86868B",
    "gauge_tick_color": "#F5F5F7",
    "gauge_threshold_color": "#FF453A",
    "gauge_steps": [
        "rgba(48,209,88,0.3)",
        "rgba(255,159,10,0.3)",
        "rgba(255,69,58,0.3)",
    ],

    # ── Pie 图 ──
    "pie_colors": ["#0A84FF", "#FF9F0A", "#30D158"],

    # ── 情绪趋势 ──
    "sentiment_line_color": "#0A84FF",
    "sentiment_fill_color": "rgba(10,132,255,0.2)",

    # ── 字体 ──
    "font_family": _FONT_FAMILY,
}


def get_theme(mode: str = "light") -> dict:
    if mode == "dark":
        return DARK_THEME
    return LIGHT_THEME

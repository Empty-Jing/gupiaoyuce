import sys
import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.widgets import safe_api_call
from components.charts import render_kline_html, render_kline, render_signal_markers
from theme import get_theme

# 预加载 Plotly CDN — 让浏览器在 iframe 创建前开始获取脚本
st.markdown(
    '<link rel="preload" href="https://cdn.plot.ly/plotly-2.35.0.min.js" as="script" crossorigin>',
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="sidebar-section-title">图表设置</div>',
    unsafe_allow_html=True,
)

watchlist = safe_api_call("GET", "/api/stocks/")
if not watchlist:
    st.warning("自选股为空。请先在总览页面添加股票。")
    st.stop()

stock_options = {f"{s['name']} ({s['symbol']})": s['symbol'] for s in watchlist}
selected_label = st.sidebar.selectbox("选择股票", list(stock_options.keys()))
selected_symbol = stock_options[selected_label]

period = st.sidebar.selectbox("K线周期", ["daily", "weekly", "monthly"], format_func=lambda x: {"daily": "日线", "weekly": "周线", "monthly": "月线"}[x])

start_date = st.sidebar.date_input("开始日期", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("结束日期", datetime.now())

st.sidebar.markdown(
    '<div class="sidebar-section-title" style="margin-top: 20px;">技术指标</div>',
    unsafe_allow_html=True,
)
indicators_selected = st.sidebar.multiselect(
    "选择要叠加的指标",
    ["MA", "BOLL", "MACD", "RSI", "KDJ"],
    default=["MA", "MACD"]
)

show_signals = st.sidebar.checkbox("标记买卖信号 (AI生成)", value=True)

with st.spinner("获取历史行情和指标数据..."):
    params = {
        "period": period,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "adjust": "qfq"
    }

    history_data = safe_api_call("GET", f"/api/stocks/{selected_symbol}/history", params=params)

    if not history_data:
        st.error("未能获取到历史行情数据。")
        st.stop()

    df = pd.DataFrame(history_data)

    theme = get_theme(st.session_state.get("theme_mode", "light"))
    fig_kline = render_kline(df, indicators=indicators_selected, show_volume=True, theme=theme, title=f"{selected_label} 走势图")
    if show_signals:
        fig_kline = render_signal_markers(fig_kline, df, theme=theme)
    kline_html, chart_h = render_kline_html(df, indicators=indicators_selected, show_volume=True, fig=fig_kline, theme=theme)

    components.html(kline_html, height=chart_h, scrolling=False)

    with st.expander("查看数据明细"):
        df_display = df.sort_values('日期', ascending=False).copy()

        # 去掉不需要展示的内部列
        df_display = df_display.drop(columns=[c for c in ['signal'] if c in df_display.columns])

        # 价格列保留2位小数
        _price_cols = ['开盘', '收盘', '最高', '最低']
        for col in _price_cols:
            if col in df_display.columns:
                df_display[col] = df_display[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "--")

        # 成交量格式化为万手（原始单位：手）
        if '成交量' in df_display.columns:
            df_display['成交量'] = df_display['成交量'].map(
                lambda v: f"{v / 10000:.1f}万" if pd.notna(v) and isinstance(v, (int, float)) else "--"
            )

        # 技术指标保留2位小数
        for col in df_display.columns:
            if col not in ['日期'] + _price_cols + ['成交量']:
                try:
                    df_display[col] = df_display[col].map(
                        lambda v: f"{v:.2f}" if pd.notna(v) and isinstance(v, (int, float)) else ("--" if pd.isna(v) else v)
                    )
                except Exception:
                    pass

        st.dataframe(df_display, use_container_width=True, hide_index=True)

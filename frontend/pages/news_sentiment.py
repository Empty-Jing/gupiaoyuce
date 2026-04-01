import sys
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.widgets import safe_api_call
from theme import get_theme

# --- 股票选择器 ---
col1, col2 = st.columns([1, 2], vertical_alignment="bottom")
with col1:
    watchlist = safe_api_call("GET", "/api/stocks/")
    if not watchlist:
        st.warning("自选股为空。请先添加股票。")
        st.stop()

    stock_options = {f"{s['name']} ({s['symbol']})": s['symbol'] for s in watchlist}
    selected_label = st.selectbox("选择相关股票", ["全部"] + list(stock_options.keys()))
    selected_symbol = stock_options[selected_label] if selected_label != "全部" else ""

with col2:
    days = st.selectbox("时间范围", [3, 7, 14, 30], index=1, format_func=lambda x: f"最近 {x} 天")

# --- 数据获取 ---
with st.spinner("正在获取最新资讯..."):
    news_params = {"days": days}
    if selected_symbol:
        news_params["symbol"] = selected_symbol
    news_data = safe_api_call("GET", "/api/news/", params=news_params)

    alert_params = {}
    if selected_symbol:
        alert_params["symbol"] = selected_symbol
    alert_params["is_read"] = False
    alerts_data = safe_api_call("GET", "/api/alerts/", params=alert_params)

# --- 新闻列表 + 预警/舆情 ---
tab_news, tab_alert, tab_trend = st.tabs(["新闻资讯", "异动预警", "舆情趋势"])

# ====== TAB 1: 新闻资讯 ======
with tab_news:
    if news_data and len(news_data) > 0:
        st.caption(f"共 {len(news_data)} 条新闻")
        for article in news_data:
            title = article.get("title", "无标题")
            url = article.get("url", "")
            source = article.get("source", "未知来源")
            publish_time = article.get("publish_time", "")
            sentiment_label = article.get("sentiment_label", "")
            sentiment_score = article.get("sentiment_score", 0) or 0

            _theme = get_theme(st.session_state.get("theme_mode", "light"))
            if sentiment_label == "positive" or sentiment_score > 0.3:
                badge_color = _theme["badge_positive"]
                badge_text = "正面"
            elif sentiment_label == "negative" or sentiment_score < -0.3:
                badge_color = _theme["badge_negative"]
                badge_text = "负面"
            else:
                badge_color = _theme["badge_neutral"]
                badge_text = "中性"

            # 格式化时间
            time_display = ""
            if publish_time:
                try:
                    dt = datetime.fromisoformat(publish_time)
                    time_display = dt.strftime("%m-%d %H:%M")
                except (ValueError, TypeError):
                    time_display = str(publish_time)[:16]

            # 新闻卡片
            title_html = f'<a href="{url}" target="_blank" style="text-decoration:none; color:{_theme["text_primary"]}; font-weight:600; font-size:16px;">{title}</a>' if url else f'<span style="font-weight:600; font-size:16px;">{title}</span>'

            symbol_part = f' &nbsp;·&nbsp; {article.get("symbol", "")}' if selected_label == '全部' else ''
            card_html = (
                f'<div style="padding:12px 16px; border-radius:12px; background-color:{_theme["card_bg"]}; margin-bottom:8px; border-left:4px solid {badge_color};">'
                f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                f'<div style="flex:1;">'
                f'{title_html}'
                f'<div style="margin-top:6px; font-size:13px; color:{_theme["text_secondary"]};">'
                f'<span style="background:{badge_color}; color:{_theme["badge_text_color"]}; padding:2px 8px; border-radius:10px; font-size:12px;">{badge_text}</span>'
                f'&nbsp; {source} &nbsp;·&nbsp; {time_display}{symbol_part}'
                f'</div></div></div></div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.info(f"{'该股票' if selected_symbol else '自选股'}最近 {days} 天暂无新闻。请确认后端新闻拉取任务已运行。")
        st.markdown("""
        **提示**：
        - 新闻由后台每30分钟自动拉取
        - 首次使用需等待一个拉取周期，或前往「系统设置」手动触发
        """)

# ====== TAB 2: 异动预警 ======
with tab_alert:
    if alerts_data and len(alerts_data) > 0:
        _theme_alert = get_theme(st.session_state.get("theme_mode", "light"))
        for alert in alerts_data:
            color = _theme_alert["text_secondary"]
            indicator = "·"
            alert_type = alert.get("alert_type", "").lower()
            val = alert.get("trigger_value", 0)

            if "sentiment" in alert_type or "新闻" in alert.get("message", ""):
                if val > 0.5:
                    color, indicator = _theme_alert["rise_color"], "▲"
                elif val < -0.5:
                    color, indicator = _theme_alert["fall_color"], "▼"
            elif "price" in alert_type:
                color = _theme_alert["rise_color"] if val > 0 else _theme_alert["fall_color"]
                indicator = "▲" if val > 0 else "▼"
            elif "volume" in alert_type:
                color, indicator = _theme_alert["text_secondary"], "◆"

            with st.container():
                type_map = {
                    "price_change": "价格异动",
                    "volume_spike": "成交量异动",
                    "sentiment_alert": "舆情预警",
                }
                display_type = type_map.get(alert.get("alert_type", ""), alert.get("alert_type", "系统预警"))

                alert_symbol = alert.get('symbol', '通用')
                alert_msg = alert.get('message', '无详细信息')
                alert_time = alert.get('created_at', '')
                alert_val = alert.get('trigger_value', '--')
                alert_html = (
                    f'<div style="padding:15px; border-radius:12px; border-left: 5px solid {color}; background-color: {_theme_alert["card_bg"]}; margin-bottom: 10px;">'
                    f'<h4 style="margin:0;">{indicator} [{alert_symbol}] {display_type}</h4>'
                    f'<p style="margin:5px 0;">{alert_msg}</p>'
                    f'<small style="color:{_theme_alert["text_secondary"]};">时间: {alert_time} | 触发值: {alert_val}</small>'
                    f'</div>'
                )
                st.markdown(alert_html, unsafe_allow_html=True)

                if st.button("标记已读", key=f"read_{alert.get('id', 'x')}"):
                    res = safe_api_call("PUT", f"/api/alerts/{alert.get('id')}/read")
                    if res:
                        st.rerun()
    else:
        st.info(f"太棒了！{selected_label if selected_label != '全部' else '当前'}没有未读预警或负面新闻。")
        st.markdown("*最近暂无触碰阈值的异动或重要舆情*")

# ====== TAB 3: 舆情趋势 ======
with tab_trend:

    # 如果有真实新闻数据，用真实数据聚合；否则用占位数据
    if news_data and len(news_data) > 5:
        df_news = pd.DataFrame(news_data)
        df_news["date"] = pd.to_datetime(df_news["publish_time"]).dt.date
        df_news["sentiment_score"] = pd.to_numeric(df_news["sentiment_score"], errors="coerce").fillna(0)
        daily = df_news.groupby("date")["sentiment_score"].mean().reset_index()
        daily = daily.sort_values("date")
        dates = daily["date"].astype(str).tolist()
        sentiment_scores = (daily["sentiment_score"] * 100).tolist()
        chart_note = "基于实际新闻情感分析得分"
    else:
        import random
        dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30, 0, -1)]
        random.seed(hash(selected_symbol) if selected_symbol else 42)
        sentiment_scores = [random.uniform(-50, 50) + (i * 1.5 if i < 15 else (30 - i) * 2) for i in range(30)]
        chart_note = "暂无足够新闻数据，显示模拟趋势"

    _theme_trend = get_theme(st.session_state.get("theme_mode", "light"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=sentiment_scores,
        mode='lines',
        name='情感指数',
        line=dict(color=_theme_trend["sentiment_line_color"], width=2),
        fill='tozeroy',
        fillcolor=_theme_trend["sentiment_fill_color"]
    ))
    fig.add_hline(y=50, line_dash="dash", line_color=_theme_trend["rise_color"], annotation_text="高度乐观")
    fig.add_hline(y=-50, line_dash="dash", line_color=_theme_trend["fall_color"], annotation_text="极度悲观")
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="日期",
        yaxis_title="情绪得分",
        showlegend=False,
        paper_bgcolor=_theme_trend["plotly_paper_bg"],
        plot_bgcolor=_theme_trend["plotly_plot_bg"],
        font=dict(color=_theme_trend["text_primary"])
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(chart_note)

    st.markdown("""
    **指标说明：**
    * > 50: 市场情绪极度乐观，需警惕回调风险
    * < -50: 市场情绪极度悲观，可能存在超跌反弹机会
    * 0 附近: 情绪中性，受消息面影响较小
    """)

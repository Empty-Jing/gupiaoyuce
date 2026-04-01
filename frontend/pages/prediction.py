import sys
import os
import time
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.widgets import safe_api_call
from theme import get_theme

# --- 股票选择器 ---
watchlist = safe_api_call("GET", "/api/stocks/")
if not watchlist:
    st.warning("自选股为空。请先添加股票。")
    st.stop()

stock_options = {f"{s['name']} ({s['symbol']})": s['symbol'] for s in watchlist}
selected_label = st.selectbox("选择要预测的股票", list(stock_options.keys()))
selected_symbol = stock_options[selected_label]

# --- 获取预测数据 ---
with st.spinner(f"正在分析 {selected_label} 的多维数据..."):
    prediction = safe_api_call("GET", f"/api/predictions/{selected_symbol}")

if prediction:
    _theme = get_theme(st.session_state.get("theme_mode", "light"))

    dir_colors = {
        "看涨": _theme["rise_color"],
        "看跌": _theme["fall_color"],
        "震荡": _theme["text_secondary"],
    }
    
    trend_emojis = {
        "强烈看涨": "▲▲",
        "看涨": "▲",
        "中性": "—",
        "看跌": "▼",
        "强烈看跌": "▼▼"
    }
    
    direction = prediction.get('direction', '未知')
    probability = prediction.get('probability', 0)
    trend = prediction.get('trend_rating', '未知')
    
    color = dir_colors.get(direction, _theme["text_secondary"])
    emoji = trend_emojis.get(trend, "—")
    
    # 顶部卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"### 预测方向\n<h2 style='color: {color}; margin-top:0;'>{direction}</h2>", unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"### 趋势评级\n<h2 style='margin-top:0;'>{emoji} {trend}</h2>", unsafe_allow_html=True)
        
    with col3:
        price_range = prediction.get('price_range', [0, 0])
        p_low = min(price_range[0], price_range[1]) * 100
        p_high = max(price_range[0], price_range[1]) * 100
        st.metric("预测上界", f"{p_high:+.2f}%", delta=f"{p_high:+.2f}%", delta_color="inverse")
        st.metric("预测下界", f"{p_low:+.2f}%", delta=f"{p_low:+.2f}%", delta_color="inverse")
        
    with col4:
        ret = prediction.get('predicted_return', 0)
        ret_color = _theme["rise_color"] if ret > 0 else _theme["fall_color"] if ret < 0 else _theme["text_secondary"]
        st.markdown(f"### 预测收益率\n<h2 style='color: {ret_color}; margin-top:0;'>{ret:+.2f}%</h2>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 下方图表
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # 置信度仪表盘
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "模型预测置信度", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': _theme["gauge_tick_color"]},
                'bar': {'color': color},
                'bgcolor': _theme["gauge_bg"],
                'borderwidth': 2,
                'bordercolor': _theme["gauge_border"],
                'steps': [
                    {'range': [0, 30], 'color': _theme["gauge_steps"][0]},
                    {'range': [30, 70], 'color': _theme["gauge_steps"][1]},
                    {'range': [70, 100], 'color': _theme["gauge_steps"][2]},
                ],
                'threshold': {
                    'line': {'color': _theme["gauge_threshold_color"], 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(
            height=350, margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor=_theme["plotly_paper_bg"],
            plot_bgcolor=_theme["plotly_plot_bg"],
            font=dict(color=_theme["plotly_font_color"], family=_theme["font_family"]),
        )
        st.plotly_chart(fig_gauge, width="stretch")
        
    with col_chart2:
        # 模型权重饼图
        weights = prediction.get('model_weights', {"lstm": 0.5, "xgboost": 0.5})
        labels = list(weights.keys())
        values = list(weights.values())
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=[l.upper() for l in labels],
            values=values,
            hole=.4,
            marker_colors=_theme["pie_colors"][:len(labels)]
        )])
        fig_pie.update_layout(
            title_text="预测模型权重融合比例",
            height=350,
            annotations=[dict(text='融合<br>策略', x=0.5, y=0.5, font_size=20, showarrow=False)],
            paper_bgcolor=_theme["plotly_paper_bg"],
            plot_bgcolor=_theme["plotly_plot_bg"],
            font=dict(color=_theme["plotly_font_color"], family=_theme["font_family"]),
        )
        st.plotly_chart(fig_pie, width="stretch")

    with st.expander("各模型预测详情"):
        col_m1, col_m2 = st.columns(2)

        lstm_res = prediction.get('lstm_result', {})
        with col_m1:
            st.markdown("**LSTM 模型**")
            if lstm_res:
                dp = lstm_res.get('direction_probs', {})
                _dir_cn = {"UP": "看涨", "DOWN": "看跌", "FLAT": "震荡"}
                if dp:
                    probs_str = " / ".join(f"{_dir_cn.get(k, k)} {v * 100:.1f}%" for k, v in dp.items())
                    st.markdown(f"方向概率：{probs_str}")
                pr = lstm_res.get('price_range')
                if pr and len(pr) == 2:
                    st.markdown(f"预测区间：{pr[0] * 100:+.2f}% ~ {pr[1] * 100:+.2f}%")
                ret = lstm_res.get('predicted_return')
                if ret is not None:
                    st.markdown(f"预测收益率：{ret * 100:+.2f}%")
            else:
                st.caption("暂无 LSTM 预测数据")

        xgb_res = prediction.get('xgboost_result', {})
        with col_m2:
            st.markdown("**XGBoost 模型**")
            if xgb_res:
                p_up = xgb_res.get('p_up')
                p_down = xgb_res.get('p_down')
                if p_up is not None:
                    st.markdown(f"看涨概率：{p_up * 100:.1f}%")
                if p_down is not None:
                    st.markdown(f"看跌概率：{p_down * 100:.1f}%")
            else:
                st.caption("暂无 XGBoost 预测数据")

else:
    running_jobs = safe_api_call("GET", f"/api/predictions/retrain/jobs/list?symbol={selected_symbol}&limit=5")
    active_job = None
    if running_jobs and isinstance(running_jobs, list):
        for j in running_jobs:
            if j.get("status") in ("pending", "running"):
                active_job = j
                break

    if active_job:
        model_name = "LSTM" if active_job.get("model_type") == "lstm" else "XGBoost"
        progress = active_job.get("progress", 0)
        message = active_job.get("message", "训练中")
        st.info(f"{model_name} 模型正在训练中...")
        st.progress(min(progress, 100), text=message)
    else:
        st.info("该股票暂无最新的预测数据，请尝试在“系统设置”中触发重新训练。")

# --- 历史预测记录 ---

st.markdown("<br>", unsafe_allow_html=True)
st.caption("以下为模型历史预测记录，仅供参考，不构成投资建议。")

history = safe_api_call("GET", f"/api/predictions/{selected_symbol}/history")

if history and isinstance(history, list) and len(history) > 0:
    df_hist = pd.DataFrame(history)

    # 删除不需要展示的列
    df_hist = df_hist.drop(columns=[c for c in ['id', 'symbol'] if c in df_hist.columns])

    # 方向字段翻译
    _dir_map = {"UP": "看涨", "DOWN": "看跌", "FLAT": "震荡"}
    if 'direction' in df_hist.columns:
        df_hist['direction'] = df_hist['direction'].map(lambda v: _dir_map.get(v, v))

    # 数值格式化
    if 'probability' in df_hist.columns:
        df_hist['probability'] = df_hist['probability'].map(lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "--")
    if 'predicted_return' in df_hist.columns:
        df_hist['predicted_return'] = df_hist['predicted_return'].map(lambda v: f"{v:+.2f}%" if pd.notna(v) else "--")
    if 'price_low' in df_hist.columns:
        df_hist['price_low'] = df_hist['price_low'].map(lambda v: f"{v * 100:+.2f}%" if pd.notna(v) else "--")
    if 'price_high' in df_hist.columns:
        df_hist['price_high'] = df_hist['price_high'].map(lambda v: f"{v * 100:+.2f}%" if pd.notna(v) else "--")
    if 'model_type' in df_hist.columns:
        _model_map = {"lstm": "LSTM", "xgboost": "XGBoost", "ensemble": "融合模型"}
        df_hist['model_type'] = df_hist['model_type'].map(lambda v: _model_map.get(v, v) if pd.notna(v) else "--")

    # 列名中文化
    _col_map = {
        'date': '预测日期',
        'model_type': '模型',
        'direction': '预测方向',
        'probability': '置信度',
        'predicted_return': '预测收益率',
        'price_low': '预测下界',
        'price_high': '预测上界',
    }
    df_hist = df_hist.rename(columns={k: v for k, v in _col_map.items() if k in df_hist.columns})

    # 按日期倒序
    if '预测日期' in df_hist.columns:
        df_hist = df_hist.sort_values('预测日期', ascending=False)

    # 涨跌颜色
    def highlight_direction(val):
        _th = get_theme(st.session_state.get("theme_mode", "light"))
        color = _th["rise_color"] if val == '看涨' else _th["fall_color"] if val == '看跌' else _th["text_secondary"]
        return f'color: {color}'

    df_styled = df_hist.style.map(highlight_direction, subset=['预测方向'] if '预测方向' in df_hist.columns else [])

    st.dataframe(df_styled, use_container_width=True, hide_index=True)
else:
    st.info("暂无历史预测记录。")

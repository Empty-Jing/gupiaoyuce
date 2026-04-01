import sys
import os
import time
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.widgets import safe_api_call

# --- 1. 监测阈值配置 ---
st.subheader("监测预警阈值")
with st.form("alert_config_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        price_change = st.slider("股价异动阈值 (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.5,
                               help="当股价日内涨跌幅超过此值时触发预警。")
                               
    with col2:
        volume_spike = st.slider("放量异动阈值 (倍数)", min_value=1.5, max_value=10.0, value=2.0, step=0.1,
                               help="当实时成交量超过平均成交量的此倍数时触发预警。")
                               
    with col3:
        sentiment = st.slider("负面舆情阈值", min_value=-1.0, max_value=0.0, value=-0.5, step=0.1,
                               help="当舆情分数低于此值时触发预警。")
                               
    submitted = st.form_submit_button("保存阈值设置")
    if submitted:
        payload = {
            "price_change_threshold": price_change,
            "volume_spike_threshold": volume_spike,
            "sentiment_threshold": sentiment
        }
        res = safe_api_call("PUT", "/api/alerts/config", json=payload)
        if res:
            st.success("阈值设置已保存")

# --- 2. AI模型重训练 ---
st.subheader("AI 预测模型管理")

watchlist = safe_api_call("GET", "/api/stocks/")
if watchlist:
    stock_options = {f"{s['name']} ({s['symbol']})": s['symbol'] for s in watchlist}
    selected_label = st.selectbox("选择需要重训练的股票", list(stock_options.keys()))
    selected_symbol = stock_options[selected_label]
    
    col_btn1, col_btn2 = st.columns(2)
    
    def _poll_training_progress(job_id: str, model_name: str):
        progress_bar = st.progress(0, text=f"{model_name} 训练已提交，等待执行...")
        status_text = st.empty()

        while True:
            status = safe_api_call("GET", f"/api/predictions/retrain/{job_id}/status")
            if status is None:
                status_text.error("无法获取训练状态")
                break

            progress = status.get("progress", 0)
            message = status.get("message", "")
            job_status = status.get("status", "unknown")

            progress_bar.progress(min(progress, 100), text=message)

            if job_status == "success":
                progress_bar.progress(100, text="训练完成")
                result = status.get("result", {})
                if result:
                    cols = st.columns(len(result))
                    labels = {
                        "train_loss": "训练损失",
                        "val_loss": "验证损失",
                        "val_accuracy": "验证准确率",
                        "train_accuracy": "训练准确率",
                    }
                    for i, (k, v) in enumerate(result.items()):
                        with cols[i]:
                            label = labels.get(k) or k
                            if "accuracy" in k:
                                st.metric(label, f"{v * 100:.1f}%")
                            else:
                                st.metric(label, f"{v:.4f}")
                st.success(f"{model_name} 训练完成")
                break

            if job_status == "failed":
                progress_bar.progress(progress, text="训练失败")
                st.error(f"训练失败: {status.get('error', '未知错误')}")
                break

            time.sleep(1.5)

    with col_btn1:
        if st.button("重新训练 LSTM 模型", key="btn_lstm", use_container_width=True):
            res = safe_api_call("POST", "/api/predictions/retrain", json={
                "symbol": selected_symbol,
                "model_type": "lstm"
            })
            if res and res.get("job_id"):
                _poll_training_progress(res["job_id"], "LSTM")
                    
    with col_btn2:
        if st.button("重新训练 XGBoost 模型", key="btn_xgb", use_container_width=True):
            res = safe_api_call("POST", "/api/predictions/retrain", json={
                "symbol": selected_symbol,
                "model_type": "xgboost"
            })
            if res and res.get("job_id"):
                _poll_training_progress(res["job_id"], "XGBoost")
else:
    st.warning("请先在仪表盘页面添加自选股。")

import os
from typing import Literal

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

def safe_api_call(method, endpoint, **kwargs):
    """安全地调用API并处理异常"""
    try:
        url = f"{API_URL}{endpoint}"
        if method.upper() == "GET":
            response = requests.get(url, **kwargs)
        elif method.upper() == "POST":
            response = requests.post(url, **kwargs)
        elif method.upper() == "PUT":
            response = requests.put(url, **kwargs)
        elif method.upper() == "DELETE":
            response = requests.delete(url, **kwargs)
        else:
            st.error(f"不支持的请求方法: {method}")
            return None
            
        response.raise_for_status()
        
        # 处理204 No Content
        if response.status_code == 204:
            return True
            
        return response.json()
        
    except requests.exceptions.ConnectionError:
        st.error(f"无法连接后端服务，请确认 FastAPI 已启动 ({API_URL})")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API 请求错误: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"发生未知错误: {str(e)}")
        return None

def stock_search_widget(key="search"):
    st.markdown("### 搜索股票")

    state_key = f"_search_results_{key}"

    col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
    with col1:
        keyword = st.text_input("输入股票代码或名称 (如: 000001或平安银行)", key=f"input_{key}")
    with col2:
        search_btn = st.button("搜索", key=f"btn_{key}", use_container_width=True)

    if search_btn and keyword:
        with st.spinner("搜索中..."):
            results = safe_api_call("GET", f"/api/stocks/search", params={"keyword": keyword})
            if results:
                st.session_state[state_key] = results
                st.success(f"找到 {len(results)} 条结果")
            elif results is not None:
                st.session_state[state_key] = []
                st.info("未找到匹配的股票")
            else:
                st.session_state[state_key] = []

    return st.session_state.get(state_key)

def render_metric_card(
    title,
    value,
    delta=None,
    delta_color: Literal["normal", "inverse", "off", "red", "orange", "yellow", "green"] = "normal",
):
    """渲染指标卡片"""
    st.metric(label=title, value=value, delta=delta, delta_color=delta_color)

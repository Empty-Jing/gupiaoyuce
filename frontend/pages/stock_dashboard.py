import sys
import os
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.widgets import safe_api_call, stock_search_widget
from theme import get_theme

# --- 顶部：搜索和操作区 ---
results = stock_search_widget(key="dashboard")
if results:
    st.write("点击添加到自选股：")
    for res in results[:5]:
        cols = st.columns([1, 2, 1])
        symbol = res.get('code', res.get('symbol', ''))
        name = res.get('name', '')
        cols[0].write(f"`{symbol}`")
        cols[1].write(name)

        if cols[2].button("添加", key=f"add_{symbol}"):
            add_res = safe_api_call("POST", "/api/stocks/watchlist", json={"symbol": symbol, "name": name, "market": "A"})
            if add_res:
                st.success(f"已成功添加 {name} ({symbol})")
                st.session_state.pop("_search_results_dashboard", None)
                st.rerun()

if st.button("刷新数据", use_container_width=True):
    st.rerun()

# --- 主体：自选股列表 ---

watchlist = safe_api_call("GET", "/api/stocks/")

if watchlist is None:
    st.warning("暂无数据，请确认后端服务已启动。")
elif not watchlist:
    st.info("自选股池为空，请先搜索并添加股票。")
else:
    all_symbols = ",".join(s['symbol'] for s in watchlist)
    symbol_to_name = {s['symbol']: s for s in watchlist}

    with st.spinner("正在获取实时行情..."):
        batch_realtime = safe_api_call("GET", "/api/stocks/batch/realtime", params={"symbols": all_symbols})

    realtime_map = {}
    if batch_realtime and isinstance(batch_realtime, list):
        for item in batch_realtime:
            sym = item.get('symbol', '')
            realtime_map[sym] = item

    if "edit_mode" not in st.session_state:
        st.session_state["edit_mode"] = False

    col_title, col_edit = st.columns([5, 1])
    with col_title:
        st.subheader("我的自选池")
    with col_edit:
        if st.session_state["edit_mode"]:
            if st.button("完成", key="finish_edit", use_container_width=True):
                st.session_state["edit_mode"] = False
                st.rerun()
        else:
            if st.button("编辑", key="start_edit", use_container_width=True):
                st.session_state["edit_mode"] = True
                st.rerun()

    editing = st.session_state["edit_mode"]

    for stock in watchlist:
        symbol = stock['symbol']
        name = stock['name']
        realtime = realtime_map.get(symbol)

        if realtime and isinstance(realtime, dict):
            latest_price = realtime.get('最新价', realtime.get('price', realtime.get('current_price', '--')))
            pct_change = realtime.get('涨跌幅', realtime.get('change_pct', realtime.get('change_percent', '--')))
            volume = realtime.get('成交量', realtime.get('volume', realtime.get('vol', '--')))

            if isinstance(latest_price, (int, float)):
                latest_price = f"{latest_price:.2f}"
            if isinstance(pct_change, (int, float)):
                pct_change = f"{pct_change:.2f}%"
            elif isinstance(pct_change, str) and not pct_change.endswith('%') and pct_change != '--':
                try:
                    pct_change = f"{float(pct_change):.2f}%"
                except Exception:
                    pass
        else:
            latest_price = "--"
            pct_change = "--"
            volume = "--"

        _theme = get_theme(st.session_state.get("theme_mode", "light"))
        try:
            pct_val = float(str(pct_change).replace('%', ''))
            pct_color = _theme["rise_color"] if pct_val > 0 else _theme["fall_color"] if pct_val < 0 else _theme["text_secondary"]
        except Exception:
            pct_color = _theme["text_secondary"]

        row_cols = st.columns([3, 1.5, 1.5, 0.5])

        with row_cols[0]:
            st.markdown(f"**{name}** `{symbol}`")
        with row_cols[1]:
            st.markdown(f"{latest_price}")
        with row_cols[2]:
            st.markdown(f'<span style="color:{pct_color}">{pct_change}</span>', unsafe_allow_html=True)

        if editing:
            with row_cols[3]:
                if st.button("✕", key=f"del_{symbol}", help=f"删除 {name}"):
                    del_res = safe_api_call("DELETE", f"/api/stocks/watchlist/{symbol}")
                    if del_res is True:
                        st.rerun()

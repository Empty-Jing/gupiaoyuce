import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "frontend"))

import pandas as pd
import plotly.graph_objects as go

from theme import LIGHT_THEME, DARK_THEME
from components.charts import render_kline, render_volume, render_signal_markers, render_kline_html


def _make_df():
    return pd.DataFrame({
        "日期": pd.date_range("2024-01-02", periods=5, freq="B").strftime("%Y-%m-%d").tolist(),
        "开盘": [10.0, 10.5, 10.3, 10.8, 10.6],
        "最高": [10.8, 10.9, 10.7, 11.0, 10.9],
        "最低": [9.8, 10.2, 10.0, 10.5, 10.3],
        "收盘": [10.5, 10.3, 10.6, 10.7, 10.8],
        "成交量": [1000, 1200, 900, 1500, 1100],
    })


def test_render_kline_light_returns_figure():
    fig = render_kline(_make_df(), theme=LIGHT_THEME)
    assert isinstance(fig, go.Figure)


def test_render_kline_dark_returns_figure():
    fig = render_kline(_make_df(), theme=DARK_THEME)
    assert isinstance(fig, go.Figure)


def test_render_kline_light_paper_bgcolor():
    fig = render_kline(_make_df(), theme=LIGHT_THEME)
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"


def test_render_kline_html_light_has_panel_bg():
    html, _ = render_kline_html(_make_df(), theme=LIGHT_THEME)
    assert LIGHT_THEME["ohlc_panel_bg"] in html
    assert "rgba(30,30,30,0.92)" not in html


def test_render_kline_html_dark_has_panel_bg():
    html, _ = render_kline_html(_make_df(), theme=DARK_THEME)
    assert DARK_THEME["ohlc_panel_bg"] in html


def test_render_kline_backward_compat():
    fig = render_kline(_make_df())
    assert isinstance(fig, go.Figure)


def test_render_volume_themed():
    fig = render_volume(_make_df(), theme=DARK_THEME)
    assert isinstance(fig, go.Figure)
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"


def test_render_kline_html_has_theme_var():
    html, _ = render_kline_html(_make_df(), theme=LIGHT_THEME)
    assert "var THEME" in html
    assert LIGHT_THEME["rise_color"] in html

import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def _get_holidays(date_series):
    dates = pd.to_datetime(date_series)
    if dates.empty:
        return []
    all_days = pd.bdate_range(dates.min(), dates.max())
    trading_days = set(dates.dt.normalize())
    return [d.strftime("%Y-%m-%d") for d in all_days if d not in trading_days]


def _resolve_theme(theme):
    if theme is not None:
        return theme
    from theme import LIGHT_THEME
    return LIGHT_THEME


def render_kline(df, indicators=None, show_volume=True, theme=None, title=None):
    if df is None or df.empty:
        return go.Figure()

    theme = _resolve_theme(theme)

    if indicators is None:
        indicators = []

    has_macd = "MACD" in indicators
    has_rsi = "RSI" in indicators
    has_kdj = "KDJ" in indicators

    sub_plots = []
    row_heights = [0.5]

    if show_volume:
        sub_plots.append("成交量")
        row_heights.append(0.15)
    if has_macd:
        sub_plots.append("MACD")
        row_heights.append(0.15)
    if has_rsi:
        sub_plots.append("RSI")
        row_heights.append(0.15)
    if has_kdj:
        sub_plots.append("KDJ")
        row_heights.append(0.15)

    total_h = sum(row_heights)
    row_heights = [h / total_h for h in row_heights]

    rows = len(row_heights)

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
    )

    closes = df['收盘'].tolist()
    hover_texts = []
    for i, (d, o, h, l, c) in enumerate(zip(df['日期'], df['开盘'], df['最高'], df['最低'], df['收盘'])):
        if i == 0:
            chg = (c - o) / o * 100 if o != 0 else 0
        else:
            prev_c = closes[i - 1]
            chg = (c - prev_c) / prev_c * 100 if prev_c != 0 else 0
        chg_color = theme["rise_color"] if chg >= 0 else theme["fall_color"]
        chg_sign = "+" if chg >= 0 else ""
        hover_texts.append(
            f"日期: {d}<br>开盘: {o:.2f}<br>最高: {h:.2f}<br>最低: {l:.2f}<br>收盘: {c:.2f}"
            f'<br>涨幅: <span style="color:{chg_color}">{chg_sign}{chg:.2f}%</span>'
        )

    fig.add_trace(go.Candlestick(
        x=df['日期'],
        open=df['开盘'],
        high=df['最高'],
        low=df['最低'],
        close=df['收盘'],
        name="K线",
        increasing_line_color=theme["rise_color"],
        decreasing_line_color=theme["fall_color"],
        text=hover_texts,
        hoverinfo="text",
    ), row=1, col=1)

    if "MA" in indicators:
        ma_cols = [c for c in df.columns if c.startswith('ma_') and c[3:].isdigit()]
        ma_colors = theme["ma_colors"]
        for i, ma in enumerate(ma_cols):
            fig.add_trace(go.Scatter(
                x=df['日期'], y=df[ma],
                mode='lines',
                name=ma.upper(),
                line=dict(width=1, color=ma_colors[i % len(ma_colors)])
            ), row=1, col=1)

    if "BOLL" in indicators:
        if 'boll_upper' in df.columns and 'boll_mid' in df.columns and 'boll_lower' in df.columns:
            boll = theme["boll_colors"]
            fig.add_trace(go.Scatter(x=df['日期'], y=df['boll_upper'], mode='lines', name='BOLL 上轨', line=dict(width=1, color=boll["upper"], dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['日期'], y=df['boll_mid'], mode='lines', name='BOLL 中轨', line=dict(width=1, color=boll["mid"], dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['日期'], y=df['boll_lower'], mode='lines', name='BOLL 下轨', line=dict(width=1, color=boll["lower"], dash='dash')), row=1, col=1)

    curr_row = 2

    if show_volume and '成交量' in df.columns:
        vol_colors = []
        for i in range(len(df)):
            if i == 0:
                vol_colors.append(theme["rise_color"] if df['收盘'].iloc[i] >= df['开盘'].iloc[i] else theme["fall_color"])
            else:
                vol_colors.append(theme["rise_color"] if df['收盘'].iloc[i] >= df['收盘'].iloc[i-1] else theme["fall_color"])
        fig.add_trace(go.Bar(
            x=df['日期'], y=df['成交量'],
            marker_color=vol_colors, name='成交量',
            showlegend=False
        ), row=curr_row, col=1)
        curr_row += 1

    if has_macd:
        if 'macd' in df.columns and 'macd_signal' in df.columns and 'macd_hist' in df.columns:
            macd = theme["macd_colors"]
            fig.add_trace(go.Scatter(x=df['日期'], y=df['macd'], mode='lines', name='MACD', line=dict(color=macd["line"])), row=curr_row, col=1)
            fig.add_trace(go.Scatter(x=df['日期'], y=df['macd_signal'], mode='lines', name='Signal', line=dict(color=macd["signal"])), row=curr_row, col=1)

            hist_colors = [macd["hist_pos"] if val >= 0 else macd["hist_neg"] for val in df['macd_hist']]
            fig.add_trace(go.Bar(x=df['日期'], y=df['macd_hist'], name='Hist', marker_color=hist_colors), row=curr_row, col=1)
        curr_row += 1

    if has_rsi:
        rsi_cols = [c for c in df.columns if c.startswith('rsi_')]
        for rsi in rsi_cols:
            fig.add_trace(go.Scatter(x=df['日期'], y=df[rsi], mode='lines', name=rsi.upper()), row=curr_row, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color=theme["rsi_hline_color"], row=curr_row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color=theme["rsi_hline_color"], row=curr_row, col=1)
        curr_row += 1

    if has_kdj:
        if 'kdj_k' in df.columns and 'kdj_d' in df.columns and 'kdj_j' in df.columns:
            kdj = theme["kdj_colors"]
            fig.add_trace(go.Scatter(x=df['日期'], y=df['kdj_k'], mode='lines', name='K', line=dict(color=kdj["k"])), row=curr_row, col=1)
            fig.add_trace(go.Scatter(x=df['日期'], y=df['kdj_d'], mode='lines', name='D', line=dict(color=kdj["d"])), row=curr_row, col=1)
            fig.add_trace(go.Scatter(x=df['日期'], y=df['kdj_j'], mode='lines', name='J', line=dict(color=kdj["j"])), row=curr_row, col=1)
        curr_row += 1

    for i in range(1, rows + 1):
        xaxis_key = f"xaxis{i}" if i > 1 else "xaxis"
        yaxis_key = f"yaxis{i}" if i > 1 else "yaxis"
        yaxis_opts = dict(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=0.5,
            spikecolor=theme["plotly_spike_color"],
            spikedash="solid",
            fixedrange=True,
            autorange=True,
        )
        # 子图标题：row 1 = K线（无标题），row 2+ = sub_plots 对应名称
        if i >= 2 and (i - 2) < len(sub_plots):
            yaxis_opts["title"] = dict(
                text=sub_plots[i - 2],
                font=dict(size=11, color=theme["text_secondary"]),
                standoff=8,
            )
        fig.update_layout(**{
            xaxis_key: dict(
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikethickness=0.5,
                spikecolor=theme["plotly_spike_color"],
                spikedash="solid",
            ),
            yaxis_key: yaxis_opts,
        })

    title_opts = {}
    if title:
        title_opts["title"] = dict(
            text=title,
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
            font=dict(size=16, color=theme["text_primary"], family=theme["font_family"]),
        )

    fig.update_layout(
        **title_opts,
        xaxis_rangeslider_visible=False,
        height=500 + 180 * len(sub_plots),
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        dragmode="pan",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme["plotly_font_color"], family=theme["font_family"]),
        modebar=dict(
            orientation="h",
            bgcolor="rgba(0,0,0,0)",
            color=theme["text_secondary"],
            activecolor=theme["accent_blue"],
        ),
    )

    holidays = _get_holidays(df['日期']) if '日期' in df.columns else []

    last_xaxis = f"xaxis{rows}" if rows > 1 else "xaxis"
    fig.update_layout(**{
        last_xaxis: dict(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(values=holidays),
            ],
        ),
    })

    return fig

def render_volume(df, theme=None):
    if df is None or df.empty or '成交量' not in df.columns:
        return go.Figure()

    theme = _resolve_theme(theme)

    colors = []
    for i in range(len(df)):
        if i == 0:
            colors.append(theme["rise_color"] if df['收盘'].iloc[i] >= df['开盘'].iloc[i] else theme["fall_color"])
        else:
            colors.append(theme["rise_color"] if df['收盘'].iloc[i] >= df['收盘'].iloc[i-1] else theme["fall_color"])

    fig = go.Figure(data=[go.Bar(
        x=df['日期'],
        y=df['成交量'],
        marker_color=colors,
        name='成交量'
    )])

    fig.update_layout(
        height=200,
        margin=dict(l=50, r=50, t=30, b=30),
        title="成交量",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme["plotly_font_color"], family=theme["font_family"]),
    )

    return fig

def render_signal_markers(fig, df, theme=None):
    if df is None or df.empty or 'signal' not in df.columns:
        return fig

    theme = _resolve_theme(theme)

    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]

    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals['日期'],
            y=buy_signals['最低'] * 0.98,
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color=theme["signal_buy_color"]),
            name='买入信号'
        ), row=1, col=1)

    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals['日期'],
            y=sell_signals['最高'] * 1.02,
            mode='markers',
            marker=dict(symbol='triangle-down', size=10, color=theme["signal_sell_color"]),
            name='卖出信号'
        ), row=1, col=1)

    return fig


def render_kline_html(df, indicators=None, show_volume=True, fig=None, theme=None):
    from theme import LIGHT_THEME, DARK_THEME

    theme = _resolve_theme(theme)

    if fig is None:
        fig = render_kline(df, indicators=indicators, show_volume=show_volume, theme=theme)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme["plotly_font_color"]),
    )

    fig.update_traces(hoverinfo="none", hovertemplate=None, selector=dict(type="candlestick"))

    closes = df['收盘'].tolist()
    ohlc_rows = []
    for i in range(len(df)):
        o, h, l, c = float(df['开盘'].iloc[i]), float(df['最高'].iloc[i]), float(df['最低'].iloc[i]), float(df['收盘'].iloc[i])
        d = str(df['日期'].iloc[i])[:10]
        prev = float(closes[i - 1]) if i > 0 else o
        chg = (c - prev) / prev * 100 if prev != 0 else 0
        vol = float(df['成交量'].iloc[i]) if '成交量' in df.columns else 0
        ohlc_rows.append({"d": d, "o": o, "h": h, "l": l, "c": c, "chg": round(chg, 2), "vol": vol})

    ohlc_json = json.dumps(ohlc_rows, ensure_ascii=False)
    fig_json = fig.to_json()

    chart_height = 500 + 180 * max(0, len([x for x in (show_volume, "MACD" in (indicators or []), "RSI" in (indicators or []), "KDJ" in (indicators or [])) if x]))

    rows = 1 + len([x for x in (show_volume, "MACD" in (indicators or []), "RSI" in (indicators or []), "KDJ" in (indicators or [])) if x])

    def _theme_js_obj(t):
        return (
            f"{{rise:'{t['rise_color']}',fall:'{t['fall_color']}',"
            f"font:'{t['plotly_font_color']}',grid:'{t['plotly_gridcolor']}',"
            f"spike:'{t['plotly_spike_color']}',accent:'{t['accent_blue']}',"
            f"textSec:'{t['text_secondary']}',textPri:'{t['text_primary']}',"
            f"panelBg:'{t['ohlc_panel_bg']}',panelText:'{t['ohlc_panel_text']}',"
            f"panelTitle:'{t['ohlc_panel_title']}',high:'{t['ohlc_panel_high']}',"
            f"low:'{t['ohlc_panel_low']}'}}"
        )

    light_js = _theme_js_obj(LIGHT_THEME)
    dark_js = _theme_js_obj(DARK_THEME)

    html = f"""
<style>
html, body {{ background: transparent !important; margin: 0; padding: 0; }}
</style>
<div id="kline-wrapper" style="position:relative; width:100%; background-color:transparent; border-radius:12px; min-height:{chart_height}px; opacity:0; transition:opacity 120ms ease-out;">
  <div id="ohlc-panel" style="
    display:none;
    position:absolute;
    z-index:1000;
    pointer-events:none;
    font-family:{theme["font_family"]};
    font-size:14px;
    line-height:1.7;
    background:{theme["ohlc_panel_bg"]};
    color:{theme["ohlc_panel_text"]};
    padding:10px 16px;
    border-radius:12px;
    box-shadow:0 4px 16px rgba(0,0,0,0.3);
    white-space:nowrap;
    min-width:180px;
  "></div>
  <div id="kline-chart" style="width:100%; height:{chart_height}px;"></div>
</div>
<script>
(function() {{
  function boot() {{
    if (typeof Plotly !== 'undefined') {{ initChart(); return; }}
    var s = document.createElement('script');
    s.src = 'https://cdn.plot.ly/plotly-2.35.0.min.js';
    s.async = true;
    s.onload = initChart;
    document.head.appendChild(s);
  }}

  function initChart() {{
  Plotly.register({{moduleType: 'locale', name: 'zh-CN', dictionary: {{
    'Zoom in': '放大', 'Zoom out': '缩小', 'Pan': '平移',
    'Reset axes': '重置视图', 'Toggle Spike Lines': '切换标记线',
    'Download plot as a png': '下载为图片',
    'Autoscale': '自动缩放', 'Produced with Plotly': ''
  }}, format: {{decimal: '.', thousands: ','}}}});

  var THEMES = {{light: {light_js}, dark: {dark_js}}};
  var THEME = THEMES.light;

  function detectDark() {{
    try {{
      var bg = window.parent.document.body ?
        getComputedStyle(window.parent.document.body).backgroundColor : '';
      if (!bg || bg === 'transparent' || bg === 'rgba(0, 0, 0, 0)') return null;
      var m = bg.match(/\\d+/g);
      if (m && m.length >= 3) {{
        var lum = (parseInt(m[0]) * 299 + parseInt(m[1]) * 587 + parseInt(m[2]) * 114) / 1000;
        return lum < 128;
      }}
    }} catch(e) {{}}
    return null;
  }}

  var figData = {fig_json};
  var ohlc = {ohlc_json};
  var chartDiv = document.getElementById('kline-chart');
  var panel = document.getElementById('ohlc-panel');
  var wrapper = document.getElementById('kline-wrapper');

  var initDark = detectDark();
  if (initDark === true) {{
    THEME = THEMES.dark;
    figData.layout.font = figData.layout.font || {{}};
    figData.layout.font.color = THEME.font;
    for (var ax = 1; ax <= {rows}; ax++) {{
      var xk = ax > 1 ? 'xaxis' + ax : 'xaxis';
      var yk = ax > 1 ? 'yaxis' + ax : 'yaxis';
      if (figData.layout[xk]) {{
        figData.layout[xk].gridcolor = THEME.grid;
        figData.layout[xk].spikecolor = THEME.spike;
      }}
      if (figData.layout[yk]) {{
        figData.layout[yk].gridcolor = THEME.grid;
        figData.layout[yk].spikecolor = THEME.spike;
        if (figData.layout[yk].title && figData.layout[yk].title.font) {{
          figData.layout[yk].title.font.color = THEME.textSec;
        }}
      }}
    }}
    if (figData.layout.modebar) figData.layout.modebar.activecolor = THEME.accent;
    if (figData.layout.title && figData.layout.title.font) figData.layout.title.font.color = THEME.textPri;
    for (var t = 0; t < figData.data.length; t++) {{
      var tr = figData.data[t];
      if (tr.type === 'candlestick') {{
        tr.increasing = tr.increasing || {{}};
        tr.increasing.line = {{color: THEME.rise}};
        tr.increasing.fillcolor = THEME.rise;
        tr.decreasing = tr.decreasing || {{}};
        tr.decreasing.line = {{color: THEME.fall}};
        tr.decreasing.fillcolor = THEME.fall;
      }}
    }}
  }}

  Plotly.newPlot(chartDiv, figData.data, figData.layout, {{
    scrollZoom: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d'],
    locale: 'zh-CN'
  }});
  wrapper.style.opacity = '1';

  var prevDark = initDark === true;

  function applyTheme(isDark) {{
    THEME = isDark ? THEMES.dark : THEMES.light;

    var layoutUpd = {{
      'font.color': THEME.font,
      'modebar.activecolor': THEME.accent,
      'title.font.color': THEME.textPri
    }};
    for (var ax = 1; ax <= {rows}; ax++) {{
      var xp = ax > 1 ? 'xaxis' + ax : 'xaxis';
      var yp = ax > 1 ? 'yaxis' + ax : 'yaxis';
      layoutUpd[xp + '.gridcolor'] = THEME.grid;
      layoutUpd[xp + '.spikecolor'] = THEME.spike;
      layoutUpd[yp + '.gridcolor'] = THEME.grid;
      layoutUpd[yp + '.spikecolor'] = THEME.spike;
      layoutUpd[yp + '.title.font.color'] = THEME.textSec;
    }}

    Plotly.relayout(chartDiv, layoutUpd);

    var indices = [];
    var data = chartDiv.data || [];
    for (var t = 0; t < data.length; t++) {{
      if (data[t].type === 'candlestick') indices.push(t);
    }}
    if (indices.length > 0) {{
      Plotly.restyle(chartDiv, {{
        'increasing.line.color': THEME.rise,
        'increasing.fillcolor': THEME.rise,
        'decreasing.line.color': THEME.fall,
        'decreasing.fillcolor': THEME.fall
      }}, indices);
    }}

    panel.style.background = THEME.panelBg;
    panel.style.color = THEME.panelText;
  }}

  setInterval(function() {{
    var d = detectDark();
    if (d === null) return;
    if (d !== prevDark) {{
      prevDark = d;
      applyTheme(d);
    }}
  }}, 200);

  function fmtVol(v) {{
    if (v >= 1e8) return (v / 1e8).toFixed(2) + '亿';
    if (v >= 1e4) return (v / 1e4).toFixed(1) + '万';
    return v.toFixed(0);
  }}

  function fmtPanel(r) {{
    var color = r.chg >= 0 ? THEME.rise : THEME.fall;
    var sign = r.chg >= 0 ? '+' : '';
    var s = '<div style="font-size:15px; font-weight:600; margin-bottom:4px; color:' + THEME.panelTitle + ';">' + r.d + '</div>';
    s += '<div>开盘 <b style="color:' + THEME.panelText + ';">' + r.o.toFixed(2) + '</b></div>';
    s += '<div>最高 <b style="color:' + THEME.high + ';">' + r.h.toFixed(2) + '</b></div>';
    s += '<div>最低 <b style="color:' + THEME.low + ';">' + r.l.toFixed(2) + '</b></div>';
    s += '<div>收盘 <b style="color:' + THEME.panelText + ';">' + r.c.toFixed(2) + '</b></div>';
    s += '<div>涨幅 <b style="color:' + color + ';">' + sign + r.chg.toFixed(2) + '%</b></div>';
    if (r.vol > 0) s += '<div>成交量 <b style="color:' + THEME.panelText + ';">' + fmtVol(r.vol) + '</b></div>';
    return s;
  }}

  var mouseX = 0, mouseY = 0;
  wrapper.addEventListener('mousemove', function(e) {{
    var rect = wrapper.getBoundingClientRect();
    mouseX = e.clientX - rect.left;
    mouseY = e.clientY - rect.top;
  }});

  function positionPanel() {{
    var ww = wrapper.offsetWidth;
    var wh = wrapper.offsetHeight;
    var pw = panel.offsetWidth || 200;
    var ph = panel.offsetHeight || 160;
    var margin = 20;
    var horizontalOffset = 600;
    var x, y;
    if (mouseX < ww / 2) {{
      x = mouseX + horizontalOffset;
    }} else {{
      x = mouseX - pw - horizontalOffset;
    }}
    if (x < margin) x = margin;
    if (x + pw > ww - margin) x = ww - pw - margin;
    y = mouseY - ph / 2;
    if (y < 5) y = 5;
    if (y + ph > wh - 5) y = wh - ph - 5;
    panel.style.left = x + 'px';
    panel.style.top = y + 'px';
  }}

  chartDiv.on('plotly_hover', function(data) {{
    if (!data || !data.points || data.points.length === 0) return;
    var pt = data.points[0];
    var idx = pt.pointIndex != null ? pt.pointIndex : (pt.pointNumber || 0);
    if (idx >= 0 && idx < ohlc.length) {{
      panel.innerHTML = fmtPanel(ohlc[idx]);
      panel.style.display = 'block';
      positionPanel();
    }}
  }});

  chartDiv.on('plotly_unhover', function() {{
    panel.style.display = 'none';
  }});

  var dates = ohlc.map(function(r) {{ return r.d; }});
  var highs = ohlc.map(function(r) {{ return r.h; }});
  var lows  = ohlc.map(function(r) {{ return r.l; }});

  chartDiv.on('plotly_relayout', function(ev) {{
    var xRange = null;
    if (ev['xaxis.range[0]'] && ev['xaxis.range[1]']) {{
      xRange = [ev['xaxis.range[0]'], ev['xaxis.range[1]']];
    }} else if (ev['xaxis.range']) {{
      xRange = ev['xaxis.range'];
    }}
    if (!xRange) return;

    var lo = String(xRange[0]).slice(0, 10);
    var hi = String(xRange[1]).slice(0, 10);
    var yMin = Infinity, yMax = -Infinity;
    for (var i = 0; i < dates.length; i++) {{
      if (dates[i] >= lo && dates[i] <= hi) {{
        if (highs[i] > yMax) yMax = highs[i];
        if (lows[i] < yMin) yMin = lows[i];
      }}
    }}
    if (yMin === Infinity) return;
    var pad = (yMax - yMin) * 0.05;
    Plotly.relayout(chartDiv, {{'yaxis.range': [yMin - pad, yMax + pad], 'yaxis.autorange': false}});
  }});

  wrapper.addEventListener('mousemove', function() {{
    if (panel.style.display === 'block') positionPanel();
  }});
  }} // end initChart

  boot();
}})();
</script>
"""
    return html, chart_height + 30

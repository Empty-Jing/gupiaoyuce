"""
技术指标计算引擎
使用 ta 库 (0.11.0) 计算各类技术指标
"""

import pandas as pd
import numpy as np
import ta


_CN_TO_EN = {"开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume"}


class IndicatorEngine:
    """技术指标计算引擎，负责计算各类技术分析指标并生成交易信号。"""

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标。

        参数:
            df: 包含 open, high, low, close, volume 列的 K 线数据

        返回:
            添加了所有技术指标列的 DataFrame 副本
        """
        result = df.copy()
        if "收盘" in result.columns:
            result = result.rename(columns=_CN_TO_EN)

        # ── MACD ──────────────────────────────────────────────────────────────
        macd_ind = ta.trend.MACD(
            close=result['close'],
            window_slow=26,
            window_fast=12,
            window_sign=9
        )
        result['macd'] = macd_ind.macd()
        result['macd_signal'] = macd_ind.macd_signal()
        result['macd_hist'] = macd_ind.macd_diff()

        # ── 简单移动平均线 (SMA/MA) ────────────────────────────────────────────
        for window in [5, 10, 20, 60, 120]:
            sma = ta.trend.SMAIndicator(close=result['close'], window=window)
            result[f'ma_{window}'] = sma.sma_indicator()

        # ── 指数移动平均线 (EMA) ───────────────────────────────────────────────
        for window in [5, 10, 20, 60, 120]:
            ema = ta.trend.EMAIndicator(close=result['close'], window=window)
            result[f'ema_{window}'] = ema.ema_indicator()

        # ── RSI ───────────────────────────────────────────────────────────────
        for window in [6, 12, 24]:
            rsi = ta.momentum.RSIIndicator(close=result['close'], window=window)
            result[f'rsi_{window}'] = rsi.rsi()

        # ── KDJ（随机指标）────────────────────────────────────────────────────
        stoch = ta.momentum.StochasticOscillator(
            high=result['high'],
            low=result['low'],
            close=result['close'],
            window=9,
            smooth_window=3
        )
        result['kdj_k'] = stoch.stoch()
        result['kdj_d'] = stoch.stoch_signal()
        result['kdj_j'] = 3 * result['kdj_k'] - 2 * result['kdj_d']

        # ── 布林带 ────────────────────────────────────────────────────────────
        bb = ta.volatility.BollingerBands(
            close=result['close'],
            window=20,
            window_dev=2
        )
        result['boll_upper'] = bb.bollinger_hband()
        result['boll_mid'] = bb.bollinger_mavg()
        result['boll_lower'] = bb.bollinger_lband()

        # ── VWAP（成交量加权平均价）──────────────────────────────────────────
        try:
            vwap_ind = ta.volume.VolumeWeightedAveragePrice(
                high=result['high'],
                low=result['low'],
                close=result['close'],
                volume=result['volume']
            )
            result['vwap'] = vwap_ind.volume_weighted_average_price()
        except Exception:
            # 若 ta 库版本不支持，手动计算累计 VWAP
            result['vwap'] = (
                (result['close'] * result['volume']).cumsum()
                / result['volume'].cumsum()
            )

        # ── OBV（能量潮）──────────────────────────────────────────────────────
        obv_ind = ta.volume.OnBalanceVolumeIndicator(
            close=result['close'],
            volume=result['volume']
        )
        result['obv'] = obv_ind.on_balance_volume()

        # ── ADX（平均趋向指标）────────────────────────────────────────────────
        adx_ind = ta.trend.ADXIndicator(
            high=result['high'],
            low=result['low'],
            close=result['close'],
            window=14
        )
        result['adx'] = adx_ind.adx()

        return result

    def generate_signal(self, df: pd.DataFrame) -> str:
        """
        根据技术指标生成交易信号。

        参数:
            df: 已包含技术指标列的 DataFrame（即 calculate_all 的输出）

        返回:
            'BUY' / 'SELL' / 'HOLD'
        """
        if len(df) < 2:
            return 'HOLD'

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # 提取关键指标值（NaN 安全处理）
        macd_now = last.get('macd', float('nan'))
        macd_sig_now = last.get('macd_signal', float('nan'))
        macd_prev = prev.get('macd', float('nan'))
        macd_sig_prev = prev.get('macd_signal', float('nan'))
        rsi6 = last.get('rsi_6', float('nan'))
        close_now = last.get('close', float('nan'))
        ma20 = last.get('ma_20', float('nan'))

        # 检查是否有足够的非 NaN 数据
        def is_valid(*vals):
            return all(not (isinstance(v, float) and np.isnan(v)) and v is not None for v in vals)

        # ── BUY 条件：MACD 金叉 AND RSI(6) < 70 AND close > MA20 ─────────────
        if is_valid(macd_now, macd_sig_now, macd_prev, macd_sig_prev, rsi6, close_now, ma20):
            macd_cross_up = (macd_now > macd_sig_now) and (macd_prev <= macd_sig_prev)
            if macd_cross_up and rsi6 < 70 and close_now > ma20:
                return 'BUY'

        # ── SELL 条件：MACD 死叉 OR RSI(6) > 80 ─────────────────────────────
        if is_valid(macd_now, macd_sig_now, macd_prev, macd_sig_prev):
            macd_cross_down = (macd_now < macd_sig_now) and (macd_prev >= macd_sig_prev)
            if macd_cross_down:
                return 'SELL'

        if is_valid(rsi6) and rsi6 > 80:
            return 'SELL'

        return 'HOLD'

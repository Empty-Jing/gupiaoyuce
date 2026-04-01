"""
测试 IndicatorEngine 模块
使用合成 OHLCV 数据验证技术指标计算和交易信号生成逻辑。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from app.core.indicator_engine import IndicatorEngine


# ── 测试辅助函数 ──────────────────────────────────────────────────────────────

def make_ohlcv_df(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """
    生成 n 行合成 OHLCV 数据，用于指标计算测试。
    使用固定随机种子保证可复现性。
    """
    rng = np.random.default_rng(seed)
    # 生成随机游走的收盘价（从 100 开始）
    returns = rng.normal(0, 0.01, n)
    close = 100.0 * np.cumprod(1 + returns)
    # 构造合理的 OHLCV
    noise = rng.uniform(0.001, 0.02, (n, 4))
    open_ = close * (1 + rng.uniform(-0.01, 0.01, n))
    high = np.maximum(close, open_) * (1 + noise[:, 0])
    low = np.minimum(close, open_) * (1 - noise[:, 1])
    volume = rng.integers(500_000, 5_000_000, n).astype(float)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ── 测试类 ────────────────────────────────────────────────────────────────────

class TestIndicatorEngine:
    """IndicatorEngine 单元测试集。"""

    def setup_method(self):
        self.engine = IndicatorEngine()
        self.df = make_ohlcv_df(n=250)

    # ── calculate_all：列存在性 ─────────────────────────────────────────────

    def test_calculate_all_returns_dataframe(self):
        """calculate_all 应返回 DataFrame 类型。"""
        result = self.engine.calculate_all(self.df)
        assert isinstance(result, pd.DataFrame)

    def test_calculate_all_preserves_row_count(self):
        """calculate_all 不应改变行数。"""
        result = self.engine.calculate_all(self.df)
        assert len(result) == len(self.df)

    def test_calculate_all_has_macd_columns(self):
        """calculate_all 结果应包含 macd/macd_signal/macd_hist 列。"""
        result = self.engine.calculate_all(self.df)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns

    def test_calculate_all_has_ma_columns(self):
        """calculate_all 结果应包含 ma_5/ma_10/ma_20/ma_60/ma_120。"""
        result = self.engine.calculate_all(self.df)
        for window in [5, 10, 20, 60, 120]:
            assert f"ma_{window}" in result.columns, f"缺少 ma_{window} 列"

    def test_calculate_all_has_ema_columns(self):
        """calculate_all 结果应包含 ema_5/ema_10/ema_20/ema_60/ema_120。"""
        result = self.engine.calculate_all(self.df)
        for window in [5, 10, 20, 60, 120]:
            assert f"ema_{window}" in result.columns, f"缺少 ema_{window} 列"

    def test_calculate_all_has_rsi_columns(self):
        """calculate_all 结果应包含 rsi_6/rsi_12/rsi_24。"""
        result = self.engine.calculate_all(self.df)
        for window in [6, 12, 24]:
            assert f"rsi_{window}" in result.columns, f"缺少 rsi_{window} 列"

    def test_calculate_all_has_kdj_columns(self):
        """calculate_all 结果应包含 kdj_k/kdj_d/kdj_j。"""
        result = self.engine.calculate_all(self.df)
        assert "kdj_k" in result.columns
        assert "kdj_d" in result.columns
        assert "kdj_j" in result.columns

    def test_calculate_all_has_bollinger_columns(self):
        """calculate_all 结果应包含 boll_upper/boll_mid/boll_lower。"""
        result = self.engine.calculate_all(self.df)
        assert "boll_upper" in result.columns
        assert "boll_mid" in result.columns
        assert "boll_lower" in result.columns

    def test_calculate_all_has_vwap_column(self):
        """calculate_all 结果应包含 vwap 列。"""
        result = self.engine.calculate_all(self.df)
        assert "vwap" in result.columns

    def test_calculate_all_has_obv_column(self):
        """calculate_all 结果应包含 obv 列。"""
        result = self.engine.calculate_all(self.df)
        assert "obv" in result.columns

    def test_calculate_all_has_adx_column(self):
        """calculate_all 结果应包含 adx 列。"""
        result = self.engine.calculate_all(self.df)
        assert "adx" in result.columns

    # ── calculate_all：指标值范围验证 ─────────────────────────────────────

    def test_rsi_range_0_to_100(self):
        """RSI 值应在 [0, 100] 范围内（忽略 NaN）。"""
        result = self.engine.calculate_all(self.df)
        for col in ["rsi_6", "rsi_12", "rsi_24"]:
            valid = result[col].dropna()
            assert (valid >= 0).all(), f"{col} 存在负值"
            assert (valid <= 100).all(), f"{col} 存在超过 100 的值"

    def test_bollinger_order_upper_ge_mid_ge_lower(self):
        """布林带上轨 ≥ 中轨 ≥ 下轨（忽略 NaN）。"""
        result = self.engine.calculate_all(self.df)
        valid_mask = result[["boll_upper", "boll_mid", "boll_lower"]].notna().all(axis=1)
        sub = result[valid_mask]
        assert (sub["boll_upper"] >= sub["boll_mid"]).all()
        assert (sub["boll_mid"] >= sub["boll_lower"]).all()

    def test_adx_range_0_to_100(self):
        """ADX 值应在 [0, 100] 范围内（忽略 NaN）。"""
        result = self.engine.calculate_all(self.df)
        valid = result["adx"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_ma20_equals_rolling_mean(self):
        """ma_20 应等于 close 的 20 日简单移动平均（最后一行对比）。"""
        result = self.engine.calculate_all(self.df)
        expected = self.df["close"].rolling(20).mean().iloc[-1]
        actual = result["ma_20"].iloc[-1]
        assert abs(actual - expected) < 1e-6

    # ── generate_signal ─────────────────────────────────────────────────────

    def test_generate_signal_returns_valid_value(self):
        """generate_signal 应返回 BUY/SELL/HOLD 之一。"""
        result = self.engine.calculate_all(self.df)
        signal = self.engine.generate_signal(result)
        assert signal in ("BUY", "SELL", "HOLD")

    def test_generate_signal_short_df_returns_hold(self):
        """行数 < 2 时 generate_signal 应返回 HOLD。"""
        short_df = make_ohlcv_df(n=1)
        signal = self.engine.generate_signal(short_df)
        assert signal == "HOLD"

    def test_generate_signal_empty_df_returns_hold(self):
        """空 DataFrame 时 generate_signal 应返回 HOLD。"""
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        signal = self.engine.generate_signal(empty_df)
        assert signal == "HOLD"

    def test_generate_signal_buy_conditions(self):
        """构造满足买入条件（MACD 金叉 + RSI<70 + close>MA20）时应返回 BUY。"""
        result = self.engine.calculate_all(self.df)
        # 在最后两行手动注入满足 BUY 条件的指标值
        result = result.copy()
        result.loc[result.index[-2], "macd"] = -0.1
        result.loc[result.index[-2], "macd_signal"] = 0.0
        result.loc[result.index[-1], "macd"] = 0.1
        result.loc[result.index[-1], "macd_signal"] = 0.0
        result.loc[result.index[-1], "rsi_6"] = 60.0
        result.loc[result.index[-1], "close"] = 110.0
        result.loc[result.index[-1], "ma_20"] = 100.0
        signal = self.engine.generate_signal(result)
        assert signal == "BUY"

    def test_generate_signal_sell_conditions_rsi(self):
        """RSI(6) > 80 时 generate_signal 应返回 SELL。"""
        result = self.engine.calculate_all(self.df)
        result = result.copy()
        result.loc[result.index[-1], "rsi_6"] = 85.0
        # 确保 MACD 不触发金叉（避免 BUY 优先）
        result.loc[result.index[-2], "macd"] = 0.1
        result.loc[result.index[-2], "macd_signal"] = -0.1
        result.loc[result.index[-1], "macd"] = 0.1
        result.loc[result.index[-1], "macd_signal"] = -0.1
        signal = self.engine.generate_signal(result)
        assert signal == "SELL"

    def test_generate_signal_no_cross_returns_hold(self):
        """无 MACD 交叉且 RSI 正常时 generate_signal 应返回 HOLD。"""
        result = self.engine.calculate_all(self.df)
        result = result.copy()
        # MACD 始终在信号线下方（无交叉）
        result.loc[result.index[-2], "macd"] = -0.2
        result.loc[result.index[-2], "macd_signal"] = 0.0
        result.loc[result.index[-1], "macd"] = -0.1
        result.loc[result.index[-1], "macd_signal"] = 0.0
        result.loc[result.index[-1], "rsi_6"] = 50.0
        signal = self.engine.generate_signal(result)
        assert signal == "HOLD"

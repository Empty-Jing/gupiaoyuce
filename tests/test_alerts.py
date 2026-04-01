"""
测试 AlertManager 模块
验证各类告警触发条件、去重逻辑和 Bollinger 突破告警。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.core.alert_manager import AlertManager


# ── 测试辅助函数 ──────────────────────────────────────────────────────────────

def make_indicators_df(
    n: int = 25,
    volume_multiplier: float = 1.0,
    macd_golden_cross: bool = False,
    macd_death_cross: bool = False,
    close_above_upper: bool = False,
    close_below_lower: bool = False,
) -> pd.DataFrame:
    """
    生成测试用的指标 DataFrame。
    
    参数:
        n: 行数（最少 20 行，用于成交量均值计算）
        volume_multiplier: 最后一行成交量相对于前 20 日均量的倍数
        macd_golden_cross: 是否在最后两行制造 MACD 金叉
        macd_death_cross: 是否在最后两行制造 MACD 死叉
        close_above_upper: 是否让最后一行 close 突破布林上轨
        close_below_lower: 是否让最后一行 close 跌破布林下轨
    """
    base_vol = 1_000_000.0
    volumes = [base_vol] * n
    if volume_multiplier != 1.0:
        volumes[-1] = base_vol * volume_multiplier

    close = [10.0] * n
    upper = [12.0] * n
    lower = [8.0] * n
    if close_above_upper:
        close[-1] = 13.0
    if close_below_lower:
        close[-1] = 7.0

    # MACD 默认无交叉（macd 始终低于 signal）
    macd = [-0.1] * n
    signal = [0.0] * n
    if macd_golden_cross and n >= 2:
        macd[-2] = -0.1   # 前一行 macd < signal
        signal[-2] = 0.0
        macd[-1] = 0.1    # 当前行 macd > signal（金叉）
        signal[-1] = 0.0
    if macd_death_cross and n >= 2:
        macd[-2] = 0.1    # 前一行 macd > signal
        signal[-2] = 0.0
        macd[-1] = -0.1   # 当前行 macd < signal（死叉）
        signal[-1] = 0.0

    return pd.DataFrame({
        "volume": volumes,
        "close": close,
        "boll_upper": upper,
        "boll_lower": lower,
        "macd": macd,
        "macd_signal": signal,
    })


# ── 测试类 ────────────────────────────────────────────────────────────────────

class TestPriceChangeAlert:
    """涨跌幅异常告警测试。"""

    def setup_method(self):
        self.manager = AlertManager()

    def test_price_change_triggers_above_threshold(self):
        """涨跌幅 > 5% 时应触发 price_change 告警。"""
        alerts = self.manager.check_alerts(
            symbol="000001",
            realtime_data={"change_pct": 6.5},
            indicators_df=pd.DataFrame(),
            sentiment_score=0.0,
        )
        price_alerts = [a for a in alerts if a["alert_type"] == "price_change"]
        assert len(price_alerts) == 1
        assert price_alerts[0]["symbol"] == "000001"
        assert price_alerts[0]["threshold"] == 5.0

    def test_price_change_triggers_negative_drop(self):
        """跌幅 > 5%（绝对值）时也应触发 price_change 告警。"""
        alerts = self.manager.check_alerts(
            symbol="000001",
            realtime_data={"change_pct": -7.2},
            indicators_df=pd.DataFrame(),
            sentiment_score=0.0,
        )
        price_alerts = [a for a in alerts if a["alert_type"] == "price_change"]
        assert len(price_alerts) == 1

    def test_price_change_does_not_trigger_below_threshold(self):
        """涨跌幅 ≤ 5% 时不应触发 price_change 告警。"""
        alerts = self.manager.check_alerts(
            symbol="000001",
            realtime_data={"change_pct": 3.0},
            indicators_df=pd.DataFrame(),
            sentiment_score=0.0,
        )
        price_alerts = [a for a in alerts if a["alert_type"] == "price_change"]
        assert len(price_alerts) == 0

    def test_price_change_at_exactly_5_not_triggered(self):
        """涨跌幅恰好等于 5% 时不应触发（条件为 > 5.0）。"""
        alerts = self.manager.check_alerts(
            symbol="000001",
            realtime_data={"change_pct": 5.0},
            indicators_df=pd.DataFrame(),
            sentiment_score=0.0,
        )
        price_alerts = [a for a in alerts if a["alert_type"] == "price_change"]
        assert len(price_alerts) == 0

    def test_price_change_alert_has_required_keys(self):
        """price_change 告警字典应包含所有必需字段。"""
        alerts = self.manager.check_alerts(
            symbol="000001",
            realtime_data={"change_pct": 8.0},
            indicators_df=pd.DataFrame(),
            sentiment_score=0.0,
        )
        assert len(alerts) == 1
        alert = alerts[0]
        required_keys = {"symbol", "alert_type", "trigger_value", "threshold", "message"}
        assert required_keys.issubset(alert.keys())


class TestVolumeSpikAlert:
    """成交量异常告警测试。"""

    def setup_method(self):
        self.manager = AlertManager()

    def test_volume_spike_triggers_when_3x_avg(self):
        """当前成交量超过 20 日均量 3 倍时应触发 volume_spike。"""
        df = make_indicators_df(n=25, volume_multiplier=4.0)
        alerts = self.manager.check_alerts(
            symbol="000001",
            realtime_data={"change_pct": 0.0},
            indicators_df=df,
            sentiment_score=0.0,
        )
        vol_alerts = [a for a in alerts if a["alert_type"] == "volume_spike"]
        assert len(vol_alerts) == 1

    def test_volume_spike_not_triggers_when_below_3x(self):
        """成交量不足 20 日均量 3 倍时不应触发 volume_spike。"""
        df = make_indicators_df(n=25, volume_multiplier=2.0)
        alerts = self.manager.check_alerts(
            symbol="000001",
            realtime_data={"change_pct": 0.0},
            indicators_df=df,
            sentiment_score=0.0,
        )
        vol_alerts = [a for a in alerts if a["alert_type"] == "volume_spike"]
        assert len(vol_alerts) == 0

    def test_volume_spike_not_triggers_when_df_too_short(self):
        """DataFrame 行数 < 20 时不应触发 volume_spike。"""
        df = make_indicators_df(n=15, volume_multiplier=5.0)
        alerts = self.manager.check_alerts(
            symbol="000001",
            realtime_data={"change_pct": 0.0},
            indicators_df=df,
            sentiment_score=0.0,
        )
        vol_alerts = [a for a in alerts if a["alert_type"] == "volume_spike"]
        assert len(vol_alerts) == 0

    def test_volume_spike_not_triggers_with_empty_df(self):
        """空 DataFrame 时不应触发 volume_spike。"""
        alerts = self.manager.check_alerts(
            symbol="000001",
            realtime_data={"change_pct": 0.0},
            indicators_df=pd.DataFrame(),
            sentiment_score=0.0,
        )
        vol_alerts = [a for a in alerts if a["alert_type"] == "volume_spike"]
        assert len(vol_alerts) == 0


class TestMACDCrossAlert:
    """MACD 金叉/死叉告警测试。"""

    def setup_method(self):
        self.manager = AlertManager()

    def test_macd_golden_cross_triggers(self):
        """MACD 金叉时应触发 macd_cross 告警。"""
        df = make_indicators_df(n=5, macd_golden_cross=True)
        alerts = self.manager.check_alerts(
            symbol="000001",
            realtime_data={"change_pct": 0.0},
            indicators_df=df,
            sentiment_score=0.0,
        )
        macd_alerts = [a for a in alerts if a["alert_type"] == "macd_cross"]
        assert len(macd_alerts) == 1
        assert "金叉" in macd_alerts[0]["message"]

    def test_macd_death_cross_triggers(self):
        """MACD 死叉时应触发 macd_cross 告警。"""
        df = make_indicators_df(n=5, macd_death_cross=True)
        alerts = self.manager.check_alerts(
            symbol="000001",
            realtime_data={"change_pct": 0.0},
            indicators_df=df,
            sentiment_score=0.0,
        )
        macd_alerts = [a for a in alerts if a["alert_type"] == "macd_cross"]
        assert len(macd_alerts) == 1
        assert "死叉" in macd_alerts[0]["message"]

    def test_macd_no_cross_no_alert(self):
        """无 MACD 交叉时不应触发 macd_cross 告警。"""
        df = make_indicators_df(n=5)  # 默认 macd < signal，无交叉
        alerts = self.manager.check_alerts(
            symbol="000001",
            realtime_data={"change_pct": 0.0},
            indicators_df=df,
            sentiment_score=0.0,
        )
        macd_alerts = [a for a in alerts if a["alert_type"] == "macd_cross"]
        assert len(macd_alerts) == 0


class TestSentimentShiftAlert:
    """情感突变告警测试。"""

    def setup_method(self):
        self.manager = AlertManager()

    def test_positive_sentiment_above_threshold(self):
        """正面情感分数 > 0.5 时应触发 sentiment_shift。"""
        alerts = self.manager.check_alerts(
            symbol="000001",
            realtime_data={"change_pct": 0.0},
            indicators_df=pd.DataFrame(),
            sentiment_score=0.8,
        )
        sent_alerts = [a for a in alerts if a["alert_type"] == "sentiment_shift"]
        assert len(sent_alerts) == 1

    def test_negative_sentiment_below_threshold(self):
        """负面情感分数 < -0.5 时应触发 sentiment_shift。"""
        alerts = self.manager.check_alerts(
            symbol="000001",
            realtime_data={"change_pct": 0.0},
            indicators_df=pd.DataFrame(),
            sentiment_score=-0.7,
        )
        sent_alerts = [a for a in alerts if a["alert_type"] == "sentiment_shift"]
        assert len(sent_alerts) == 1

    def test_neutral_sentiment_no_alert(self):
        """情感分数绝对值 ≤ 0.5 时不应触发 sentiment_shift。"""
        alerts = self.manager.check_alerts(
            symbol="000001",
            realtime_data={"change_pct": 0.0},
            indicators_df=pd.DataFrame(),
            sentiment_score=0.3,
        )
        sent_alerts = [a for a in alerts if a["alert_type"] == "sentiment_shift"]
        assert len(sent_alerts) == 0

    def test_zero_sentiment_no_alert(self):
        """情感分数为 0 时不应触发告警。"""
        alerts = self.manager.check_alerts(
            symbol="000001",
            realtime_data={"change_pct": 0.0},
            indicators_df=pd.DataFrame(),
            sentiment_score=0.0,
        )
        assert len(alerts) == 0


class TestBollingerBreakAlert:
    """布林突破告警测试。"""

    def setup_method(self):
        self.manager = AlertManager()

    def test_bollinger_upper_break_triggers(self):
        """收盘价突破布林上轨时应触发 bollinger_break 告警。"""
        df = make_indicators_df(n=5, close_above_upper=True)
        alerts = self.manager.check_alerts(
            symbol="000001",
            realtime_data={"change_pct": 0.0},
            indicators_df=df,
            sentiment_score=0.0,
        )
        boll_alerts = [a for a in alerts if a["alert_type"] == "bollinger_break"]
        assert len(boll_alerts) == 1
        assert "上穿上轨" in boll_alerts[0]["message"]

    def test_bollinger_lower_break_triggers(self):
        """收盘价跌破布林下轨时应触发 bollinger_break 告警。"""
        df = make_indicators_df(n=5, close_below_lower=True)
        alerts = self.manager.check_alerts(
            symbol="000001",
            realtime_data={"change_pct": 0.0},
            indicators_df=df,
            sentiment_score=0.0,
        )
        boll_alerts = [a for a in alerts if a["alert_type"] == "bollinger_break"]
        assert len(boll_alerts) == 1
        assert "下穿下轨" in boll_alerts[0]["message"]

    def test_bollinger_no_break_no_alert(self):
        """收盘价在布林带内时不应触发 bollinger_break。"""
        df = make_indicators_df(n=5)  # close=10.0, upper=12.0, lower=8.0
        alerts = self.manager.check_alerts(
            symbol="000001",
            realtime_data={"change_pct": 0.0},
            indicators_df=df,
            sentiment_score=0.0,
        )
        boll_alerts = [a for a in alerts if a["alert_type"] == "bollinger_break"]
        assert len(boll_alerts) == 0


class TestAlertDeduplication:
    """告警去重逻辑测试。"""

    def test_same_alert_within_1_hour_suppressed(self):
        """同一股票同一类型告警在 1 小时内重复触发应被抑制。"""
        manager = AlertManager()
        realtime = {"change_pct": 6.5}
        # 第一次触发
        alerts1 = manager.check_alerts(
            symbol="000001",
            realtime_data=realtime,
            indicators_df=pd.DataFrame(),
            sentiment_score=0.0,
        )
        assert len([a for a in alerts1 if a["alert_type"] == "price_change"]) == 1

        # 第二次触发（同一告警类型，应被去重）
        alerts2 = manager.check_alerts(
            symbol="000001",
            realtime_data=realtime,
            indicators_df=pd.DataFrame(),
            sentiment_score=0.0,
        )
        assert len([a for a in alerts2 if a["alert_type"] == "price_change"]) == 0

    def test_alert_triggers_after_dedup_window_expired(self):
        """去重缓存超过 1 小时后，同一告警应再次触发。"""
        manager = AlertManager()
        realtime = {"change_pct": 6.5}

        # 第一次触发，设置缓存时间为 2 小时前（已过期）
        manager.check_alerts(
            symbol="000001",
            realtime_data=realtime,
            indicators_df=pd.DataFrame(),
            sentiment_score=0.0,
        )
        # 手动将缓存时间设置为 2 小时前（模拟超时）
        manager._dedup_cache["000001_price_change"] = datetime.now() - timedelta(hours=2)

        # 第二次触发应成功（缓存已过期）
        alerts = manager.check_alerts(
            symbol="000001",
            realtime_data=realtime,
            indicators_df=pd.DataFrame(),
            sentiment_score=0.0,
        )
        assert len([a for a in alerts if a["alert_type"] == "price_change"]) == 1

    def test_different_symbols_not_deduplicated(self):
        """不同股票的同类告警不应相互去重。"""
        manager = AlertManager()
        realtime = {"change_pct": 6.5}

        alerts1 = manager.check_alerts(
            symbol="000001",
            realtime_data=realtime,
            indicators_df=pd.DataFrame(),
            sentiment_score=0.0,
        )
        alerts2 = manager.check_alerts(
            symbol="600036",
            realtime_data=realtime,
            indicators_df=pd.DataFrame(),
            sentiment_score=0.0,
        )
        assert len([a for a in alerts1 if a["alert_type"] == "price_change"]) == 1
        assert len([a for a in alerts2 if a["alert_type"] == "price_change"]) == 1

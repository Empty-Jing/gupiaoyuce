"""
告警管理模块 — AlertManager
负责检测股票异常信号并通过邮件/Webhook 发送告警通知。
"""
import logging
import smtplib
import ssl
from datetime import datetime, timedelta
from email.mime.text import MIMEText

import pandas as pd
import requests

from app.config import settings

logger = logging.getLogger(__name__)

# 去重窗口：1 小时
_DEDUP_WINDOW = timedelta(hours=1)

# 北向资金异常阈值：5 亿
_NORTH_FLOW_THRESHOLD = 500_000_000.0


class AlertManager:
    """告警管理器，负责多维度告警检测与通知发送。"""

    def __init__(self) -> None:
        # 从 settings 读取 SMTP 配置
        self._smtp_sender: str = settings.ALERT_EMAIL_SENDER
        self._smtp_password: str = settings.ALERT_EMAIL_PASSWORD
        self._smtp_receiver: str = settings.ALERT_EMAIL_RECEIVER
        self._smtp_host: str = settings.ALERT_EMAIL_SMTP_HOST
        self._smtp_port: int = settings.ALERT_EMAIL_SMTP_PORT
        self._webhook_url: str = settings.ALERT_WEBHOOK_URL

        # 告警去重缓存：key = "{symbol}_{alert_type}"，value = 上次触发时间
        self._dedup_cache: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    def check_alerts(
        self,
        symbol: str,
        realtime_data: dict,
        indicators_df: pd.DataFrame,
        sentiment_score: float = 0.0,
    ) -> list:
        """
        检测多维度告警条件，返回触发告警列表（plain dict）。

        每个告警字典格式：
        {"symbol": str, "alert_type": str, "trigger_value": float,
         "threshold": float, "message": str}
        """
        triggered: list[dict] = []

        # --- a. 涨跌幅异常 ---
        change_pct = float(realtime_data.get("change_pct", 0))
        if abs(change_pct) > 5.0:
            alert = {
                "symbol": symbol,
                "alert_type": "price_change",
                "trigger_value": change_pct,
                "threshold": 5.0,
                "message": (
                    f"【涨跌幅异常】{symbol} 涨跌幅 {change_pct:.2f}%，超过阈值 5%"
                ),
            }
            if not self._is_duplicate(f"{symbol}_price_change"):
                self._update_cache(f"{symbol}_price_change")
                triggered.append(alert)

        # --- b. 成交量异常 ---
        if (
            indicators_df is not None
            and not indicators_df.empty
            and "volume" in indicators_df.columns
            and len(indicators_df) >= 20
        ):
            volumes = indicators_df["volume"].dropna()
            if len(volumes) >= 20:
                avg_vol_20 = float(volumes.iloc[-20:].mean())
                current_vol = float(volumes.iloc[-1])
                if avg_vol_20 > 0 and current_vol > avg_vol_20 * 3:
                    alert = {
                        "symbol": symbol,
                        "alert_type": "volume_spike",
                        "trigger_value": current_vol,
                        "threshold": avg_vol_20 * 3,
                        "message": (
                            f"【成交量异常】{symbol} 当前成交量 {current_vol:.0f}，"
                            f"超过20日均量 {avg_vol_20:.0f} 的3倍"
                        ),
                    }
                    if not self._is_duplicate(f"{symbol}_volume_spike"):
                        self._update_cache(f"{symbol}_volume_spike")
                        triggered.append(alert)

        # --- c. MACD 金叉/死叉 ---
        if (
            indicators_df is not None
            and not indicators_df.empty
            and "macd" in indicators_df.columns
            and "macd_signal" in indicators_df.columns
            and len(indicators_df) >= 2
        ):
            macd_vals = indicators_df["macd"].dropna()
            signal_vals = indicators_df["macd_signal"].dropna()
            # 对齐索引取最后两行
            aligned = pd.concat([macd_vals, signal_vals], axis=1).dropna()
            if len(aligned) >= 2:
                macd_prev = float(aligned.iloc[-2, 0])
                signal_prev = float(aligned.iloc[-2, 1])
                macd_curr = float(aligned.iloc[-1, 0])
                signal_curr = float(aligned.iloc[-1, 1])
                # 金叉：macd 从下方穿越 signal
                golden_cross = macd_prev < signal_prev and macd_curr > signal_curr
                # 死叉：macd 从上方穿越 signal
                death_cross = macd_prev > signal_prev and macd_curr < signal_curr
                if golden_cross or death_cross:
                    cross_type = "金叉" if golden_cross else "死叉"
                    alert = {
                        "symbol": symbol,
                        "alert_type": "macd_cross",
                        "trigger_value": macd_curr,
                        "threshold": signal_curr,
                        "message": (
                            f"【MACD {cross_type}】{symbol} MACD={macd_curr:.4f}，"
                            f"Signal={signal_curr:.4f}，发生{cross_type}"
                        ),
                    }
                    if not self._is_duplicate(f"{symbol}_macd_cross"):
                        self._update_cache(f"{symbol}_macd_cross")
                        triggered.append(alert)

        # --- d. 情感突变 ---
        if abs(sentiment_score) > 0.5:
            alert = {
                "symbol": symbol,
                "alert_type": "sentiment_shift",
                "trigger_value": sentiment_score,
                "threshold": 0.5,
                "message": (
                    f"【情感突变】{symbol} 情感得分 {sentiment_score:.3f}，"
                    f"绝对值超过阈值 0.5"
                ),
            }
            if not self._is_duplicate(f"{symbol}_sentiment_shift"):
                self._update_cache(f"{symbol}_sentiment_shift")
                triggered.append(alert)

        # --- e. 北向资金异常 ---
        if (
            indicators_df is not None
            and not indicators_df.empty
            and "north_net_inflow" in indicators_df.columns
        ):
            north_val = indicators_df["north_net_inflow"].dropna()
            if len(north_val) > 0:
                last_inflow = float(north_val.iloc[-1])
                if abs(last_inflow) > _NORTH_FLOW_THRESHOLD:
                    alert = {
                        "symbol": symbol,
                        "alert_type": "north_flow",
                        "trigger_value": last_inflow,
                        "threshold": _NORTH_FLOW_THRESHOLD,
                        "message": (
                            f"【北向资金异常】{symbol} 北向净流入 {last_inflow:.0f}，"
                            f"超过阈值 {_NORTH_FLOW_THRESHOLD:.0f}"
                        ),
                    }
                    if not self._is_duplicate(f"{symbol}_north_flow"):
                        self._update_cache(f"{symbol}_north_flow")
                        triggered.append(alert)

        # --- f. 布林突破 ---
        if (
            indicators_df is not None
            and not indicators_df.empty
            and "close" in indicators_df.columns
            and "boll_upper" in indicators_df.columns
            and "boll_lower" in indicators_df.columns
        ):
            close_vals = indicators_df["close"].dropna()
            boll_upper_vals = indicators_df["boll_upper"].dropna()
            boll_lower_vals = indicators_df["boll_lower"].dropna()
            if len(close_vals) > 0 and len(boll_upper_vals) > 0 and len(boll_lower_vals) > 0:
                close_last = float(close_vals.iloc[-1])
                upper_last = float(boll_upper_vals.iloc[-1])
                lower_last = float(boll_lower_vals.iloc[-1])
                if close_last > upper_last or close_last < lower_last:
                    direction = "上穿上轨" if close_last > upper_last else "下穿下轨"
                    alert = {
                        "symbol": symbol,
                        "alert_type": "bollinger_break",
                        "trigger_value": close_last,
                        "threshold": upper_last if close_last > upper_last else lower_last,
                        "message": (
                            f"【布林突破】{symbol} 收盘价 {close_last:.2f} {direction}，"
                            f"上轨 {upper_last:.2f}，下轨 {lower_last:.2f}"
                        ),
                    }
                    if not self._is_duplicate(f"{symbol}_bollinger_break"):
                        self._update_cache(f"{symbol}_bollinger_break")
                        triggered.append(alert)

        return triggered

    def send_email_alert(self, alert: dict) -> bool:
        """
        通过 SMTP_SSL 发送告警邮件。
        未配置 sender/receiver 时记录 WARNING 并返回 False。
        """
        if not self._smtp_sender or not self._smtp_receiver:
            logger.warning(
                "邮件告警未配置（ALERT_EMAIL_SENDER 或 ALERT_EMAIL_RECEIVER 为空），跳过发送"
            )
            return False

        subject = f"[股票告警] {alert.get('symbol', '')} - {alert.get('alert_type', '')}"
        body = alert.get("message", "")

        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = self._smtp_sender
        msg["To"] = self._smtp_receiver

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(
                self._smtp_host, self._smtp_port, context=context
            ) as server:
                server.login(self._smtp_sender, self._smtp_password)
                server.sendmail(
                    self._smtp_sender, [self._smtp_receiver], msg.as_string()
                )
            logger.info(
                "邮件告警发送成功: %s -> %s", alert.get("symbol"), alert.get("alert_type")
            )
            return True
        except Exception as exc:
            logger.error("邮件告警发送失败: %s", exc)
            return False

    def send_webhook_alert(self, alert: dict) -> bool:
        """
        通过 HTTP POST 发送 Webhook 告警（企业微信兼容格式）。
        未配置 URL 时记录 WARNING 并返回 False。
        """
        if not self._webhook_url:
            logger.warning("Webhook 告警未配置（ALERT_WEBHOOK_URL 为空），跳过发送")
            return False

        payload = {
            "msgtype": "text",
            "text": {"content": alert.get("message", "")},
        }

        try:
            resp = requests.post(self._webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
            # 企业微信等平台返回 HTTP 200 但 body 中可能携带业务错误码
            try:
                resp_json = resp.json()
                errcode = resp_json.get("errcode", 0)
                if errcode != 0:
                    logger.error(
                        "Webhook 告警发送失败（业务错误）: errcode=%s, errmsg=%s",
                        errcode,
                        resp_json.get("errmsg", ""),
                    )
                    return False
            except ValueError:
                pass  # 响应体非 JSON，忽略业务错误码检查
            logger.info(
                "Webhook 告警发送成功: %s -> %s", alert.get("symbol"), alert.get("alert_type")
            )
            return True
        except Exception as exc:
            logger.error("Webhook 告警发送失败: %s", exc)
            return False

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _is_duplicate(self, key: str) -> bool:
        """
        检查 key 是否在 1 小时内已触发过。
        同时清理超过 1 小时的过期缓存条目。
        """
        now = datetime.now()

        # 清理过期条目
        expired = [k for k, ts in self._dedup_cache.items() if now - ts > _DEDUP_WINDOW]
        for k in expired:
            del self._dedup_cache[k]

        if key in self._dedup_cache:
            last_triggered = self._dedup_cache[key]
            if now - last_triggered < _DEDUP_WINDOW:
                return True  # 仍在去重窗口内
        return False

    def _update_cache(self, key: str) -> None:
        """更新去重缓存时间戳。"""
        self._dedup_cache[key] = datetime.now()

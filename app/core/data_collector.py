import asyncio
import logging
import os
import re
import threading
import time
from datetime import date as date_type
from typing import Callable, Optional, TypeVar, cast

import akshare as ak
import pandas as pd
import requests

_T = TypeVar("_T")

logger = logging.getLogger(__name__)

_PROXY_VARS = ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")
_NO_PROXY_VARS = ("NO_PROXY", "no_proxy")
_proxy_warned = False
_proxy_lock = threading.RLock()

# 东方财富 API 熔断器：失败后 _EM_CIRCUIT_COOLDOWN 秒内跳过
_EM_CIRCUIT_COOLDOWN = 300  # 5 分钟
_em_last_failure: float = 0.0  # 上次失败时间戳
_em_circuit_lock = threading.Lock()


def _em_circuit_open() -> bool:
    """检查东方财富熔断器是否打开（即是否应跳过东方财富请求）"""
    return (time.time() - _em_last_failure) < _EM_CIRCUIT_COOLDOWN


def _em_circuit_trip() -> None:
    """触发东方财富熔断"""
    global _em_last_failure
    with _em_circuit_lock:
        _em_last_failure = time.time()
        logger.info("东方财富 API 熔断器触发，%d 秒内将跳过并直接使用新浪数据源", _EM_CIRCUIT_COOLDOWN)

HIST_COLUMNS_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "turnover",
    "振幅": "amplitude",
    "涨跌幅": "change_pct",
    "涨跌额": "change_amount",
    "换手率": "turnover_rate",
    "股票代码": "symbol_code",  # AKShare 1.18+ 版本会额外返回此列
}

REALTIME_COLUMNS_MAP = {
    "代码": "symbol",
    "名称": "name",
    "最新价": "price",
    "涨跌幅": "change_pct",
    "成交量": "volume",
    "成交额": "amount",
    "最高": "high",
    "最低": "low",
    "今开": "open",
    "昨收": "pre_close",
}

# 新浪实时行情列名映射 (stock_zh_a_spot → 统一格式)
REALTIME_SINA_COLUMNS_MAP = {
    "代码": "symbol",
    "名称": "name",
    "最新价": "price",
    "涨跌幅": "change_pct",
    "成交量": "volume",
    "成交额": "amount",
    "最高": "high",
    "最低": "low",
    "今开": "open",
    "昨收": "pre_close",
}

# 新浪日K线英文列名 → 中文列名映射（与东方财富 stock_zh_a_hist 原始输出一致）
HIST_SINA_TO_CN_MAP = {
    "date": "日期",
    "open": "开盘",
    "high": "最高",
    "low": "最低",
    "close": "收盘",
    "volume": "成交量",
    "amount": "成交额",
    "turnover": "换手率",
}

# 新浪日K线 symbol 参数需要 "sh"/"sz"/"bj" 前缀
_SINA_SYMBOL_PREFIX = {"6": "sh", "0": "sz", "3": "sz", "4": "bj", "8": "bj", "9": "bj"}


def _retry(func: Callable[[], _T], max_retries: int = 1, delay: float = 0.5) -> _T:
    global _proxy_warned
    has_proxy = any(k in os.environ for k in _PROXY_VARS)
    if has_proxy and not _proxy_warned:
        logger.info("检测到系统代理，AkShare 请求将自动绕过代理直连")
        _proxy_warned = True

    for attempt in range(max_retries):
        try:
            if has_proxy:
                with _proxy_lock:
                    saved = {k: os.environ.get(k) for k in _PROXY_VARS + _NO_PROXY_VARS}
                    for k in _PROXY_VARS:
                        os.environ.pop(k, None)
                    os.environ["NO_PROXY"] = "*"
                    os.environ["no_proxy"] = "*"
                    try:
                        return func()
                    finally:
                        for k in _PROXY_VARS + _NO_PROXY_VARS:
                            v = saved[k]
                            if v is None:
                                os.environ.pop(k, None)
                            else:
                                os.environ[k] = v
            else:
                return func()
        except Exception as e:
            logger.warning(f"第 {attempt + 1} 次尝试失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("_retry: unreachable")


# 新浪 hq.sinajs.cn 实时行情字段索引（0-based）
_SINA_RT_IDX_NAME = 0
_SINA_RT_IDX_OPEN = 1
_SINA_RT_IDX_PRE_CLOSE = 2
_SINA_RT_IDX_PRICE = 3
_SINA_RT_IDX_HIGH = 4
_SINA_RT_IDX_LOW = 5
_SINA_RT_IDX_VOLUME = 8   # 股
_SINA_RT_IDX_AMOUNT = 9   # 元

_SINA_RT_PATTERN = re.compile(r'var hq_str_(\w+)="([^"]*)"')


def _fetch_sina_realtime(symbols: list[str]) -> pd.DataFrame:
    sina_codes = []
    for s in symbols:
        prefix = _SINA_SYMBOL_PREFIX.get(s[0], "sz")
        sina_codes.append(f"{prefix}{s}")

    saved_proxies = {}
    has_proxy = any(k in os.environ for k in _PROXY_VARS)
    if has_proxy:
        for k in _PROXY_VARS:
            v = os.environ.pop(k, None)
            if v is not None:
                saved_proxies[k] = v
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"

    try:
        url = f"https://hq.sinajs.cn/list={','.join(sina_codes)}"
        resp = requests.get(url, headers={"Referer": "https://finance.sina.com.cn"}, timeout=5)
        resp.encoding = "gbk"
        text = resp.text
    finally:
        if has_proxy:
            for k in _PROXY_VARS:
                os.environ.pop(k, None)
            os.environ.pop("NO_PROXY", None)
            os.environ.pop("no_proxy", None)
            for k, v in saved_proxies.items():
                os.environ[k] = v

    rows = []
    for match in _SINA_RT_PATTERN.finditer(text):
        code_with_prefix = match.group(1)
        fields = match.group(2).split(",")
        if len(fields) < 10 or not fields[_SINA_RT_IDX_PRICE]:
            continue
        raw_symbol = code_with_prefix[2:]
        try:
            price = float(fields[_SINA_RT_IDX_PRICE])
            pre_close = float(fields[_SINA_RT_IDX_PRE_CLOSE])
            change_pct = round((price - pre_close) / pre_close * 100, 2) if pre_close else 0.0
        except (ValueError, ZeroDivisionError):
            change_pct = 0.0
            price = 0.0
            pre_close = 0.0

        rows.append({
            "代码": raw_symbol,
            "名称": fields[_SINA_RT_IDX_NAME],
            "最新价": price,
            "涨跌幅": change_pct,
            "成交量": int(float(fields[_SINA_RT_IDX_VOLUME])) if fields[_SINA_RT_IDX_VOLUME] else 0,
            "成交额": float(fields[_SINA_RT_IDX_AMOUNT]) if fields[_SINA_RT_IDX_AMOUNT] else 0.0,
            "最高": float(fields[_SINA_RT_IDX_HIGH]) if fields[_SINA_RT_IDX_HIGH] else 0.0,
            "最低": float(fields[_SINA_RT_IDX_LOW]) if fields[_SINA_RT_IDX_LOW] else 0.0,
            "今开": float(fields[_SINA_RT_IDX_OPEN]) if fields[_SINA_RT_IDX_OPEN] else 0.0,
            "昨收": pre_close,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


class DataCollector:

    def fetch_stock_history(
        self,
        symbol: str,
        period: str = "daily",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        if not _em_circuit_open():
            try:
                kwargs: dict = {"symbol": symbol, "period": period, "adjust": adjust}
                if start_date is not None:
                    kwargs["start_date"] = start_date
                if end_date is not None:
                    kwargs["end_date"] = end_date
                df = _retry(lambda: ak.stock_zh_a_hist(**kwargs))
                df = pd.DataFrame(df)
                if "股票代码" in df.columns:
                    df = df.drop(columns=["股票代码"])
                return cast(pd.DataFrame, df)
            except Exception as em_exc:
                _em_circuit_trip()
                logger.warning(f"东方财富历史行情获取失败，降级到新浪数据源: {em_exc}")
        else:
            logger.debug("东方财富熔断中，直接使用新浪历史数据源")

        prefix = _SINA_SYMBOL_PREFIX.get(symbol[0], "sz")
        sina_symbol = f"{prefix}{symbol}"
        sina_kwargs: dict = {"symbol": sina_symbol}
        if adjust == "qfq":
            sina_kwargs["adjust"] = "qfq"
        elif adjust == "hfq":
            sina_kwargs["adjust"] = "hfq"
        if start_date is not None:
            sina_kwargs["start_date"] = start_date
        if end_date is not None:
            sina_kwargs["end_date"] = end_date
        df = _retry(lambda: ak.stock_zh_a_daily(**sina_kwargs))
        df = pd.DataFrame(df).rename(columns=HIST_SINA_TO_CN_MAP)
        if "outstanding_share" in df.columns:
            df = df.drop(columns=["outstanding_share"])
        return cast(pd.DataFrame, df)

    def fetch_realtime_quotes(self, symbols: list) -> pd.DataFrame:
        if not _em_circuit_open():
            try:
                raw = _retry(lambda: ak.stock_zh_a_spot_em())
                df = pd.DataFrame(raw[raw["代码"].isin(symbols)].copy())
                df = cast(pd.DataFrame, df.rename(columns=REALTIME_COLUMNS_MAP))
            except Exception as em_exc:
                _em_circuit_trip()
                logger.warning(f"东方财富实时行情获取失败，降级到新浪数据源: {em_exc}")
                raw = _fetch_sina_realtime(symbols)
                df = pd.DataFrame(raw[raw["代码"].isin(symbols)].copy()) if not raw.empty else pd.DataFrame()
                df = cast(pd.DataFrame, df.rename(columns=REALTIME_SINA_COLUMNS_MAP))
        else:
            logger.debug("东方财富熔断中，直接使用新浪实时数据源")
            raw = _fetch_sina_realtime(symbols)
            df = pd.DataFrame(raw[raw["代码"].isin(symbols)].copy()) if not raw.empty else pd.DataFrame()
            df = cast(pd.DataFrame, df.rename(columns=REALTIME_SINA_COLUMNS_MAP))
        keep_cols = ["symbol", "name", "price", "change_pct", "volume",
                     "amount", "high", "low", "open", "pre_close"]
        existing_cols = [c for c in keep_cols if c in df.columns]
        result = cast(pd.DataFrame, df[existing_cols].reset_index(drop=True))
        return result

    def fetch_intraday(self, symbol: str) -> pd.DataFrame:
        df: pd.DataFrame = _retry(lambda: ak.stock_intraday_em(symbol=symbol))
        return df

    def fetch_fund_flow(self, symbol: str) -> pd.DataFrame:
        market = "沪A" if symbol.startswith("6") else "深A"
        df: pd.DataFrame = _retry(
            lambda: ak.stock_individual_fund_flow(stock=symbol, market=market)
        )
        return df

    def fetch_north_flow(self) -> pd.DataFrame:
        df: pd.DataFrame = _retry(
            lambda: ak.stock_hsgt_hist_em(symbol="北向资金")
        )
        return df

    def fetch_news(self, symbol: str) -> pd.DataFrame:
        df: pd.DataFrame = _retry(lambda: ak.stock_news_em(symbol=symbol))
        return df

    def fetch_stock_code_name_list(self) -> pd.DataFrame:
        """获取 A 股代码+名称列表（轻量级，用于搜索）。

        直接调用交易所公开的股票代码名称接口，无需拉取全量行情。
        返回 DataFrame 包含 ``code`` 和 ``name`` 两列。
        """
        df: pd.DataFrame = _retry(lambda: ak.stock_info_a_code_name())
        return cast(pd.DataFrame, df)

    def fetch_stock_list(self) -> pd.DataFrame:
        if not _em_circuit_open():
            try:
                df: pd.DataFrame = _retry(lambda: ak.stock_zh_a_spot_em())
                return df
            except Exception as em_exc:
                _em_circuit_trip()
                logger.warning(f"东方财富全市场行情失败，降级到新浪: {em_exc}")
        else:
            logger.debug("东方财富熔断中，直接使用备用数据源获取股票列表")
        # 第二级：新浪实时行情
        try:
            df = _retry(lambda: ak.stock_zh_a_spot())
            return df
        except Exception as sina_exc:
            logger.warning(f"新浪全市场行情失败，降级到股票代码名称列表: {sina_exc}")
        # 第三级：交易所股票代码列表
        df = _retry(lambda: ak.stock_info_a_code_name())
        return cast(pd.DataFrame, df)

    def save_history_to_db(
        self,
        symbol: str,
        period: str = "daily",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq",
    ) -> int:
        from app.database.crud import upsert_stock_daily
        from app.database.session import init_db

        df = self.fetch_stock_history(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )

        if df.empty:
            logger.info(f"[{symbol}] 无数据可写入")
            return 0

        records: list[dict] = []
        for _, row in df.iterrows():
            turnover_raw = row.get("成交额")
            raw_date = row["日期"]
            if isinstance(raw_date, date_type):
                parsed_date = raw_date
            else:
                parsed_date = pd.Timestamp(str(raw_date)).date()
            records.append({
                "symbol": symbol,
                "date": parsed_date,
                "open": float(row["开盘"]),
                "high": float(row["最高"]),
                "low": float(row["最低"]),
                "close": float(row["收盘"]),
                "volume": int(row["成交量"]),
                "turnover": float(turnover_raw) if turnover_raw is not None and pd.notna(turnover_raw) else None,
                "adjust_type": adjust,
            })

        async def _write() -> int:
            await init_db()
            return await upsert_stock_daily(records)

        try:
            inserted = asyncio.run(_write())
        except RuntimeError as e:
            logger.warning(f"asyncio.run() 失败，尝试已有事件循环: {e}")
            loop = asyncio.get_event_loop()
            inserted = loop.run_until_complete(_write())

        logger.info(f"[{symbol}] 写入 {inserted} 条新记录（跳过已存在）")
        return inserted

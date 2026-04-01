import asyncio
import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.database import crud

logger = logging.getLogger(__name__)

router = APIRouter()


class WatchlistAddRequest(BaseModel):
    symbol: str
    name: str
    market: str = "A"


@router.get("/")
async def get_watchlist():
    """获取自选股列表"""
    items = await crud.get_watchlist()
    return [
        {
            "id": item.id,
            "symbol": item.symbol,
            "name": item.name,
            "market": item.market,
            "added_at": item.added_at.isoformat() if item.added_at else None,
        }
        for item in items
    ]


@router.post("/watchlist", status_code=201)
async def add_watchlist(body: WatchlistAddRequest):
    """添加自选股"""
    item = await crud.add_watchlist(body.symbol, body.name, body.market)
    return {
        "id": item.id,
        "symbol": item.symbol,
        "name": item.name,
        "market": item.market,
        "added_at": item.added_at.isoformat() if item.added_at else None,
    }


@router.delete("/watchlist/{symbol}", status_code=204)
async def remove_watchlist(symbol: str):
    """删除自选股"""
    success = await crud.remove_watchlist(symbol)
    if not success:
        raise HTTPException(status_code=404, detail=f"股票 {symbol} 不在自选股列表中")


@router.get("/search")
async def search_stocks(keyword: str = Query(..., description="搜索关键词")):
    """搜索股票"""
    try:
        from app.core.data_collector import DataCollector

        dc = DataCollector()
        df = await asyncio.to_thread(dc.fetch_stock_code_name_list)
        if df is None or df.empty:
            return []

        code_col = df["code"].astype(str)
        name_col = df["name"].astype(str)
        mask = (
            code_col.str.contains(keyword, case=False, na=False)
            | name_col.str.contains(keyword, case=False, na=False)
        )
        filtered = df[mask].head(50)
        return filtered.to_dict(orient="records")  # type: ignore[call-overload]
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"网络错误: {e}")
    except Exception as e:
        logger.error(f"搜索股票失败: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")


@router.get("/batch/realtime")
async def get_batch_realtime(symbols: str = Query(..., description="逗号分隔的股票代码列表")):
    """批量获取实时行情"""
    try:
        from app.core.data_collector import DataCollector

        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        if not symbol_list:
            return []

        dc = DataCollector()
        df = await asyncio.to_thread(dc.fetch_realtime_quotes, symbol_list)
        if df is None or df.empty:
            return []
        records = df.to_dict(orient="records")
        for record in records:
            for k, v in record.items():
                try:
                    import json
                    json.dumps(v)
                except (TypeError, ValueError):
                    record[k] = str(v)
        return records
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"网络错误: {e}")
    except Exception as e:
        logger.error(f"批量获取实时行情失败: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")


@router.get("/{symbol}/history")
async def get_stock_history(
    symbol: str,
    period: str = Query("daily", description="数据周期: daily/weekly/monthly"),
    start_date: Optional[str] = Query(None, description="开始日期 YYYYMMDD"),
    end_date: Optional[str] = Query(None, description="结束日期 YYYYMMDD"),
    adjust: str = Query("qfq", description="复权类型: qfq/hfq/none"),
):
    """获取K线历史数据（含技术指标）"""
    try:
        from app.core.data_collector import DataCollector
        from app.core.indicator_engine import IndicatorEngine

        dc = DataCollector()

        df_full = await asyncio.to_thread(
            dc.fetch_stock_history, symbol, period, None, end_date, adjust
        )
        if df_full is None or df_full.empty:
            return []

        ie = IndicatorEngine()
        df_full = await asyncio.to_thread(ie.calculate_all, df_full)

        signal_val = ie.generate_signal(df_full)
        df_full["signal"] = 0
        if signal_val == "BUY":
            df_full.iloc[-1, df_full.columns.get_loc("signal")] = 1
        elif signal_val == "SELL":
            df_full.iloc[-1, df_full.columns.get_loc("signal")] = -1

        _EN_TO_CN = {"open": "开盘", "close": "收盘", "high": "最高", "low": "最低", "volume": "成交量"}
        df_full = df_full.rename(columns={k: v for k, v in _EN_TO_CN.items() if k in df_full.columns})

        if start_date and "日期" in df_full.columns:
            df_full["日期"] = pd.to_datetime(df_full["日期"]).dt.strftime("%Y-%m-%d")
            df_full = df_full[df_full["日期"] >= start_date]

        df_full = df_full.where(df_full.notna(), None)
        for col in df_full.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
            df_full[col] = df_full[col].astype(str)
        return df_full.to_dict(orient='records')
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"网络错误: {e}")
    except Exception as e:
        logger.error(f"获取K线数据失败 {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")


@router.get("/{symbol}/realtime")
async def get_realtime_quote(symbol: str):
    """获取实时行情"""
    try:
        from app.core.data_collector import DataCollector

        dc = DataCollector()
        df = await asyncio.to_thread(dc.fetch_realtime_quotes, [symbol])
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"未找到 {symbol} 的实时行情")
        record = df.iloc[0].to_dict()
        for k, v in record.items():
            try:
                import json
                json.dumps(v)
            except (TypeError, ValueError):
                record[k] = str(v)
        return record
    except HTTPException:
        raise
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"网络错误: {e}")
    except Exception as e:
        logger.error(f"获取实时行情失败 {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")


@router.get("/{symbol}/indicators")
async def get_indicators(symbol: str):
    """获取技术指标（最新一行）"""
    try:
        from app.core.data_collector import DataCollector
        from app.core.indicator_engine import IndicatorEngine

        dc = DataCollector()
        ie = IndicatorEngine()

        df = await asyncio.to_thread(dc.fetch_stock_history, symbol, "daily", None, None, "qfq")
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"未找到 {symbol} 的历史数据")

        df_with_indicators = await asyncio.to_thread(ie.calculate_all, df)
        if df_with_indicators is None or df_with_indicators.empty:
            raise HTTPException(status_code=404, detail=f"指标计算失败")

        last_row = df_with_indicators.iloc[-1].to_dict()
        # 将不可序列化的类型转为字符串
        for k, v in last_row.items():
            try:
                import json
                json.dumps(v)
            except (TypeError, ValueError):
                last_row[k] = str(v)
        return last_row
    except HTTPException:
        raise
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"网络错误: {e}")
    except Exception as e:
        logger.error(f"获取技术指标失败 {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")


@router.get("/{symbol}/fund-flow")
async def get_fund_flow(symbol: str):
    """获取资金流数据"""
    try:
        from app.core.data_collector import DataCollector

        dc = DataCollector()
        df = await asyncio.to_thread(dc.fetch_fund_flow, symbol)
        if df is None or df.empty:
            return []
        for col in df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
            df[col] = df[col].astype(str)
        return df.to_dict(orient='records')
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"网络错误: {e}")
    except Exception as e:
        logger.error(f"获取资金流失败 {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select

from app.database.models import (
    Alert,
    FundFlow,
    NewsArticle,
    Prediction,
    StockDaily,
    StockRealtime,
    WatchList,
)
from app.database.session import async_session


async def add_watchlist(symbol: str, name: str, market: str = "A") -> WatchList:
    async with async_session() as session:
        existing = await session.scalar(select(WatchList).where(WatchList.symbol == symbol))
        if existing:
            return existing
        item = WatchList(symbol=symbol, name=name, market=market)
        session.add(item)
        await session.commit()
        await session.refresh(item)
        return item


async def get_watchlist() -> list[WatchList]:
    async with async_session() as session:
        result = await session.execute(select(WatchList).order_by(WatchList.added_at))
        return list(result.scalars().all())


async def remove_watchlist(symbol: str) -> bool:
    async with async_session() as session:
        item = await session.scalar(select(WatchList).where(WatchList.symbol == symbol))
        if not item:
            return False
        await session.delete(item)
        await session.commit()
        return True


async def upsert_stock_daily(records: list[dict]) -> int:
    inserted = 0
    async with async_session() as session:
        for rec in records:
            existing = await session.scalar(
                select(StockDaily).where(
                    StockDaily.symbol == rec["symbol"],
                    StockDaily.date == rec["date"],
                    StockDaily.adjust_type == rec.get("adjust_type", "qfq"),
                )
            )
            if not existing:
                session.add(StockDaily(**rec))
                inserted += 1
        await session.commit()
    return inserted


async def get_stock_daily(
    symbol: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> list[StockDaily]:
    async with async_session() as session:
        stmt = select(StockDaily).where(StockDaily.symbol == symbol)
        if start_date:
            stmt = stmt.where(StockDaily.date >= start_date)
        if end_date:
            stmt = stmt.where(StockDaily.date <= end_date)
        stmt = stmt.order_by(StockDaily.date)
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def save_realtime(records: list[dict]) -> int:
    async with async_session() as session:
        for rec in records:
            session.add(StockRealtime(**rec))
        await session.commit()
    return len(records)


async def get_latest_realtime(symbol: str) -> Optional[StockRealtime]:
    async with async_session() as session:
        return await session.scalar(
            select(StockRealtime)
            .where(StockRealtime.symbol == symbol)
            .order_by(StockRealtime.timestamp.desc())
            .limit(1)
        )


async def save_fund_flow(records: list[dict]) -> int:
    async with async_session() as session:
        for rec in records:
            session.add(FundFlow(**rec))
        await session.commit()
    return len(records)


async def get_fund_flow(symbol: str, days: int = 30) -> list[FundFlow]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    async with async_session() as session:
        result = await session.execute(
            select(FundFlow)
            .where(FundFlow.symbol == symbol, FundFlow.date >= cutoff)
            .order_by(FundFlow.date)
        )
        return list(result.scalars().all())


async def save_news(records: list[dict]) -> int:
    inserted = 0
    async with async_session() as session:
        for rec in records:
            url = (rec.get("url") or "").strip()
            if url:
                existing = await session.scalar(
                    select(NewsArticle).where(NewsArticle.url == url)
                )
            else:
                existing = await session.scalar(
                    select(NewsArticle).where(
                        NewsArticle.title == rec["title"],
                        NewsArticle.publish_time == rec["publish_time"],
                    )
                )
            if not existing:
                session.add(NewsArticle(**rec))
                inserted += 1
        await session.commit()
    return inserted


async def get_news(symbol: str, days: int = 7) -> list[NewsArticle]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    async with async_session() as session:
        result = await session.execute(
            select(NewsArticle)
            .where(NewsArticle.symbol == symbol, NewsArticle.publish_time >= cutoff)
            .order_by(NewsArticle.publish_time.desc())
        )
        return list(result.scalars().all())


async def save_prediction(record: dict) -> Prediction:
    async with async_session() as session:
        pred = Prediction(**record)
        session.add(pred)
        await session.commit()
        await session.refresh(pred)
        return pred


async def get_predictions(symbol: str, days: int = 30) -> list[Prediction]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    async with async_session() as session:
        result = await session.execute(
            select(Prediction)
            .where(Prediction.symbol == symbol, Prediction.date >= cutoff)
            .order_by(Prediction.date.desc())
        )
        return list(result.scalars().all())


async def save_alert(record: dict) -> Alert:
    async with async_session() as session:
        alert = Alert(**record)
        session.add(alert)
        await session.commit()
        await session.refresh(alert)
        return alert


async def get_alerts(
    symbol: Optional[str] = None,
    is_read: Optional[bool] = None,
) -> list[Alert]:
    async with async_session() as session:
        stmt = select(Alert)
        if symbol is not None:
            stmt = stmt.where(Alert.symbol == symbol)
        if is_read is not None:
            stmt = stmt.where(Alert.is_read == is_read)
        stmt = stmt.order_by(Alert.created_at.desc())
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def mark_alert_read(alert_id: int) -> bool:
    async with async_session() as session:
        alert = await session.get(Alert, alert_id)
        if not alert:
            return False
        alert.is_read = True
        await session.commit()
        return True

"""
测试数据库 CRUD 操作
使用内存 SQLite 数据库（aiosqlite）验证异步 CRUD 函数正确性。

注意：crud.py 使用 `from app.database.session import async_session` 直接绑定，
因此 monkey-patch 必须针对 crud 模块内的 async_session 名称，而非 session 模块。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime, timezone
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.database.session import Base
from app.database import crud


# ── 测试专用 Fixture ──────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def db():
    """
    创建内存 SQLite 数据库，直接 monkey-patch crud 模块使用的 async_session，
    测试结束后恢复原始 session。每次测试都创建新的内存数据库保证隔离。
    """
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    test_session = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # crud.py 直接绑定了 async_session，需 patch crud 模块内的名称
    original_session = crud.async_session
    crud.async_session = test_session

    yield test_session

    crud.async_session = original_session
    await test_engine.dispose()


# ── WatchList 测试 ────────────────────────────────────────────────────────────

class TestWatchListCRUD:
    """WatchList CRUD 操作测试。"""

    async def test_add_watchlist(self, db):
        """add_watchlist 应成功插入一条记录并返回 WatchList 对象。"""
        item = await crud.add_watchlist("000001", "平安银行", "A")
        assert item.symbol == "000001"
        assert item.name == "平安银行"
        assert item.market == "A"
        assert item.id is not None

    async def test_add_watchlist_duplicate_returns_existing(self, db):
        """重复添加同一股票应返回已存在的记录（不创建重复）。"""
        item1 = await crud.add_watchlist("000001", "平安银行", "A")
        item2 = await crud.add_watchlist("000001", "平安银行（重复）", "A")
        assert item1.id == item2.id

    async def test_get_watchlist_empty(self, db):
        """初始状态下 get_watchlist 应返回空列表。"""
        items = await crud.get_watchlist()
        assert items == []

    async def test_get_watchlist_multiple(self, db):
        """添加多个股票后 get_watchlist 应返回所有记录。"""
        await crud.add_watchlist("000001", "平安银行", "A")
        await crud.add_watchlist("600036", "招商银行", "A")
        items = await crud.get_watchlist()
        assert len(items) == 2
        symbols = {item.symbol for item in items}
        assert symbols == {"000001", "600036"}

    async def test_remove_watchlist_existing(self, db):
        """remove_watchlist 删除已存在的股票应返回 True。"""
        await crud.add_watchlist("000001", "平安银行", "A")
        success = await crud.remove_watchlist("000001")
        assert success is True

    async def test_remove_watchlist_not_existing(self, db):
        """remove_watchlist 删除不存在的股票应返回 False。"""
        success = await crud.remove_watchlist("999999")
        assert success is False

    async def test_remove_watchlist_actually_removes(self, db):
        """remove_watchlist 后 get_watchlist 不应再包含该股票。"""
        await crud.add_watchlist("000001", "平安银行", "A")
        await crud.remove_watchlist("000001")
        items = await crud.get_watchlist()
        assert all(item.symbol != "000001" for item in items)


# ── StockDaily 测试 ───────────────────────────────────────────────────────────

class TestStockDailyCRUD:
    """StockDaily CRUD 操作测试。"""

    def _make_record(self, symbol: str = "000001", day: int = 1) -> dict:
        return {
            "symbol": symbol,
            "date": date(2024, 1, day),
            "open": 10.0 + day * 0.1,
            "high": 11.0 + day * 0.1,
            "low": 9.5 + day * 0.1,
            "close": 10.5 + day * 0.1,
            "volume": 100000 + day * 1000,
            "turnover": 1050000.0 + day * 10000,
            "adjust_type": "qfq",
        }

    async def test_upsert_stock_daily_inserts_new_records(self, db):
        """upsert_stock_daily 应成功插入新记录并返回插入条数。"""
        records = [self._make_record(day=i) for i in range(1, 4)]
        count = await crud.upsert_stock_daily(records)
        assert count == 3

    async def test_upsert_stock_daily_skips_duplicates(self, db):
        """重复 upsert 同一日期/股票/复权类型时应跳过（返回 0）。"""
        records = [self._make_record(day=1)]
        await crud.upsert_stock_daily(records)
        count2 = await crud.upsert_stock_daily(records)
        assert count2 == 0

    async def test_upsert_stock_daily_partial_new(self, db):
        """部分已存在、部分新记录时应只计入新插入的数量。"""
        records_day1 = [self._make_record(day=1)]
        await crud.upsert_stock_daily(records_day1)
        # 第二次插入：day1 已存在，day2 是新的
        records_day1_2 = [self._make_record(day=1), self._make_record(day=2)]
        count = await crud.upsert_stock_daily(records_day1_2)
        assert count == 1

    async def test_get_stock_daily_returns_records(self, db):
        """get_stock_daily 应返回已插入的历史记录。"""
        records = [self._make_record(day=i) for i in range(1, 4)]
        await crud.upsert_stock_daily(records)
        result = await crud.get_stock_daily("000001")
        assert len(result) == 3

    async def test_get_stock_daily_empty_when_no_records(self, db):
        """无记录时 get_stock_daily 应返回空列表。"""
        result = await crud.get_stock_daily("999999")
        assert result == []


# ── Alert 测试 ────────────────────────────────────────────────────────────────

class TestAlertCRUD:
    """Alert CRUD 操作测试。"""

    def _make_alert_record(self, symbol: str = "000001", alert_type: str = "price_change") -> dict:
        return {
            "symbol": symbol,
            "alert_type": alert_type,
            "trigger_value": 6.5,
            "threshold": 5.0,
            "message": f"【测试告警】{symbol} 触发 {alert_type}",
            "created_at": datetime.now(timezone.utc),
            "is_read": False,
        }

    async def test_save_alert_returns_alert_object(self, db):
        """save_alert 应返回含 id 的 Alert 对象。"""
        record = self._make_alert_record()
        alert = await crud.save_alert(record)
        assert alert.id is not None
        assert alert.symbol == "000001"
        assert alert.is_read is False

    async def test_get_alerts_returns_all(self, db):
        """get_alerts 应返回所有已保存的告警。"""
        await crud.save_alert(self._make_alert_record("000001", "price_change"))
        await crud.save_alert(self._make_alert_record("600036", "volume_spike"))
        alerts = await crud.get_alerts()
        assert len(alerts) == 2

    async def test_get_alerts_filter_by_symbol(self, db):
        """get_alerts 按股票过滤时只返回该股票的告警。"""
        await crud.save_alert(self._make_alert_record("000001", "price_change"))
        await crud.save_alert(self._make_alert_record("600036", "volume_spike"))
        alerts = await crud.get_alerts(symbol="000001")
        assert len(alerts) == 1
        assert alerts[0].symbol == "000001"

    async def test_get_alerts_filter_by_is_read(self, db):
        """get_alerts 按 is_read 过滤时只返回对应状态的告警。"""
        await crud.save_alert(self._make_alert_record("000001"))
        alerts_unread = await crud.get_alerts(is_read=False)
        assert len(alerts_unread) == 1
        alerts_read = await crud.get_alerts(is_read=True)
        assert len(alerts_read) == 0

    async def test_mark_alert_read_returns_true(self, db):
        """mark_alert_read 标记已存在的告警应返回 True。"""
        alert = await crud.save_alert(self._make_alert_record())
        success = await crud.mark_alert_read(alert.id)
        assert success is True

    async def test_mark_alert_read_updates_flag(self, db):
        """mark_alert_read 后 is_read 应变为 True。"""
        alert = await crud.save_alert(self._make_alert_record())
        await crud.mark_alert_read(alert.id)
        alerts = await crud.get_alerts(is_read=True)
        assert len(alerts) == 1
        assert alerts[0].is_read is True

    async def test_mark_alert_read_not_found(self, db):
        """mark_alert_read 对不存在的 id 应返回 False。"""
        success = await crud.mark_alert_read(99999)
        assert success is False

    async def test_get_alerts_empty_initially(self, db):
        """初始状态下 get_alerts 应返回空列表。"""
        alerts = await crud.get_alerts()
        assert alerts == []

"""
测试 FastAPI 端点
使用 httpx AsyncClient + ASGITransport 对各 API 端点进行集成测试。
数据库层被替换为内存 SQLite，外部 AKShare/模型调用被 mock。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.database.session import Base
from app.database import crud as crud_module


# ── 共享 Fixture ──────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def api_client():
    """创建内存 SQLite 并启动 FastAPI 测试客户端，每次 yield 后销毁。"""
    # 创建内存数据库
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    test_session = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    original_session = crud_module.async_session
    crud_module.async_session = test_session

    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client

    crud_module.async_session = original_session
    await test_engine.dispose()


# ── 根路由测试 ────────────────────────────────────────────────────────────────

class TestRootEndpoint:
    """根路由端点测试。"""

    async def test_root_returns_200(self, api_client):
        """GET / 应返回 200。"""
        resp = await api_client.get("/")
        assert resp.status_code == 200

    async def test_root_returns_status_ok(self, api_client):
        """GET / 应返回 {"status": "ok", ...}。"""
        resp = await api_client.get("/")
        data = resp.json()
        assert data["status"] == "ok"


# ── 自选股 API 测试 ────────────────────────────────────────────────────────────

class TestStocksAPI:
    """自选股 CRUD API 端点测试。"""

    async def test_get_watchlist_empty(self, api_client):
        """GET /api/stocks/ 初始状态应返回 200 和空列表。"""
        resp = await api_client.get("/api/stocks/")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_add_watchlist_returns_201(self, api_client):
        """POST /api/stocks/watchlist 应返回 201。"""
        resp = await api_client.post(
            "/api/stocks/watchlist",
            json={"symbol": "000001", "name": "平安银行", "market": "A"},
        )
        assert resp.status_code == 201

    async def test_add_watchlist_response_body(self, api_client):
        """POST /api/stocks/watchlist 返回体应包含 symbol/name/market。"""
        resp = await api_client.post(
            "/api/stocks/watchlist",
            json={"symbol": "000001", "name": "平安银行", "market": "A"},
        )
        data = resp.json()
        assert data["symbol"] == "000001"
        assert data["name"] == "平安银行"
        assert data["market"] == "A"
        assert data["id"] is not None

    async def test_get_watchlist_after_add(self, api_client):
        """添加自选股后 GET /api/stocks/ 应返回包含该股票的列表。"""
        await api_client.post(
            "/api/stocks/watchlist",
            json={"symbol": "000001", "name": "平安银行", "market": "A"},
        )
        resp = await api_client.get("/api/stocks/")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1
        assert items[0]["symbol"] == "000001"

    async def test_delete_watchlist_returns_204(self, api_client):
        """DELETE /api/stocks/watchlist/{symbol} 应返回 204。"""
        await api_client.post(
            "/api/stocks/watchlist",
            json={"symbol": "000001", "name": "平安银行", "market": "A"},
        )
        resp = await api_client.delete("/api/stocks/watchlist/000001")
        assert resp.status_code == 204

    async def test_delete_watchlist_not_found_returns_404(self, api_client):
        """删除不存在的自选股应返回 404。"""
        resp = await api_client.delete("/api/stocks/watchlist/999999")
        assert resp.status_code == 404

    async def test_add_duplicate_watchlist(self, api_client):
        """重复添加同一股票应返回 201（幂等）。"""
        await api_client.post(
            "/api/stocks/watchlist",
            json={"symbol": "000001", "name": "平安银行", "market": "A"},
        )
        resp = await api_client.post(
            "/api/stocks/watchlist",
            json={"symbol": "000001", "name": "平安银行", "market": "A"},
        )
        assert resp.status_code == 201


# ── 告警 API 测试 ──────────────────────────────────────────────────────────────

class TestAlertsAPI:
    """告警 API 端点测试。"""

    async def test_get_alerts_empty(self, api_client):
        """GET /api/alerts/ 初始状态应返回 200 和空列表。"""
        resp = await api_client.get("/api/alerts/")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_get_alerts_after_insert(self, api_client):
        """插入告警后 GET /api/alerts/ 应返回包含该告警的列表。"""
        from app.database import crud
        await crud.save_alert({
            "symbol": "000001",
            "alert_type": "price_change",
            "trigger_value": 6.5,
            "threshold": 5.0,
            "message": "测试告警",
            "created_at": datetime.now(timezone.utc),
            "is_read": False,
        })
        resp = await api_client.get("/api/alerts/")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1
        assert items[0]["symbol"] == "000001"

    async def test_get_alerts_filter_by_symbol(self, api_client):
        """GET /api/alerts/?symbol=000001 应只返回该股票的告警。"""
        from app.database import crud
        await crud.save_alert({
            "symbol": "000001",
            "alert_type": "price_change",
            "trigger_value": 6.5,
            "threshold": 5.0,
            "message": "告警1",
            "created_at": datetime.now(timezone.utc),
            "is_read": False,
        })
        await crud.save_alert({
            "symbol": "600036",
            "alert_type": "price_change",
            "trigger_value": 7.0,
            "threshold": 5.0,
            "message": "告警2",
            "created_at": datetime.now(timezone.utc),
            "is_read": False,
        })
        resp = await api_client.get("/api/alerts/?symbol=000001")
        assert resp.status_code == 200
        items = resp.json()
        assert all(item["symbol"] == "000001" for item in items)

    async def test_mark_alert_read_returns_200(self, api_client):
        """PUT /api/alerts/{id}/read 对已存在的告警应返回 200。"""
        from app.database import crud
        alert = await crud.save_alert({
            "symbol": "000001",
            "alert_type": "price_change",
            "trigger_value": 6.5,
            "threshold": 5.0,
            "message": "测试告警",
            "created_at": datetime.now(timezone.utc),
            "is_read": False,
        })
        resp = await api_client.put(f"/api/alerts/{alert.id}/read")
        assert resp.status_code == 200

    async def test_mark_alert_read_not_found(self, api_client):
        """PUT /api/alerts/99999/read 对不存在的告警应返回 404。"""
        resp = await api_client.put("/api/alerts/99999/read")
        assert resp.status_code == 404

    async def test_update_alert_config(self, api_client):
        """PUT /api/alerts/config 应返回 200 和配置信息。"""
        resp = await api_client.put(
            "/api/alerts/config",
            json={
                "price_change_threshold": 5.0,
                "volume_spike_threshold": 3.0,
                "sentiment_threshold": 0.5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


# ── 预测 API 测试 ──────────────────────────────────────────────────────────────

class TestPredictionsAPI:
    """预测 API 端点测试。"""

    async def test_get_prediction_returns_non_500(self, api_client):
        """GET /api/predictions/{symbol} 应返回非 500 状态码（无模型时返回默认结果）。"""
        # 无模型文件，PredictionEngine 应返回默认结果
        with patch("app.core.prediction_engine.PredictionEngine.predict") as mock_pred:
            mock_pred.return_value = {
                "direction": "FLAT",
                "probability": 0.5,
                "price_range": [0.0, 0.0],
                "predicted_return": 0.0,
                "trend_rating": "中性",
            }
            resp = await api_client.get("/api/predictions/000001")
        assert resp.status_code != 500

    async def test_get_prediction_response_has_required_keys(self, api_client):
        """GET /api/predictions/{symbol} 响应应包含 direction/probability 等键。"""
        with patch("app.core.prediction_engine.PredictionEngine.predict") as mock_pred:
            mock_pred.return_value = {
                "direction": "UP",
                "probability": 0.72,
                "price_range": [0.01, 0.03],
                "predicted_return": 0.02,
                "trend_rating": "看涨",
            }
            resp = await api_client.get("/api/predictions/000001")
        assert resp.status_code == 200
        data = resp.json()
        assert "direction" in data
        assert "probability" in data

    async def test_get_prediction_history_empty(self, api_client):
        """GET /api/predictions/{symbol}/history 无数据时应返回空列表。"""
        resp = await api_client.get("/api/predictions/000001/history")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_retrain_unknown_model_type(self, api_client):
        """POST /api/predictions/retrain 传入未知 model_type 应返回 400。"""
        resp = await api_client.post(
            "/api/predictions/retrain",
            json={"symbol": "000001", "model_type": "unknown_model"},
        )
        assert resp.status_code == 400

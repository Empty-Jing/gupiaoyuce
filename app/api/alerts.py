import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.database import crud

logger = logging.getLogger(__name__)

router = APIRouter()


class AlertConfigRequest(BaseModel):
    price_change_threshold: Optional[float] = None
    volume_spike_threshold: Optional[float] = None
    sentiment_threshold: Optional[float] = None


@router.get("/")
async def get_alerts(
    symbol: Optional[str] = Query(None, description="股票代码（可选过滤）"),
    is_read: Optional[bool] = Query(None, description="是否已读（可选过滤）"),
):
    """获取告警列表"""
    alerts = await crud.get_alerts(symbol=symbol, is_read=is_read)
    return [
        {
            "id": a.id,
            "symbol": a.symbol,
            "alert_type": a.alert_type,
            "trigger_value": a.trigger_value,
            "threshold": a.threshold,
            "message": a.message,
            "created_at": a.created_at.isoformat() if a.created_at else None,
            "is_read": a.is_read,
        }
        for a in alerts
    ]


@router.put("/{alert_id}/read")
async def mark_alert_read(alert_id: int):
    """标记告警为已读"""
    success = await crud.mark_alert_read(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"告警 ID {alert_id} 不存在")
    return {"status": "ok", "alert_id": alert_id}


@router.put("/config")
async def update_alert_config(body: AlertConfigRequest):
    """更新告警配置（Phase 1 简化实现：返回当前配置）"""
    # Phase 1: 仅返回当前接收到的配置，不做持久化
    config = {
        "price_change_threshold": body.price_change_threshold,
        "volume_spike_threshold": body.volume_spike_threshold,
        "sentiment_threshold": body.sentiment_threshold,
    }
    return {"status": "ok", "config": config}

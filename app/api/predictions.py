import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.database import crud

logger = logging.getLogger(__name__)

router = APIRouter()


class RetrainRequest(BaseModel):
    symbol: str
    model_type: str  # 'lstm' 或 'xgboost'


@router.get("/{symbol}")
async def get_prediction(symbol: str):
    try:
        from app.core.prediction_engine import PredictionEngine

        pe = PredictionEngine()
        result = await asyncio.to_thread(pe.predict, symbol)
        return result
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"网络错误: {e}")
    except Exception as e:
        logger.error(f"获取预测结果失败 {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")


@router.get("/{symbol}/history")
async def get_prediction_history(symbol: str):
    preds = await crud.get_predictions(symbol)
    return [
        {
            "id": p.id,
            "symbol": p.symbol,
            "date": p.date.isoformat() if p.date else None,
            "model_type": p.model_type,
            "direction": p.direction,
            "probability": p.probability,
            "price_low": p.price_low,
            "price_high": p.price_high,
            "predicted_return": p.predicted_return,
        }
        for p in preds
    ]


@router.post("/retrain", status_code=202)
async def retrain(body: RetrainRequest):
    if body.model_type not in ("lstm", "xgboost"):
        raise HTTPException(status_code=400, detail="model_type must be 'lstm' or 'xgboost'")

    from app.core.train_manager import submit_job

    job_id = submit_job(body.symbol, body.model_type)

    return {
        "status": "accepted",
        "job_id": job_id,
        "symbol": body.symbol,
        "model_type": body.model_type,
    }


@router.get("/retrain/jobs/list")
async def retrain_list(symbol: str = "", limit: int = 20):
    from app.core.train_manager import list_jobs

    return list_jobs(symbol=symbol or None, limit=limit)


@router.get("/retrain/{job_id}/status")
async def retrain_status(job_id: str):
    from app.core.train_manager import get_job_status

    job = get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"训练任务不存在: {job_id}")

    return job

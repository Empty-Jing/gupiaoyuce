"""
训练任务管理器
提供训练任务的提交、状态查询、进度回调机制
使用内存 dict 存储任务状态，线程池执行训练
"""

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# 全局训练任务存储（进程内内存）
_TRAIN_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()

# 线程池（最多同时训练 2 个模型，避免 CPU 过载）
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="train")


def submit_job(symbol: str, model_type: str) -> str:
    """
    提交训练任务到线程池。

    参数:
        symbol: 股票代码
        model_type: 'lstm' 或 'xgboost'

    返回:
        job_id: 任务唯一标识
    """
    job_id = uuid.uuid4().hex[:12]

    job = {
        "job_id": job_id,
        "symbol": symbol,
        "model_type": model_type,
        "status": "pending",  # pending -> running -> success / failed
        "progress": 0,
        "message": "等待执行",
        "result": None,
        "error": None,
        "started_at": None,
        "finished_at": None,
        "created_at": datetime.now().isoformat(),
    }

    with _JOBS_LOCK:
        _TRAIN_JOBS[job_id] = job

    # 提交到线程池
    _EXECUTOR.submit(_run_training, job_id, symbol, model_type)
    logger.info(f"训练任务已提交: job_id={job_id}, symbol={symbol}, model_type={model_type}")

    return job_id


def get_job_status(job_id: str) -> Optional[dict]:
    """
    查询训练任务状态。

    返回:
        任务状态 dict，不存在返回 None
    """
    with _JOBS_LOCK:
        job = _TRAIN_JOBS.get(job_id)
        if job is None:
            return None
        return dict(job)  # 返回副本


def list_jobs(symbol: Optional[str] = None, limit: int = 20) -> list[dict]:
    """
    列出训练任务（按创建时间倒序）。

    参数:
        symbol: 可选，按股票代码过滤
        limit: 最多返回条数

    返回:
        任务列表
    """
    with _JOBS_LOCK:
        jobs = list(_TRAIN_JOBS.values())

    if symbol:
        jobs = [j for j in jobs if j["symbol"] == symbol]

    # 按创建时间倒序
    jobs.sort(key=lambda j: j["created_at"], reverse=True)

    return [dict(j) for j in jobs[:limit]]


def _make_progress_callback(job_id: str) -> Callable:
    """
    创建进度回调函数，供 train_lstm / train_xgboost 在训练过程中调用。

    回调签名: callback(progress: int, message: str)
        progress: 0-100 整数
        message: 当前阶段描述
    """

    def callback(progress: int, message: str = "") -> None:
        with _JOBS_LOCK:
            job = _TRAIN_JOBS.get(job_id)
            if job is None:
                return
            job["progress"] = min(max(progress, 0), 100)
            if message:
                job["message"] = message

    return callback


def _run_training(job_id: str, symbol: str, model_type: str) -> None:
    """
    在线程池中执行训练任务（内部方法）。
    """
    # 标记 running
    with _JOBS_LOCK:
        job = _TRAIN_JOBS.get(job_id)
        if job is None:
            return
        job["status"] = "running"
        job["progress"] = 0
        job["message"] = "正在准备训练数据"
        job["started_at"] = datetime.now().isoformat()

    callback = _make_progress_callback(job_id)

    try:
        from app.models.train import train_lstm, train_xgboost

        if model_type == "lstm":
            result = train_lstm(symbol, progress_callback=callback)
        elif model_type == "xgboost":
            result = train_xgboost(symbol, progress_callback=callback)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 标记成功
        with _JOBS_LOCK:
            job = _TRAIN_JOBS.get(job_id)
            if job:
                job["status"] = "success"
                job["progress"] = 100
                job["message"] = "训练完成"
                job["result"] = result
                job["finished_at"] = datetime.now().isoformat()

        logger.info(f"训练任务完成: job_id={job_id}, result={result}")

    except Exception as e:
        # 标记失败
        with _JOBS_LOCK:
            job = _TRAIN_JOBS.get(job_id)
            if job:
                job["status"] = "failed"
                job["message"] = f"训练失败: {str(e)}"
                job["error"] = str(e)
                job["finished_at"] = datetime.now().isoformat()

        logger.error(f"训练任务失败: job_id={job_id}, error={e}")

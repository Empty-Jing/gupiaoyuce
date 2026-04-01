import asyncio
import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.session import init_db
from app.scheduler.tasks import register_all_tasks, task_retrain

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("数据库初始化完成")

    scheduler = AsyncIOScheduler()
    register_all_tasks(scheduler)
    scheduler.start()
    logger.info("调度器已启动")

    asyncio.create_task(task_retrain())
    logger.info("启动时全量训练已触发（后台运行）")

    yield

    scheduler.shutdown(wait=False)
    logger.info("调度器已关闭")


app = FastAPI(title="股票量化监测系统", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
from app.api import alerts, news, predictions, stocks  # noqa: E402

app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["alerts"])
app.include_router(news.router, prefix="/api/news", tags=["news"])


@app.get("/")
async def root():
    return {"status": "ok", "service": "股票量化监测系统"}

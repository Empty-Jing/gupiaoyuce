import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger = logging.getLogger(__name__)


def register_all_tasks(scheduler: AsyncIOScheduler) -> None:
    """注册所有定时任务"""

    # 交易时段实时行情 (每分钟, 9:30-11:30 + 13:00-15:00, 周一到周五)
    scheduler.add_job(task_realtime_fetch, 'cron', day_of_week='mon-fri',
                      hour='9-11,13-14', minute='*', id='realtime_fetch',
                      misfire_grace_time=60)

    # 新闻拉取 (每30分钟)
    scheduler.add_job(task_news_fetch, 'interval', minutes=30, id='news_fetch',
                      misfire_grace_time=300)

    # 日终更新 (每日15:30, 周一到周五)
    scheduler.add_job(task_daily_update, 'cron', day_of_week='mon-fri',
                      hour=15, minute=30, id='daily_update',
                      misfire_grace_time=600)

    # 预测推断 (每日16:00, 周一到周五)
    scheduler.add_job(task_prediction, 'cron', day_of_week='mon-fri',
                      hour=16, minute=0, id='prediction',
                      misfire_grace_time=600)

    # 模型重训练 (每12小时)
    scheduler.add_job(task_retrain, 'interval', hours=12, id='retrain',
                      misfire_grace_time=3600)


async def task_realtime_fetch():
    """拉取自选股实时行情"""
    try:
        from app.config import settings
        from app.core.data_collector import DataCollector
        from app.database.crud import save_realtime
        import datetime

        dc = DataCollector()
        for symbol in settings.WATCH_LIST:
            df = dc.fetch_realtime_quotes([symbol])
            if df is not None and not df.empty:
                records = df.to_dict(orient='records')
                await save_realtime([{
                    'symbol': symbol,
                    'timestamp': records[0].get('timestamp') or datetime.datetime.now(),
                    'price': records[0].get('price', 0),
                    'change_pct': records[0].get('change_pct', 0),
                    'volume': int(records[0].get('volume', 0)),
                    'amount': float(records[0].get('amount', 0)),
                }])
        logger.info("实时行情更新完成")
    except Exception as e:
        logger.error(f"实时行情拉取失败: {e}")


async def task_news_fetch():
    """拉取新闻 + 情感分析"""
    try:
        import datetime
        from app.config import settings
        from app.core.data_collector import DataCollector
        from app.core.sentiment_analyzer import SentimentAnalyzer
        from app.database.crud import save_news

        dc = DataCollector()
        sa = SentimentAnalyzer()

        for symbol in settings.WATCH_LIST:
            df = dc.fetch_news(symbol)
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    title = str(row.get('新闻标题', '') or row.get('title', '') or '')
                    result = sa.analyze(title)
                    # 解析发布时间为 datetime 对象（akshare 返回字符串格式）
                    raw_time = row.get('发布时间') or row.get('publish_time')
                    if isinstance(raw_time, str):
                        try:
                            pub_time = datetime.datetime.strptime(raw_time, '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            pub_time = datetime.datetime.now()
                    elif isinstance(raw_time, datetime.datetime):
                        pub_time = raw_time
                    else:
                        pub_time = datetime.datetime.now()
                    await save_news([{
                        'symbol': symbol,
                        'title': title,
                        'content': str(row.get('新闻内容', '') or row.get('content', '') or ''),
                        'publish_time': pub_time,
                        'source': str(row.get('文章来源', '') or row.get('source', '') or ''),
                        'url': str(row.get('新闻链接', '') or row.get('url', '') or ''),
                        'sentiment_label': result['label'],
                        'sentiment_score': result['score'],
                    }])
        logger.info("新闻拉取 + 情感分析完成")
    except Exception as e:
        logger.error(f"新闻拉取失败: {e}")


async def task_daily_update(symbol: str = None):
    """日终更新: 拉取日K线 + 计算指标 + 存储"""
    try:
        from app.config import settings
        from app.core.data_collector import DataCollector

        dc = DataCollector()
        symbols = [symbol] if symbol else settings.WATCH_LIST

        for sym in symbols:
            dc.save_history_to_db(sym)
        logger.info(f"日终更新完成: {symbols}")
    except Exception as e:
        logger.error(f"日终更新失败: {e}")


async def task_prediction():
    """运行预测模型"""
    try:
        from datetime import date
        from app.config import settings
        from app.core.prediction_engine import PredictionEngine
        from app.database.crud import save_prediction

        pe = PredictionEngine()
        for symbol in settings.WATCH_LIST:
            result = pe.predict(symbol)
            await save_prediction({
                'symbol': symbol,
                'date': date.today(),
                'model_type': 'fusion',
                'direction': result['direction'],
                'probability': result['probability'],
                'price_low': result['price_range'][0] if result.get('price_range') else None,
                'price_high': result['price_range'][1] if result.get('price_range') else None,
                'predicted_return': result.get('predicted_return'),
            })
        logger.info("预测推断完成")
    except Exception as e:
        logger.error(f"预测推断失败: {e}")


async def task_retrain():
    """重训练所有模型"""
    try:
        import asyncio
        from app.config import settings
        from app.models.train import train_lstm, train_xgboost

        symbols = settings.WATCH_LIST
        logger.info(
            f"开始模型重训练: {symbols} | "
            f"数据源: 东方财富(优先)/新浪(降级) → 历史日K线 → 技术指标特征"
        )

        loop = asyncio.get_event_loop()
        for symbol in symbols:
            await loop.run_in_executor(None, train_lstm, symbol)
            await loop.run_in_executor(None, train_xgboost, symbol)
        logger.info(f"模型重训练完成: {symbols}")
    except Exception as e:
        logger.error(f"模型重训练失败: {e}")

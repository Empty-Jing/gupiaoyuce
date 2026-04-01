import logging
from typing import Optional

from fastapi import APIRouter, Query

from app.database import crud

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def get_news(
    symbol: Optional[str] = Query(None, description="股票代码（可选过滤）"),
    days: int = Query(7, description="获取最近N天的新闻"),
):
    """获取新闻列表（支持按股票代码过滤）"""
    if not symbol:
        # 获取所有自选股的新闻
        from app.config import settings
        all_news = []
        for sym in settings.WATCH_LIST:
            articles = await crud.get_news(symbol=sym, days=days)
            all_news.extend(articles)
        # 跨 symbol 去重：同一 url 或同一 title+publish_time 只保留一条
        seen: dict[str, object] = {}
        deduped = []
        for a in all_news:
            url = (getattr(a, "url", "") or "").strip()
            key = url if url else f"{a.title}__{a.publish_time.isoformat() if a.publish_time else ''}"
            if key not in seen:
                seen[key] = True
                deduped.append(a)
        all_news = deduped
        # 按发布时间倒序排列
        all_news.sort(key=lambda a: a.publish_time or "", reverse=True)
    else:
        all_news = await crud.get_news(symbol=symbol, days=days)

    return [
        {
            "id": a.id,
            "symbol": a.symbol,
            "title": a.title,
            "content": a.content,
            "publish_time": a.publish_time.isoformat() if a.publish_time else None,
            "source": a.source,
            "url": getattr(a, "url", None) or "",
            "sentiment_label": a.sentiment_label,
            "sentiment_score": a.sentiment_score,
        }
        for a in all_news
    ]

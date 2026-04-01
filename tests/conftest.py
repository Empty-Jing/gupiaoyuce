"""
共享测试 Fixtures — conftest.py
提供数据库、应用客户端等公共测试夹具。
"""
import sys
import os

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.database.session import Base
from app.database import session as session_module


@pytest_asyncio.fixture
async def test_db_session():
    """
    创建内存 SQLite 数据库 session，用于 CRUD 测试。
    自动 monkey-patch session_module.async_session，测试结束后恢复。
    """
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    test_session = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    # 创建所有表
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 保存原始 session 并替换
    original_session = session_module.async_session
    session_module.async_session = test_session

    yield test_session

    # 恢复原始 session 并释放资源
    session_module.async_session = original_session
    await test_engine.dispose()

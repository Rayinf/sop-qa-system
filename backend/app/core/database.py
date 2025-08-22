from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import redis.asyncio as redis
from typing import AsyncGenerator
import logging
from .config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy 同步引擎
engine = create_engine(
    settings.database_url_sync,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.debug
)

# SQLAlchemy 异步引擎
async_engine = create_async_engine(
    settings.database_url_async,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.debug
)

# 会话工厂
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# 异步会话工厂
AsyncSessionLocal = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# 基础模型类
Base = declarative_base()

# Redis 连接池
redis_pool = None

async def init_redis():
    """初始化Redis连接池"""
    global redis_pool
    try:
        redis_pool = redis.ConnectionPool.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20
        )
        # 测试连接
        redis_client = redis.Redis(connection_pool=redis_pool)
        await redis_client.ping()
        logger.info("Redis连接成功")
    except Exception as e:
        logger.error(f"Redis连接失败: {e}")
        raise

async def get_redis() -> redis.Redis:
    """获取Redis客户端"""
    if redis_pool is None:
        await init_redis()
    return redis.Redis(connection_pool=redis_pool)

def get_redis_client() -> redis.Redis:
    """获取Redis客户端（同步版本）"""
    if redis_pool is None:
        raise RuntimeError("Redis连接池未初始化")
    return redis.Redis(connection_pool=redis_pool)

async def close_redis():
    """关闭Redis连接池"""
    global redis_pool
    if redis_pool:
        await redis_pool.disconnect()
        redis_pool = None
        logger.info("Redis连接池已关闭")

# 数据库依赖注入
def get_db():
    """获取数据库会话（同步）"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """获取数据库会话（异步）"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# 数据库初始化
async def init_db():
    """初始化数据库"""
    try:
        # 创建所有表
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("数据库表创建成功")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise

async def close_db():
    """关闭数据库连接"""
    await async_engine.dispose()
    engine.dispose()
    logger.info("数据库连接已关闭")

# 数据库健康检查
async def check_db_health() -> bool:
    """检查数据库连接健康状态"""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"数据库健康检查失败: {e}")
        return False

async def check_redis_health() -> bool:
    """检查Redis连接健康状态"""
    try:
        redis_client = await get_redis()
        await redis_client.ping()
        return True
    except Exception as e:
        logger.error(f"Redis健康检查失败: {e}")
        return False
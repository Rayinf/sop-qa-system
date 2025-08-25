import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time
import uvicorn

from app.api import api_router
from app.core.config import settings
from app.core.database import engine, Base
from app.services.vector_service import VectorService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 请求日志中间件
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # 记录请求信息
        logger.info(
            f"请求开始: {request.method} {request.url.path} - "
            f"客户端: {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 记录响应信息
            logger.info(
                f"请求完成: {request.method} {request.url.path} - "
                f"状态码: {response.status_code} - "
                f"处理时间: {process_time:.3f}s"
            )
            
            # 添加处理时间到响应头
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"请求异常: {request.method} {request.url.path} - "
                f"错误: {str(e)} - "
                f"处理时间: {process_time:.3f}s"
            )
            raise

# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    logger.info("应用启动中...")
    
    try:
        # 创建数据库表
        logger.info("创建数据库表...")
        Base.metadata.create_all(bind=engine)
        
        # 初始化向量服务
        logger.info("初始化向量服务...")
        vector_service = VectorService.get_instance()
        
        # 如果支持则加载已有的向量数据库
        try:
            if hasattr(vector_service, "load_vector_store"):
                vector_service.load_vector_store()
                logger.info("已加载向量数据库")
        except Exception as e:
            logger.warning(f"加载向量数据库失败，将使用空向量库: {e}")
        
        # 设置全局向量服务实例
        app.state.vector_service = vector_service
        
        logger.info("应用启动完成")
        
    except Exception as e:
        logger.error(f"应用启动失败: {e}")
        raise
    
    yield
    
    # 关闭时执行
    logger.info("应用关闭中...")
    
    try:
        # 清理资源
        if hasattr(app.state, 'vector_service'):
            # 保存向量数据库
            if getattr(app.state.vector_service, 'vector_store', None) is not None:
                app.state.vector_service.save_vector_store(app.state.vector_service.vector_store)
                logger.info("向量数据库已保存")
            else:
                logger.info("未检测到内存中的向量数据库实例，跳过保存")
        
        logger.info("应用关闭完成")
        
    except Exception as e:
        logger.error(f"应用关闭时出错: {e}")

# 创建FastAPI应用
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="langchain知识库问答系统API",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# 配置CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# 配置受信任主机
if settings.ALLOWED_HOSTS:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )

# 添加请求日志中间件
app.add_middleware(LoggingMiddleware)

# 全局异常处理器
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(
        f"HTTP异常: {request.method} {request.url.path} - "
        f"状态码: {exc.status_code} - "
        f"详情: {exc.detail}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "http_error"
            }
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        f"请求验证错误: {request.method} {request.url.path} - "
        f"错误: {exc.errors()}"
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": 422,
                "message": "请求参数验证失败",
                "type": "validation_error",
                "details": exc.errors()
            }
        }
    )

@app.exception_handler(StarletteHTTPException)
async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(
        f"服务器异常: {request.method} {request.url.path} - "
        f"状态码: {exc.status_code} - "
        f"详情: {exc.detail}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "server_error"
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"未处理异常: {request.method} {request.url.path} - "
        f"错误: {str(exc)}",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "服务器内部错误" if not settings.DEBUG else str(exc),
                "type": "internal_error"
            }
        }
    )

# 健康检查端点
@app.get("/health")
async def health_check():
    """
    系统健康检查
    """
    try:
        # 检查数据库连接
        from app.core.database import get_db
        db = next(get_db())
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        logger.error(f"数据库健康检查失败: {e}")
        db_status = "unhealthy"
    
    # 检查向量服务
    vector_status = "healthy"
    try:
        if hasattr(app.state, 'vector_service'):
            vector_service = app.state.vector_service
            if not vector_service.vector_store:
                vector_status = "unhealthy"
        else:
            vector_status = "not_initialized"
    except Exception as e:
        logger.error(f"向量服务健康检查失败: {e}")
        vector_status = "unhealthy"
    
    # 检查Redis连接
    redis_status = "healthy"
    try:
        from app.core.database import get_redis_client
        redis_client = get_redis_client()
        if redis_client:
            redis_client.ping()
        else:
            redis_status = "unavailable"
    except Exception as e:
        logger.error(f"Redis健康检查失败: {e}")
        redis_status = "unhealthy"
    
    overall_status = "healthy" if all([
        db_status == "healthy",
        vector_status == "healthy",
        redis_status == "healthy"
    ]) else "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": time.time(),
        "components": {
            "database": db_status,
            "vector_service": vector_status,
            "redis": redis_status
        },
        "version": "1.0.0"
    }

# 根路径
@app.get("/")
async def root():
    """
    API根路径
    """
    return {
        "message": "langchain知识库问答系统API",
        "version": "1.0.0",
        "docs_url": "/docs" if settings.DEBUG else None,
        "health_url": "/health"
    }

# 包含API路由
app.include_router(api_router, prefix=settings.API_V1_STR)

# 开发服务器启动
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )

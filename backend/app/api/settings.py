from typing import Dict, Any
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.core.database import get_db
from app.core.auth import get_current_user, require_role
from app.core.config import settings
from app.models.database import User
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["settings"])

def get_vector_service(request: Request) -> VectorService:
    """从应用状态获取全局VectorService实例"""
    return request.app.state.vector_service

@router.get("/embedding", response_model=Dict[str, Any])
def get_embedding_settings(
    current_user: User = Depends(get_current_user),
    vector_service: VectorService = Depends(get_vector_service)
):
    """
    获取当前embedding设置
    """
    try:
        return {
            "embedding_mode": settings.embedding_mode,
            "current_mode": vector_service.embedding_mode,
            "api_config": {
                "model_name": settings.embedding_model_name,
                "base_url": settings.embedding_base_url,
                "dimensions": settings.embedding_dimensions
            } if settings.embedding_mode == "api" else None,
            "local_config": {
                "model_name": settings.local_embedding_model,
                "device": settings.local_embedding_device
            } if settings.embedding_mode == "local" else None
        }
    except Exception as e:
        logger.error(f"获取embedding设置失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取embedding设置失败: {str(e)}")

class EmbeddingModeRequest(BaseModel):
    mode: str

@router.post("/embedding/switch")
def switch_embedding_mode(
    request: EmbeddingModeRequest,
    db: Session = Depends(get_db),
    current_user: User = require_role(["admin"]),
    vector_service: VectorService = Depends(get_vector_service)
):
    """
    切换embedding模式
    
    需要管理员权限
    - **mode**: embedding模式 ("api" 或 "local")
    """
    try:
        if request.mode not in ["api", "local"]:
            raise HTTPException(
                status_code=400,
                detail="embedding_mode必须是'api'或'local'"
            )
        
        # 切换embedding模式
        vector_service.switch_embedding_mode(request.mode)
        
        # 更新配置（注意：这里只是运行时切换，不会持久化到配置文件）
        settings.embedding_mode = request.mode
        
        logger.info(f"用户 {current_user.username} 切换embedding模式到: {request.mode}")
        
        return {
            "message": f"embedding模式已切换到: {request.mode}",
            "success": True,
            "new_mode": request.mode
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"切换embedding模式失败: {e}")
        raise HTTPException(status_code=500, detail=f"切换embedding模式失败: {str(e)}")

@router.get("/embedding/status")
def get_embedding_status(
    current_user: User = Depends(get_current_user),
    vector_service: VectorService = Depends(get_vector_service)
):
    """
    获取embedding服务状态
    """
    try:
        return {
            "current_mode": vector_service.embedding_mode,
            "vector_store_loaded": vector_service.vector_store is not None,
            "version_fingerprint": vector_service.version_fingerprint
        }
    except Exception as e:
        logger.error(f"获取embedding状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取embedding状态失败: {str(e)}")
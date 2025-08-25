from typing import List, Optional
import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.auth import get_current_user, require_role
from app.models.database import User
from app.models.schemas import (
    KBCreate, KBUpdate, KBResponse, KBListResponse, KBStatsResponse,
    UserKBPrefsCreate, UserKBPrefsUpdate, UserKBPrefsResponse,
    KBVisibility, PaginatedResponse
)
from app.services.kb_service import KBService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["knowledge-base"])

def get_kb_service() -> KBService:
    """获取KBService实例"""
    return KBService()

@router.post("/", response_model=KBResponse)
def create_kb(
    kb_data: KBCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    kb_service: KBService = Depends(get_kb_service)
):
    """
    创建知识库
    
    - **name**: 知识库名称
    - **code**: 知识库代码（唯一标识，大写字母和数字）
    - **description**: 描述（可选）
    - **is_active**: 是否激活（默认true）
    - **visibility**: 可见性（private/org/public，默认private）
    - **kb_metadata**: 元数据（可选）
    """
    try:
        kb = kb_service.create_kb(db, kb_data, current_user.id)
        return kb_service.get_kb_response(kb)
    except Exception as e:
        logger.error(f"创建知识库失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=KBListResponse)
def get_kbs(
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(20, ge=1, le=100, description="返回的记录数"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    is_active: Optional[bool] = Query(None, description="是否激活过滤"),
    visibility: Optional[KBVisibility] = Query(None, description="可见性过滤"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    kb_service: KBService = Depends(get_kb_service)
):
    """
    获取知识库列表
    
    支持分页、搜索和过滤功能
    """
    try:
        kbs, total = kb_service.get_kbs(
            db=db,
            skip=skip,
            limit=limit,
            search_query=search,
            is_active=is_active,
            visibility=visibility,
            user_id=current_user.id
        )
        
        page = (skip // limit) + 1
        return kb_service.get_kb_list_response(kbs, total, page, limit, db)
        
    except Exception as e:
        logger.error(f"获取知识库列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active", response_model=List[KBResponse])
def get_user_active_kbs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    kb_service: KBService = Depends(get_kb_service)
):
    """
    获取用户当前激活的知识库列表
    
    返回用户置顶的知识库，如果没有则返回默认知识库
    """
    try:
        active_kbs = kb_service.get_user_active_kbs(db, current_user.id)
        return [kb_service.get_kb_response(kb) for kb in active_kbs]
        
    except Exception as e:
        logger.error(f"获取用户激活知识库失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{kb_id}", response_model=KBResponse)
def get_kb(
    kb_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    kb_service: KBService = Depends(get_kb_service)
):
    """
    根据ID获取知识库详情
    """
    try:
        kb = kb_service.get_kb_by_id(db, kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")
        
        # 获取文档数量
        from app.models.database import Document as DocumentModel
        doc_count = db.query(DocumentModel).filter(DocumentModel.kb_id == kb_id).count()
        
        return kb_service.get_kb_response(kb, doc_count)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取知识库详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/{kb_id}", response_model=KBResponse)
def update_kb(
    kb_id: uuid.UUID,
    kb_update: KBUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    kb_service: KBService = Depends(get_kb_service)
):
    """
    更新知识库信息
    
    只能更新提供的字段，未提供的字段保持不变
    """
    try:
        kb = kb_service.update_kb(db, kb_id, kb_update, current_user.id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")
        
        return kb_service.get_kb_response(kb)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新知识库失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{kb_id}")
def delete_kb(
    kb_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    kb_service: KBService = Depends(get_kb_service)
):
    """
    删除知识库
    
    注意：
    - 不能删除默认知识库
    - 知识库中有文档时不能删除，需要先移除或转移文档
    """
    try:
        success = kb_service.delete_kb(db, kb_id, current_user.id)
        if success:
            return {"message": "知识库删除成功"}
        else:
            raise HTTPException(status_code=500, detail="删除知识库失败")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除知识库失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{kb_id}/stats", response_model=KBStatsResponse)
def get_kb_stats(
    kb_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    kb_service: KBService = Depends(get_kb_service)
):
    """
    获取知识库统计信息
    
    包括文档数量、向量化状态、最近更新时间、热门主题等
    """
    try:
        stats = kb_service.get_kb_stats(db, kb_id)
        return KBStatsResponse(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取知识库统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{kb_id}/documents/move")
def move_documents_to_kb(
    kb_id: uuid.UUID,
    document_ids: List[uuid.UUID],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    kb_service: KBService = Depends(get_kb_service)
):
    """
    将文档移动到指定知识库
    
    - **document_ids**: 要移动的文档ID列表
    """
    try:
        if not document_ids:
            raise HTTPException(status_code=400, detail="文档ID列表不能为空")
        
        result = kb_service.move_documents_to_kb(db, document_ids, kb_id, current_user.id)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"移动文档到知识库失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 用户知识库偏好相关接口
@router.get("/prefs/my", response_model=List[UserKBPrefsResponse])
def get_my_kb_prefs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    kb_service: KBService = Depends(get_kb_service)
):
    """
    获取我的知识库偏好设置
    
    返回用户对各个知识库的偏好设置（如是否置顶等）
    """
    try:
        prefs = kb_service.get_user_kb_prefs(db, current_user.id)
        
        result = []
        for pref in prefs:
            kb_response = kb_service.get_kb_response(pref.kb)
            result.append(UserKBPrefsResponse(
                id=pref.id,
                user_id=pref.user_id,
                kb_id=pref.kb_id,
                pinned=pref.pinned,
                created_at=pref.created_at,
                updated_at=pref.updated_at,
                kb=kb_response
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"获取用户知识库偏好失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/prefs/{kb_id}", response_model=UserKBPrefsResponse)
def update_kb_prefs(
    kb_id: uuid.UUID,
    prefs_update: UserKBPrefsUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    kb_service: KBService = Depends(get_kb_service)
):
    """
    更新对指定知识库的偏好设置
    
    - **pinned**: 是否置顶该知识库
    """
    try:
        prefs = kb_service.update_user_kb_prefs(db, current_user.id, kb_id, prefs_update)
        
        # 获取知识库信息
        kb = kb_service.get_kb_by_id(db, kb_id)
        kb_response = kb_service.get_kb_response(kb) if kb else None
        
        return UserKBPrefsResponse(
            id=prefs.id,
            user_id=prefs.user_id,
            kb_id=prefs.kb_id,
            pinned=prefs.pinned,
            created_at=prefs.created_at,
            updated_at=prefs.updated_at,
            kb=kb_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新知识库偏好失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 枚举接口
@router.get("/enums/visibility")
def get_kb_visibility_options():
    """
    获取知识库可见性选项
    """
    return {
        "visibility_options": [
            {"value": "private", "label": "私有", "description": "仅创建者可见"},
            {"value": "org", "label": "组织", "description": "组织内可见"},
            {"value": "public", "label": "公开", "description": "所有人可见"}
        ]
    }

@router.get("/code/{code}", response_model=KBResponse)
def get_kb_by_code(
    code: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    kb_service: KBService = Depends(get_kb_service)
):
    """
    根据代码获取知识库
    
    - **code**: 知识库代码
    """
    try:
        kb = kb_service.get_kb_by_code(db, code)
        if not kb:
            raise HTTPException(status_code=404, detail=f"知识库代码 '{code}' 不存在")
        
        # 获取文档数量
        from app.models.database import Document as DocumentModel
        doc_count = db.query(DocumentModel).filter(DocumentModel.kb_id == kb.id).count()
        
        return kb_service.get_kb_response(kb, doc_count)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"根据代码获取知识库失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
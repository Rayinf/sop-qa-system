from typing import List, Optional
import logging
import time
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.auth import get_current_user, require_role
from app.models.database import User
from app.models.schemas import (
    DocumentCreate, DocumentUpdate, DocumentResponse,
    PaginatedResponse, SearchParams, BulkOperationRequest,
    DocumentStatus, ProcessingStatus
)
from app.services.document_service import DocumentService
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

# 获取全局服务实例的依赖函数
def get_vector_service(request: Request) -> VectorService:
    """获取全局VectorService实例"""
    return request.app.state.vector_service

def get_document_service(request: Request) -> DocumentService:
    """获取DocumentService实例，使用全局VectorService"""
    vector_service = get_vector_service(request)
    return DocumentService(vector_service=vector_service)

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Query(..., description="文档标题"),
    category: Optional[str] = Query(None, description="文档类别，留空则自动分类"),
    tags: Optional[str] = Query(None, description="标签，用逗号分隔"),
    version: str = Query("1.0", description="版本号"),
    auto_vectorize: bool = Query(True, description="是否自动向量化"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    上传文档
    
    - **file**: 文档文件（支持PDF、Word、TXT、Markdown）
    - **title**: 文档标题
    - **category**: 文档类别
    - **tags**: 标签列表
    - **version**: 版本号
    - **auto_vectorize**: 是否自动向量化
    """
    try:
        # 解析标签
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else []
        
        # 创建文档数据
        document_data = DocumentCreate(
            title=title,
            category=category or "通用文档",  # 如果未指定类别，使用默认值触发自动分类
            tags=tag_list,
            version=version
        )
        
        # 创建文档
        document = await document_service.create_document(
            db=db,
            file=file,
            document_data=document_data,
            user_id=current_user.id
        )
        
        # 如果需要自动向量化，添加到后台任务
        if auto_vectorize and document.status == "processed":
            background_tasks.add_task(
                vectorize_document_task,
                db,
                document.id,
                document_service
            )
        
        logger.info(f"文档上传成功: {document.id}")
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传文档失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传文档失败: {str(e)}")

@router.get("/", response_model=PaginatedResponse[DocumentResponse])
def get_documents(
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(20, ge=1, le=100, description="返回的记录数"),
    category: Optional[str] = Query(None, description="文档类别过滤"),
    status: Optional[str] = Query(None, description="文档状态过滤"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    my_documents: bool = Query(False, description="只显示我的文档"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    获取文档列表
    
    支持分页、过滤和搜索功能
    """
    try:
        user_id = current_user.id if my_documents else None
        
        documents, total = document_service.get_documents(
            db=db,
            skip=skip,
            limit=limit,
            category=category,
            status=status,
            user_id=user_id,
            search_query=search
        )
        
        # 转换为响应格式
        document_responses = [
            document_service.get_document_response(doc) for doc in documents
        ]
        
        current_page = skip // limit + 1
        total_pages = (total + limit - 1) // limit
        
        return PaginatedResponse(
            items=document_responses,
            total=total,
            page=current_page,
            size=limit,
            pages=total_pages,
            has_next=current_page < total_pages,
            has_prev=current_page > 1
        )
        
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")

@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    获取单个文档详情
    """
    try:
        document = document_service.get_document_by_id(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="文档不存在")
        
        return document_service.get_document_response(document)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档详情失败: {str(e)}")

@router.put("/{document_id}", response_model=DocumentResponse)
def update_document(
    document_id: str,
    document_update: DocumentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    更新文档信息
    
    只有文档上传者可以更新文档
    """
    try:
        document = document_service.update_document(
            db=db,
            document_id=document_id,
            document_update=document_update,
            user_id=current_user.id
        )
        
        if not document:
            raise HTTPException(status_code=404, detail="文档不存在")
        
        return document_service.get_document_response(document)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新文档失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新文档失败: {str(e)}")

@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    删除文档
    
    只有文档上传者或管理员可以删除文档
    """
    try:
        success = document_service.delete_document(
            db=db,
            document_id=document_id,
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="文档不存在或无权限删除")
        
        return JSONResponse(
            status_code=200,
            content={"message": "文档删除成功", "document_id": document_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")

@router.get("/{document_id}/chunks")
def get_document_chunks(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    获取文档片段
    """
    try:
        chunks = document_service.get_document_chunks(db, document_id)
        
        return {
            "document_id": document_id,
            "total_chunks": len(chunks),
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.chunk_text,
                    "page_number": chunk.page_number,
                    "metadata": chunk.vector_metadata or {}
                }
                for chunk in chunks
            ]
        }
        
    except Exception as e:
        logger.error(f"获取文档片段失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档片段失败: {str(e)}")

@router.post("/{document_id}/vectorize")
def vectorize_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    手动向量化文档
    """
    try:
        # 检查文档是否存在
        document = document_service.get_document_by_id(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="文档不存在")
        
        # 检查文档状态
        if document.status not in ["processed", "completed", "vectorized"]:
            raise HTTPException(
                status_code=400, 
                detail="文档状态不允许向量化"
            )
        
        # 添加向量化任务到后台
        background_tasks.add_task(
            vectorize_document_task,
            db,
            document_id,
            document_service
        )
        
        return JSONResponse(
            status_code=202,
            content={
                "message": "向量化任务已启动",
                "document_id": document_id,
                "task_id": f"vectorize_{document_id}_{int(time.time())}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动向量化任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动向量化任务失败: {str(e)}")

@router.get("/{document_id}/vectorize/progress")
def get_vectorization_progress(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    获取文档向量化进度
    """
    try:
        # 检查文档是否存在
        document = document_service.get_document_by_id(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="文档不存在")
        
        # 从Redis获取向量化进度
        progress_info = document_service.vector_service.get_vectorization_progress(document_id)
        
        if progress_info is None:
            # 如果没有进度信息，根据文档状态返回默认进度
            if document.status == "vectorized":
                return {
                    "document_id": document_id,
                    "status": "completed",
                    "progress": 100,
                    "current_step": "向量化完成",
                    "total_steps": 4,
                    "current_step_index": 4,
                    "message": "文档已完成向量化",
                    "error": None
                }
            elif document.processing_status == "vectorizing":
                return {
                    "document_id": document_id,
                    "status": "processing",
                    "progress": 50,
                    "current_step": "向量化处理中",
                    "total_steps": 4,
                    "current_step_index": 2,
                    "message": "正在处理文档向量化",
                    "error": None
                }
            else:
                return {
                    "document_id": document_id,
                    "status": "pending",
                    "progress": 0,
                    "current_step": "等待开始",
                    "total_steps": 4,
                    "current_step_index": 0,
                    "message": "向量化任务尚未开始",
                    "error": None
                }
        
        return progress_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取向量化进度失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取向量化进度失败: {str(e)}")

@router.post("/search")
def search_documents(
    request_data: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    搜索文档
    
    基于向量相似度搜索文档内容
    """
    try:
        # 从请求数据中提取参数
        query = request_data.get("query", "")
        filters = request_data.get("filters", {})
        limit = request_data.get("limit", 10)
        
        # 从filters中提取category，如果没有则使用顶级category参数
        category = filters.get("category") or request_data.get("category")
        
        results = document_service.search_documents(
            query=query,
            k=limit,
            category=category
        )
        
        return {
            "query": query,
            "total_results": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"搜索文档失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索文档失败: {str(e)}")

@router.get("/statistics/overview")
def get_document_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    获取文档统计信息
    """
    try:
        stats = document_service.get_document_statistics(db)
        return stats
        
    except Exception as e:
        logger.error(f"获取文档统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档统计失败: {str(e)}")

@router.post("/bulk/operation")
def bulk_operation(
    operation_request: BulkOperationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    批量操作文档
    
    支持批量向量化、删除等操作
    需要管理员权限
    """
    # 权限检查
    if current_user.role not in ["admin", "manager"] and not current_user.is_superuser:
        raise HTTPException(
            status_code=403,
            detail="需要管理员或经理权限"
        )
    
    try:
        # 添加到后台任务
        background_tasks.add_task(
            bulk_operation_task,
            db,
            operation_request.document_ids,
            operation_request.operation,
            current_user.id,
            document_service
        )
        
        return JSONResponse(
            status_code=202,
            content={
                "message": f"批量{operation_request.operation}任务已启动",
                "document_count": len(operation_request.document_ids)
            }
        )
        
    except Exception as e:
        logger.error(f"批量操作失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量操作失败: {str(e)}")

@router.post("/rebuild-index")
def rebuild_vector_index(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    重建向量索引
    
    需要管理员权限
    """
    # 权限检查
    if current_user.role != "admin" and not current_user.is_superuser:
        raise HTTPException(
            status_code=403,
            detail="需要管理员权限"
        )
    
    try:
        # 添加到后台任务
        background_tasks.add_task(
            rebuild_index_task,
            db,
            document_service
        )
        
        return JSONResponse(
            status_code=202,
            content={"message": "向量索引重建任务已启动"}
        )
        
    except Exception as e:
        logger.error(f"重建向量索引失败: {e}")
        raise HTTPException(status_code=500, detail=f"重建向量索引失败: {str(e)}")

# 后台任务函数
def vectorize_document_task(db: Session, document_id: str, document_service: DocumentService):
    """
    向量化文档的后台任务
    """
    try:
        logger.info(f"开始向量化文档: {document_id}")
        
        # 更新文档状态为处理中
        document_service.document_processor.update_document_status(
            db, document_id, "vectorizing"
        )
        
        # 执行向量化
        success = document_service.vectorize_document(db, document_id)
        
        if success:
            logger.info(f"文档向量化成功: {document_id}")
        else:
            logger.error(f"文档向量化失败: {document_id}")
            document_service.document_processor.update_document_status(
                db, document_id, "error"
            )
            
    except Exception as e:
        logger.error(f"向量化任务执行失败 {document_id}: {e}")
        document_service.document_processor.update_document_status(
            db, document_id, "error"
        )

def bulk_operation_task(db: Session, document_ids: List[str], operation: str, user_id: int, document_service: DocumentService):
    """
    批量操作的后台任务
    """
    try:
        logger.info(f"开始批量{operation}操作，文档数量: {len(document_ids)}")
        
        results = document_service.batch_process_documents(
            db, document_ids, operation
        )
        
        logger.info(
            f"批量{operation}操作完成: 成功 {len(results['success'])}, "
            f"失败 {len(results['failed'])}"
        )
        
    except Exception as e:
        logger.error(f"批量操作任务执行失败: {e}")

def rebuild_index_task(db: Session, document_service: DocumentService):
    """
    重建向量索引的后台任务
    """
    try:
        logger.info("开始重建向量索引")
        
        success = document_service.rebuild_vector_index(db)
        
        if success:
            logger.info("向量索引重建成功")
        else:
            logger.error("向量索引重建失败")
            
    except Exception as e:
        logger.error(f"重建索引任务执行失败: {e}")

# 文档类别和状态枚举接口
@router.get("/enums/categories")
def get_document_categories():
    """
    获取文档类别列表
    """
    return {
        "categories": [
            "通用文档",
            "操作手册",
            "政策文件",
            "指导文件",
            "流程文档",
            "技术规范",
            "培训资料",
            "质量文档",
            "安全文档",
            "其他"
        ]
    }

@router.get("/enums/statuses")
def get_document_statuses():
    """
    获取文档状态列表
    """
    return {
        "statuses": [
            "uploaded",
            "processing",
            "processed",
            "vectorizing",
            "vectorized",
            "error"
        ]
    }
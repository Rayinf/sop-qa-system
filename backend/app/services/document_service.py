from typing import List, Optional, Dict, Any, Tuple
import os
import shutil
import logging
from datetime import datetime
from pathlib import Path

from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from app.core.config import settings
from app.models.database import Document as DocumentModel, VectorIndex, User
from app.models.schemas import (
    DocumentCreate, DocumentUpdate, DocumentResponse,
    DocumentStatus, ProcessingStatus
)
from app.services.document_processor import DocumentProcessor
from app.services.vector_service import VectorService
from app.core.database import get_redis_client

logger = logging.getLogger(__name__)

class DocumentService:
    """文档管理服务类"""
    
    def __init__(self, vector_service: Optional[VectorService] = None):
        self.document_processor = DocumentProcessor()
        self.vector_service = vector_service or VectorService.get_instance()
        self.upload_dir = settings.upload_dir
        self.ensure_upload_directory()
    
    def ensure_upload_directory(self):
        """确保上传目录存在"""
        os.makedirs(self.upload_dir, exist_ok=True)
    
    async def upload_file(self, 
                         file: UploadFile, 
                         user_id: int,
                         title: str,
                         category: str = "通用文档",
                         tags: List[str] = None,
                         version: str = "1.0") -> Dict[str, Any]:
        """上传文件"""
        try:
            # 验证文件类型
            if not self.document_processor.is_supported_file(file.filename):
                raise HTTPException(
                    status_code=400, 
                    detail=f"不支持的文件类型: {Path(file.filename).suffix}"
                )
            
            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = Path(file.filename).suffix
            safe_filename = f"{timestamp}_{file.filename}"
            file_path = os.path.join(self.upload_dir, safe_filename)
            
            # 保存文件
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # 验证文件大小
            if not self.validate_file_size(file_path):
                os.remove(file_path)
                raise HTTPException(
                    status_code=400,
                    detail=f"文件大小超过限制 ({settings.max_file_size_mb}MB)"
                )
            
            logger.info(f"文件上传成功: {file_path}")
            
            return {
                'file_path': file_path,
                'original_filename': file.filename,
                'saved_filename': safe_filename,
                'file_size': os.path.getsize(file_path)
            }
            
        except Exception as e:
            logger.error(f"文件上传失败: {e}")
            # 清理已上传的文件
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")
    
    def validate_file_size(self, file_path: str) -> bool:
        """验证文件大小"""
        try:
            file_size = os.path.getsize(file_path)
            max_size = settings.max_file_size_mb * 1024 * 1024
            return file_size <= max_size
        except Exception:
            return False
    
    def process_document(self, 
                        db: Session,
                        file_path: str,
                        title: str,
                        category: str,
                        tags: List[str],
                        version: str,
                        user_id: int) -> DocumentModel:
        """处理文档"""
        try:
            # 检查文件是否已存在（基于哈希）
            file_hash = self.document_processor.calculate_file_hash(file_path)
            existing_doc = self.document_processor.get_document_by_hash(db, file_hash)
            
            if existing_doc:
                logger.info(f"文档已存在: {existing_doc.id}")
                # 删除重复上传的文件
                if os.path.exists(file_path):
                    os.remove(file_path)
                return existing_doc
            
            # 处理文档
            processing_result = self.document_processor.process_document(
                file_path=file_path,
                title=title,
                category=category,
                tags=tags,
                version=version,
                user_id=user_id
            )
            
            # 保存文档到数据库
            document = self.document_processor.save_document_to_db(db, processing_result)
            
            # 保存文档片段到数据库
            self.document_processor.save_chunks_to_db(
                db, 
                str(document.id), 
                processing_result['chunks']
            )
            
            logger.info(f"文档处理完成: {document.id}")
            return document
            
        except Exception as e:
            logger.error(f"文档处理失败: {e}")
            raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")
    
    def vectorize_document(self, db: Session, document_id: str) -> bool:
        """向量化文档"""
        try:
            # 获取文档片段
            vector_indices = db.query(VectorIndex).filter(
                VectorIndex.document_id == document_id
            ).all()
            
            if not vector_indices:
                logger.warning(f"文档 {document_id} 没有找到片段")
                return False
            
            # 转换为Document对象
            documents = []
            for vi in vector_indices:
                doc = {
                    'page_content': vi.chunk_text,
                    'metadata': {
                        **vi.vector_metadata,
                        'chunk_id': vi.chunk_id,
                        'chunk_index': vi.chunk_index,
                        'document_id': document_id
                    }
                }
                from langchain.schema import Document
                documents.append(Document(
                    page_content=doc['page_content'],
                    metadata=doc['metadata']
                ))
            
            # 执行向量化
            success = self.vector_service.vectorize_document(db, document_id, documents)
            
            if success:
                logger.info(f"文档向量化成功: {document_id}")
                # 更新处理状态为completed
                self.document_processor.update_processing_status(db, document_id, "completed")
            else:
                logger.error(f"文档向量化失败: {document_id}")
                # 更新处理状态为failed
                self.document_processor.update_processing_status(db, document_id, "failed")
            
            return success
            
        except Exception as e:
            logger.error(f"文档向量化失败 {document_id}: {e}")
            return False
    
    async def create_document(self, 
                             db: Session,
                             file: UploadFile,
                             document_data: DocumentCreate,
                             user_id: int) -> DocumentResponse:
        """创建文档（完整流程）"""
        try:
            # 1. 上传文件
            upload_result = await self.upload_file(
                file=file,
                user_id=user_id,
                title=document_data.title,
                category=document_data.category,
                tags=document_data.tags,
                version=document_data.version
            )
            
            # 2. 处理文档
            document = self.process_document(
                db=db,
                file_path=upload_result['file_path'],
                title=document_data.title,
                category=document_data.category,
                tags=document_data.tags or [],
                version=document_data.version,
                user_id=user_id
            )
            
            # 3. 异步向量化（可选择同步或异步）
            if settings.auto_vectorize:
                vectorize_success = self.vectorize_document(db, document.id)
                if not vectorize_success:
                    logger.warning(f"文档 {document.id} 向量化失败，但文档已创建")
            
            # 4. 返回文档信息
            return self.get_document_response(document)
            
        except Exception as e:
            logger.error(f"创建文档失败: {e}")
            raise
    
    def get_document_by_id(self, db: Session, document_id: str) -> Optional[DocumentModel]:
        """根据ID获取文档"""
        try:
            return db.query(DocumentModel).filter(
                DocumentModel.id == document_id
            ).first()
        except Exception as e:
            logger.error(f"获取文档失败: {e}")
            return None
    
    def get_documents(self, 
                     db: Session,
                     skip: int = 0,
                     limit: int = 20,
                     category: Optional[str] = None,
                     status: Optional[str] = None,
                     user_id: Optional[int] = None,
                     search_query: Optional[str] = None) -> Tuple[List[DocumentModel], int]:
        """获取文档列表"""
        try:
            query = db.query(DocumentModel)
            
            # 应用过滤条件
            if category:
                query = query.filter(DocumentModel.category == category)
            
            if status:
                query = query.filter(DocumentModel.status == status)
            
            if user_id:
                query = query.filter(DocumentModel.uploaded_by == user_id)
            
            if search_query:
                search_filter = or_(
                    DocumentModel.title.ilike(f"%{search_query}%"),
                    DocumentModel.category.ilike(f"%{search_query}%")
                )
                query = query.filter(search_filter)
            
            # 获取总数
            total = query.count()
            
            # 应用分页和排序
            documents = query.order_by(desc(DocumentModel.created_at)).offset(skip).limit(limit).all()
            
            return documents, total
            
        except Exception as e:
            logger.error(f"获取文档列表失败: {e}")
            return [], 0
    
    def update_document(self, 
                       db: Session,
                       document_id: str,
                       document_update: DocumentUpdate,
                       user_id: int) -> Optional[DocumentModel]:
        """更新文档"""
        try:
            document = self.get_document_by_id(db, document_id)
            if not document:
                return None
            
            # 检查权限（只有上传者或管理员可以更新）
            if document.uploaded_by != user_id:
                # 检查是否为管理员
                user = db.query(User).filter(User.id == user_id).first()
                if not user or (user.role != "admin" and not user.is_superuser):
                    raise HTTPException(status_code=403, detail="没有权限更新此文档")
            
            # 更新字段
            update_data = document_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(document, field, value)
            
            document.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(document)
            
            logger.info(f"文档更新成功: {document_id}")
            return document
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"更新文档失败: {e}")
            db.rollback()
            raise
    
    def delete_document(self, 
                       db: Session,
                       document_id: str,
                       user_id: int) -> bool:
        """删除文档"""
        try:
            document = self.get_document_by_id(db, document_id)
            if not document:
                return False
            
            # 检查权限（只有上传者或管理员可以删除）
            if document.uploaded_by and document.uploaded_by != user_id:
                # 检查是否为管理员
                user = db.query(User).filter(User.id == user_id).first()
                if not user or (user.role != "admin" and not user.is_superuser):
                    raise HTTPException(status_code=403, detail="没有权限删除此文档")
            
            # 从向量数据库中删除（标记为删除）
            self.vector_service.delete_document_from_vector_store(document_id)
            
            # 删除文档记录和相关数据
            success = self.document_processor.delete_document(db, document_id)
            
            if success:
                logger.info(f"文档删除成功: {document_id}")
            
            return success
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False
    
    def get_document_chunks(self, 
                           db: Session,
                           document_id: str) -> List[VectorIndex]:
        """获取文档片段"""
        try:
            return db.query(VectorIndex).filter(
                VectorIndex.document_id == document_id
            ).order_by(VectorIndex.chunk_index).all()
        except Exception as e:
            logger.error(f"获取文档片段失败: {e}")
            return []
    
    def search_documents(self, 
                        query: str,
                        k: int = 5,
                        category: Optional[str] = None) -> List[Dict[str, Any]]:
        """搜索文档"""
        try:
            # 检查缓存
            cache_key = f"{query}_{k}_{category or 'all'}"
            cached_result = self.vector_service.get_cached_search_result(cache_key)
            if cached_result:
                logger.info("返回缓存的搜索结果")
                return cached_result
            
            # 构建过滤条件
            filter_dict = {}
            if category:
                filter_dict['category'] = category
            
            # 执行向量搜索
            search_results = self.vector_service.search_similar_documents(
                query=query,
                k=k,
                filter_dict=filter_dict if filter_dict else None
            )
            
            # 格式化结果
            formatted_results = []
            for doc, score in search_results:
                result = {
                    'document_id': doc.metadata.get('document_id'),
                    'chunk_id': doc.metadata.get('chunk_id'),
                    'title': doc.metadata.get('title'),
                    'content': doc.page_content,
                    'score': float(score),
                    'metadata': doc.metadata
                }
                formatted_results.append(result)
            
            # 缓存结果
            self.vector_service.cache_search_result(cache_key, formatted_results)
            
            logger.info(f"搜索完成，找到 {len(formatted_results)} 个结果")
            return formatted_results
            
        except Exception as e:
            logger.error(f"搜索文档失败: {e}")
            return []
    
    def get_document_statistics(self, db: Session) -> Dict[str, Any]:
        """获取文档统计信息"""
        try:
            # 基本统计
            total_documents = db.query(DocumentModel).count()
            
            # 按状态统计
            status_stats = {}
            for status in ["uploaded", "processing", "processed", "vectorized", "error"]:
                count = db.query(DocumentModel).filter(
                    DocumentModel.status == status
                ).count()
                status_stats[status] = count
            
            # 按类别统计
            from sqlalchemy import func
            category_stats = db.query(
                DocumentModel.category,
                func.count(DocumentModel.id).label('count')
            ).group_by(DocumentModel.category).all()
            
            category_dict = {cat: count for cat, count in category_stats}
            
            # 向量数据库统计
            vector_stats = self.vector_service.get_vector_store_stats()
            
            return {
                'total_documents': total_documents,
                'status_distribution': status_stats,
                'category_distribution': category_dict,
                'vector_store_stats': vector_stats,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取文档统计失败: {e}")
            return {}
    
    def rebuild_vector_index(self, db: Session) -> bool:
        """重建向量索引"""
        try:
            logger.info("开始重建向量索引")
            success = self.vector_service.rebuild_vector_store(db)
            
            if success:
                logger.info("向量索引重建成功")
            else:
                logger.error("向量索引重建失败")
            
            return success
            
        except Exception as e:
            logger.error(f"重建向量索引失败: {e}")
            return False
    
    def get_document_response(self, document: DocumentModel) -> DocumentResponse:
        """转换为响应格式"""
        return DocumentResponse(
            id=document.id,
            title=document.title,
            filename=os.path.basename(document.file_path) if document.file_path else '',
            file_type=document.file_type,
            file_size=document.file_size,
            version=document.version,
            status=document.status,
            uploaded_by=document.uploaded_by,
            processing_status=document.processing_status,
            processing_error=document.processing_error,
            category=document.category,
            tags=document.tags,
            description=document.description,
            language=document.language,
            metadata=document.doc_metadata or {},
            created_at=document.created_at,
            updated_at=document.updated_at
        )
    
    def batch_process_documents(self, 
                               db: Session,
                               document_ids: List[str],
                               operation: str) -> Dict[str, Any]:
        """批量处理文档"""
        try:
            results = {
                'success': [],
                'failed': [],
                'total': len(document_ids)
            }
            
            for doc_id in document_ids:
                try:
                    if operation == "vectorize":
                        success = self.vectorize_document(db, doc_id)
                    elif operation == "delete":
                        success = self.delete_document(db, doc_id, user_id=None)  # 需要管理员权限
                    else:
                        success = False
                    
                    if success:
                        results['success'].append(doc_id)
                    else:
                        results['failed'].append(doc_id)
                        
                except Exception as e:
                    logger.error(f"批量处理文档 {doc_id} 失败: {e}")
                    results['failed'].append(doc_id)
            
            logger.info(f"批量处理完成: 成功 {len(results['success'])}, 失败 {len(results['failed'])}")
            return results
            
        except Exception as e:
            logger.error(f"批量处理文档失败: {e}")
            return {'success': [], 'failed': document_ids, 'total': len(document_ids)}

# 文档服务工具函数
def get_file_info(file_path: str) -> Dict[str, Any]:
    """获取文件信息"""
    try:
        stat = os.stat(file_path)
        return {
            'size': stat.st_size,
            'created_at': datetime.fromtimestamp(stat.st_ctime),
            'modified_at': datetime.fromtimestamp(stat.st_mtime),
            'extension': Path(file_path).suffix.lower()
        }
    except Exception as e:
        logger.error(f"获取文件信息失败: {e}")
        return {}

def cleanup_orphaned_files(upload_dir: str, db: Session) -> int:
    """清理孤立文件"""
    try:
        cleaned_count = 0
        
        # 获取数据库中所有文件路径
        db_file_paths = set(
            path[0] for path in db.query(DocumentModel.file_path).all()
        )
        
        # 检查上传目录中的文件
        for file_path in Path(upload_dir).rglob('*'):
            if file_path.is_file():
                file_path_str = str(file_path)
                if file_path_str not in db_file_paths:
                    # 删除孤立文件
                    os.remove(file_path_str)
                    cleaned_count += 1
                    logger.info(f"删除孤立文件: {file_path_str}")
        
        return cleaned_count
        
    except Exception as e:
        logger.error(f"清理孤立文件失败: {e}")
        return 0
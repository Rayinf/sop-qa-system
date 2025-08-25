from typing import List, Optional, Dict, Any, Tuple
import logging
from datetime import datetime
import uuid

from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func

from app.models.database import KB as KBModel, Document as DocumentModel, UserKBPrefs, User
from app.models.schemas import (
    KBCreate, KBUpdate, KBResponse, KBListResponse, KBStatsResponse,
    UserKBPrefsCreate, UserKBPrefsUpdate, UserKBPrefsResponse,
    KBVisibility
)

logger = logging.getLogger(__name__)

class KBService:
    """知识库管理服务类"""
    
    def __init__(self):
        pass
    
    def create_kb(self, db: Session, kb_data: KBCreate, user_id: uuid.UUID) -> KBModel:
        """创建知识库"""
        try:
            # 检查代码是否已存在
            existing_kb = db.query(KBModel).filter(KBModel.code == kb_data.code).first()
            if existing_kb:
                raise HTTPException(
                    status_code=400,
                    detail=f"知识库代码 '{kb_data.code}' 已存在"
                )
            
            # 创建知识库
            kb = KBModel(
                name=kb_data.name,
                code=kb_data.code,
                description=kb_data.description,
                is_active=kb_data.is_active,
                visibility=kb_data.visibility,
                kb_metadata=kb_data.kb_metadata or {}
            )
            
            db.add(kb)
            db.commit()
            db.refresh(kb)
            
            logger.info(f"用户 {user_id} 创建了知识库: {kb.name} ({kb.code})")
            return kb
            
        except Exception as e:
            db.rollback()
            logger.error(f"创建知识库失败: {e}")
            raise HTTPException(status_code=500, detail=f"创建知识库失败: {str(e)}")
    
    def get_kb_by_id(self, db: Session, kb_id: uuid.UUID) -> Optional[KBModel]:
        """根据ID获取知识库"""
        return db.query(KBModel).filter(KBModel.id == kb_id).first()
    
    def get_kb_by_code(self, db: Session, code: str) -> Optional[KBModel]:
        """根据代码获取知识库"""
        return db.query(KBModel).filter(KBModel.code == code).first()
    
    def get_kbs(self, 
                db: Session,
                skip: int = 0,
                limit: int = 20,
                search_query: Optional[str] = None,
                is_active: Optional[bool] = None,
                visibility: Optional[KBVisibility] = None,
                user_id: Optional[uuid.UUID] = None) -> Tuple[List[KBModel], int]:
        """获取知识库列表"""
        try:
            query = db.query(KBModel)
            
            # 搜索过滤
            if search_query:
                search_pattern = f"%{search_query}%"
                query = query.filter(
                    or_(
                        KBModel.name.ilike(search_pattern),
                        KBModel.code.ilike(search_pattern),
                        KBModel.description.ilike(search_pattern)
                    )
                )
            
            # 状态过滤
            if is_active is not None:
                query = query.filter(KBModel.is_active == is_active)
            
            # 可见性过滤
            if visibility:
                query = query.filter(KBModel.visibility == visibility)
            
            # 获取总数
            total = query.count()
            
            # 分页和排序
            kbs = query.order_by(desc(KBModel.updated_at)).offset(skip).limit(limit).all()
            
            return kbs, total
            
        except Exception as e:
            logger.error(f"获取知识库列表失败: {e}")
            raise HTTPException(status_code=500, detail=f"获取知识库列表失败: {str(e)}")
    
    def update_kb(self, 
                  db: Session, 
                  kb_id: uuid.UUID, 
                  kb_update: KBUpdate,
                  user_id: uuid.UUID) -> Optional[KBModel]:
        """更新知识库"""
        try:
            kb = self.get_kb_by_id(db, kb_id)
            if not kb:
                raise HTTPException(status_code=404, detail="知识库不存在")
            
            # 更新字段
            update_data = kb_update.dict(exclude_unset=True)
            
            # 检查代码唯一性（如果要更新代码）
            if "code" in update_data and update_data["code"] != kb.code:
                existing_kb = db.query(KBModel).filter(
                    and_(
                        KBModel.code == update_data["code"],
                        KBModel.id != kb_id
                    )
                ).first()
                if existing_kb:
                    raise HTTPException(
                        status_code=400,
                        detail=f"知识库代码 '{update_data['code']}' 已存在"
                    )
            
            for field, value in update_data.items():
                setattr(kb, field, value)
            
            kb.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(kb)
            
            logger.info(f"用户 {user_id} 更新了知识库: {kb.name} ({kb.code})")
            return kb
            
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"更新知识库失败: {e}")
            raise HTTPException(status_code=500, detail=f"更新知识库失败: {str(e)}")
    
    def delete_kb(self, db: Session, kb_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """删除知识库"""
        try:
            kb = self.get_kb_by_id(db, kb_id)
            if not kb:
                raise HTTPException(status_code=404, detail="知识库不存在")
            
            # 检查是否为默认知识库
            if kb.code == "DEFAULT":
                raise HTTPException(status_code=400, detail="不能删除默认知识库")
            
            # 检查是否有关联的文档
            doc_count = db.query(DocumentModel).filter(DocumentModel.kb_id == kb_id).count()
            if doc_count > 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"知识库中还有 {doc_count} 个文档，请先移除或转移文档后再删除"
                )
            
            # 删除用户偏好
            db.query(UserKBPrefs).filter(UserKBPrefs.kb_id == kb_id).delete()
            
            # 删除知识库
            db.delete(kb)
            db.commit()
            
            logger.info(f"用户 {user_id} 删除了知识库: {kb.name} ({kb.code})")
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"删除知识库失败: {e}")
            raise HTTPException(status_code=500, detail=f"删除知识库失败: {str(e)}")
    
    def get_kb_stats(self, db: Session, kb_id: uuid.UUID) -> Dict[str, Any]:
        """获取知识库统计信息"""
        try:
            kb = self.get_kb_by_id(db, kb_id)
            if not kb:
                raise HTTPException(status_code=404, detail="知识库不存在")
            
            # 文档统计
            doc_stats = db.query(
                func.count(DocumentModel.id).label('total_docs'),
                func.count(DocumentModel.id).filter(DocumentModel.status == 'vectorized').label('vectorized_docs')
            ).filter(DocumentModel.kb_id == kb_id).first()
            
            # 最近更新时间
            last_doc = db.query(DocumentModel).filter(
                DocumentModel.kb_id == kb_id
            ).order_by(desc(DocumentModel.updated_at)).first()
            
            # 热门主题（基于文档分类）
            popular_topics = db.query(
                DocumentModel.category,
                func.count(DocumentModel.id).label('count')
            ).filter(
                DocumentModel.kb_id == kb_id
            ).group_by(DocumentModel.category).order_by(desc('count')).limit(5).all()
            
            return {
                'id': kb.id,
                'name': kb.name,
                'code': kb.code,
                'document_count': doc_stats.total_docs or 0,
                'vectorized_count': doc_stats.vectorized_docs or 0,
                'last_updated': last_doc.updated_at if last_doc else None,
                'popular_topics': [topic.category for topic in popular_topics if topic.category]
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"获取知识库统计失败: {e}")
            raise HTTPException(status_code=500, detail=f"获取知识库统计失败: {str(e)}")
    
    def get_user_kb_prefs(self, db: Session, user_id: uuid.UUID) -> List[UserKBPrefs]:
        """获取用户知识库偏好"""
        return db.query(UserKBPrefs).filter(
            UserKBPrefs.user_id == user_id
        ).join(KBModel).filter(KBModel.is_active == True).all()
    
    def update_user_kb_prefs(self, 
                             db: Session, 
                             user_id: uuid.UUID, 
                             kb_id: uuid.UUID, 
                             prefs_update: UserKBPrefsUpdate) -> UserKBPrefs:
        """更新用户知识库偏好"""
        try:
            # 检查知识库是否存在
            kb = self.get_kb_by_id(db, kb_id)
            if not kb:
                raise HTTPException(status_code=404, detail="知识库不存在")
            
            # 查找或创建偏好记录
            prefs = db.query(UserKBPrefs).filter(
                and_(
                    UserKBPrefs.user_id == user_id,
                    UserKBPrefs.kb_id == kb_id
                )
            ).first()
            
            if not prefs:
                prefs = UserKBPrefs(
                    user_id=user_id,
                    kb_id=kb_id,
                    pinned=prefs_update.pinned if prefs_update.pinned is not None else False
                )
                db.add(prefs)
            else:
                if prefs_update.pinned is not None:
                    prefs.pinned = prefs_update.pinned
                prefs.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(prefs)
            
            return prefs
            
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"更新用户知识库偏好失败: {e}")
            raise HTTPException(status_code=500, detail=f"更新用户知识库偏好失败: {str(e)}")
    
    def get_user_active_kbs(self, db: Session, user_id: uuid.UUID) -> List[KBModel]:
        """获取用户激活的知识库列表"""
        try:
            # 获取用户置顶的知识库
            pinned_kbs = db.query(KBModel).join(UserKBPrefs).filter(
                and_(
                    UserKBPrefs.user_id == user_id,
                    UserKBPrefs.pinned == True,
                    KBModel.is_active == True
                )
            ).order_by(KBModel.name).all()
            
            # 如果没有置顶的知识库，返回默认知识库
            if not pinned_kbs:
                default_kb = self.get_kb_by_code(db, "DEFAULT")
                if default_kb and default_kb.is_active:
                    return [default_kb]
                else:
                    # 如果默认知识库不存在或不活跃，返回第一个活跃的知识库
                    first_active_kb = db.query(KBModel).filter(
                        KBModel.is_active == True
                    ).order_by(KBModel.created_at).first()
                    return [first_active_kb] if first_active_kb else []
            
            return pinned_kbs
            
        except Exception as e:
            logger.error(f"获取用户激活知识库失败: {e}")
            return []
    
    def move_documents_to_kb(self, 
                            db: Session, 
                            document_ids: List[uuid.UUID], 
                            target_kb_id: uuid.UUID,
                            user_id: uuid.UUID) -> Dict[str, Any]:
        """将文档移动到指定知识库"""
        try:
            # 检查目标知识库是否存在
            target_kb = self.get_kb_by_id(db, target_kb_id)
            if not target_kb:
                raise HTTPException(status_code=404, detail="目标知识库不存在")
            
            # 更新文档的知识库ID
            updated_count = db.query(DocumentModel).filter(
                DocumentModel.id.in_(document_ids)
            ).update(
                {DocumentModel.kb_id: target_kb_id, DocumentModel.updated_at: datetime.utcnow()},
                synchronize_session=False
            )
            
            # 更新相关的向量索引
            from app.models.database import VectorIndex
            db.query(VectorIndex).filter(
                VectorIndex.document_id.in_(document_ids)
            ).update(
                {VectorIndex.kb_id: target_kb_id},
                synchronize_session=False
            )
            
            db.commit()
            
            logger.info(f"用户 {user_id} 将 {updated_count} 个文档移动到知识库 {target_kb.name}")
            
            return {
                'success': True,
                'moved_count': updated_count,
                'target_kb': {
                    'id': target_kb.id,
                    'name': target_kb.name,
                    'code': target_kb.code
                }
            }
            
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"移动文档到知识库失败: {e}")
            raise HTTPException(status_code=500, detail=f"移动文档失败: {str(e)}")
    
    def get_kb_response(self, kb: KBModel, document_count: int = 0) -> KBResponse:
        """转换为响应模型"""
        return KBResponse(
            id=kb.id,
            name=kb.name,
            code=kb.code,
            description=kb.description,
            is_active=kb.is_active,
            visibility=kb.visibility,
            kb_metadata=kb.kb_metadata,
            created_at=kb.created_at,
            updated_at=kb.updated_at,
            document_count=document_count
        )
    
    def get_kb_list_response(self, 
                            kbs: List[KBModel], 
                            total: int, 
                            page: int, 
                            size: int,
                            db: Session) -> KBListResponse:
        """转换为列表响应模型"""
        kb_responses = []
        for kb in kbs:
            doc_count = db.query(DocumentModel).filter(DocumentModel.kb_id == kb.id).count()
            kb_responses.append(self.get_kb_response(kb, doc_count))
        
        return KBListResponse(
            kbs=kb_responses,
            total=total,
            page=page,
            size=size
        )
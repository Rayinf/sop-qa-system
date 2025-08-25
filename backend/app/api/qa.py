from typing import List, Optional
import logging
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import tempfile
import os

from app.core.database import get_db
from app.core.auth import get_current_user, get_current_user_optional
from app.models.database import User
from app.models.schemas import (
    QuestionRequest, AnswerResponse, QALogResponse,
    FeedbackRequest, PaginatedResponse, ModelSwitchRequest
)
from app.services.qa_service import QAService
from app.services.kimi_file_service import KimiFileService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["qa"])
qa_service = QAService()
kimi_file_service = KimiFileService()

@router.post("/switch-model", response_model=dict)
async def switch_model(
    request: ModelSwitchRequest,
    current_user: User = Depends(get_current_user)
):
    """
    切换问答模型
    
    - **model_name**: 模型名称
    - **model_config**: 模型配置参数（可选）
    """
    try:
        success = qa_service.switch_model(
            model_name=request.model_name,
            model_config=request.config
        )
        
        if success:
            return {"message": f"已成功切换到模型: {request.model_name}", "success": True}
        else:
            raise HTTPException(status_code=400, detail="模型切换失败")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模型切换失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/current-model", response_model=dict)
async def get_current_model(
    current_user: User = Depends(get_current_user)
):
    """获取当前使用的模型信息"""
    try:
        current_model = qa_service.get_current_model()
        return {"current_model": current_model}
    except Exception as e:
        logger.error(f"获取当前模型失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-models", response_model=dict)
async def get_available_models(
    current_user: User = Depends(get_current_user)
):
    """获取可用模型列表"""
    try:
        models = qa_service.get_available_models()
        return {"available_models": models}
    except Exception as e:
        logger.error(f"获取可用模型失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask", response_model=AnswerResponse)
def ask_question(
    question_request: QuestionRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    提问接口
    
    - **question**: 用户问题
    - **category**: 文档类别过滤（可选）
    - **session_id**: 会话ID（可选，用于多轮对话）
    - **context**: 上下文信息（可选）
    """
    try:
        # 验证问题长度
        if len(question_request.question.strip()) < 2:
            raise HTTPException(
                status_code=400,
                detail="问题长度至少需要2个字符"
            )
        
        if len(question_request.question) > 500:
            raise HTTPException(
                status_code=400,
                detail="问题长度不能超过500个字符"
            )
        
        # 获取用户ID
        user_id = current_user.id if current_user else None
        
        # 调用问答服务
        answer = qa_service.ask_question(
            db=db,
            question=question_request.question,
            user_id=user_id,
            category=question_request.category,
            session_id=question_request.session_id,
            overrides=question_request.overrides,
            kimi_files=question_request.kimi_files,
            active_kb_ids=question_request.active_kb_ids
        )
        
        logger.info(
            f"问答完成 - 用户: {user_id}, 问题: {question_request.question[:50]}..."
        )
        
        return answer
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"问答失败: {e}")
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")

@router.get("/history", response_model=PaginatedResponse[QALogResponse])
def get_qa_history(
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(20, ge=1, le=100, description="返回的记录数"),
    session_id: Optional[str] = Query(None, description="会话ID过滤"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取问答历史
    
    返回当前用户的问答历史记录
    """
    try:
        # 获取问答历史
        qa_logs = qa_service.get_qa_history(
            db=db,
            user_id=current_user.id,
            session_id=session_id,
            limit=limit + skip  # 简化分页处理
        )
        
        # 应用分页
        paginated_logs = qa_logs[skip:skip + limit]
        total = len(qa_logs)
        
        current_page = skip // limit + 1
        total_pages = (total + limit - 1) // limit
        
        return PaginatedResponse(
            items=paginated_logs,
            total=total,
            page=current_page,
            size=limit,
            pages=total_pages,
            has_next=current_page < total_pages,
            has_prev=current_page > 1
        )
        
    except Exception as e:
        logger.error(f"获取问答历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取问答历史失败: {str(e)}")

@router.get("/history/{qa_log_id}", response_model=QALogResponse)
def get_qa_log(
    qa_log_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取单个问答记录详情
    """
    try:
        from app.models.database import QALog
        
        qa_log = db.query(QALog).filter(
            QALog.id == qa_log_id,
            QALog.user_id == current_user.id
        ).first()
        
        if not qa_log:
            raise HTTPException(status_code=404, detail="问答记录不存在")
        
        return QALogResponse(
            id=qa_log.id,
            question=qa_log.question,
            answer=qa_log.answer,
            user_id=qa_log.user_id,
            session_id=qa_log.session_id,
            retrieved_documents=qa_log.retrieved_documents,
            response_time=qa_log.response_time,
            satisfaction_score=qa_log.satisfaction_score,
            feedback=qa_log.feedback,
            is_helpful=qa_log.is_helpful,
            created_at=qa_log.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取问答记录失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取问答记录失败: {str(e)}")

@router.post("/feedback")
def submit_feedback(
    feedback_request: FeedbackRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    提交问答反馈
    
    - **qa_log_id**: 问答记录ID
    - **score**: 评分（1-5分）
    - **comment**: 反馈评论（可选）
    """
    try:
        # 验证评分范围
        if not 1 <= feedback_request.satisfaction_score <= 5:
            raise HTTPException(
                status_code=400,
                detail="评分必须在1-5之间"
            )
        
        # 验证问答记录是否属于当前用户
        from app.models.database import QALog
        qa_log = db.query(QALog).filter(
            QALog.id == feedback_request.qa_log_id,
            QALog.user_id == current_user.id
        ).first()
        
        if not qa_log:
            raise HTTPException(
                status_code=404,
                detail="问答记录不存在或无权限"
            )
        
        # 提交反馈
        success = qa_service.submit_feedback(
            db=db,
            qa_log_id=feedback_request.qa_log_id,
            score=feedback_request.satisfaction_score,
            comment=feedback_request.feedback,
            is_helpful=feedback_request.is_helpful
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="提交反馈失败")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "反馈提交成功",
                "qa_log_id": str(feedback_request.qa_log_id)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"提交反馈失败: {e}")
        raise HTTPException(status_code=500, detail=f"提交反馈失败: {str(e)}")

@router.get("/statistics")
def get_qa_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取问答统计信息
    
    返回系统整体的问答统计数据
    """
    try:
        stats = qa_service.get_qa_statistics(db)
        return stats
        
    except Exception as e:
        logger.error(f"获取问答统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取问答统计失败: {str(e)}")

@router.get("/statistics/personal")
def get_personal_qa_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取个人问答统计信息
    
    返回当前用户的问答统计数据
    """
    try:
        from app.models.database import QALog
        from sqlalchemy import func, desc
        from datetime import datetime, timedelta
        
        # 基本统计
        total_questions = db.query(QALog).filter(
            QALog.user_id == current_user.id
        ).count()
        
        # 最近7天的问答数
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_questions = db.query(QALog).filter(
            QALog.user_id == current_user.id,
            QALog.created_at >= week_ago
        ).count()
        
        # 平均处理时间
        avg_processing_time = db.query(
            func.avg(QALog.response_time)
        ).filter(
            QALog.user_id == current_user.id
        ).scalar() or 0.0
        
        # 反馈统计
        feedback_stats = db.query(
            QALog.satisfaction_score,
            func.count(QALog.id).label('count')
        ).filter(
            QALog.user_id == current_user.id,
            QALog.satisfaction_score.isnot(None)
        ).group_by(QALog.satisfaction_score).all()
        
        feedback_distribution = {score: count for score, count in feedback_stats}
        
        # 最近的问题
        recent_qa = db.query(QALog).filter(
            QALog.user_id == current_user.id
        ).order_by(desc(QALog.created_at)).limit(5).all()
        
        recent_questions_list = [
            {
                "id": qa.id,
                "question": qa.question[:100] + "..." if len(qa.question) > 100 else qa.question,
                "created_at": qa.created_at.isoformat(),
                "feedback_score": qa.satisfaction_score
            }
            for qa in recent_qa
        ]
        
        return {
            "total_questions": total_questions,
            "recent_questions_count": recent_questions,
            "average_processing_time": round(avg_processing_time, 2),
            "feedback_distribution": feedback_distribution,
            "recent_questions": recent_questions_list,
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取个人问答统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取个人问答统计失败: {str(e)}")

@router.post("/clear-cache")
def clear_qa_cache(
    pattern: Optional[str] = Query("qa_answer:*", description="缓存模式"),
    current_user: User = Depends(get_current_user)
):
    """
    清除问答缓存
    
    需要登录用户权限
    """
    try:
        cleared_count = qa_service.clear_answer_cache(pattern)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"清除了 {cleared_count} 个缓存项",
                "pattern": pattern
            }
        )
        
    except Exception as e:
        logger.error(f"清除缓存失败: {e}")
        raise HTTPException(status_code=500, detail=f"清除缓存失败: {str(e)}")

@router.get("/suggestions")
def get_question_suggestions(
    query: Optional[str] = Query(None, description="查询关键词"),
    limit: int = Query(10, ge=1, le=20, description="返回数量"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    获取问题建议
    
    基于历史问答记录提供问题建议
    """
    try:
        from app.models.database import QALog
        from sqlalchemy import func, desc
        
        # 构建查询
        query_obj = db.query(
            QALog.question,
            func.count(QALog.id).label('count')
        )
        
        # 如果有查询关键词，进行过滤
        if query:
            query_obj = query_obj.filter(
                QALog.question.ilike(f"%{query}%")
            )
        
        # 获取热门问题
        popular_questions = query_obj.group_by(
            QALog.question
        ).order_by(
            desc('count')
        ).limit(limit).all()
        
        suggestions = [
            {
                "question": question,
                "count": count,
                "category": "popular"
            }
            for question, count in popular_questions
        ]
        
        # 如果结果不足，添加一些预设的常见问题
        if len(suggestions) < limit:
            common_questions = [
                "如何执行标准操作流程？",
                "安全注意事项有哪些？",
                "操作步骤是什么？",
                "需要准备哪些材料？",
                "如何处理异常情况？",
                "质量控制要求是什么？",
                "操作完成后需要做什么？",
                "相关规范标准有哪些？"
            ]
            
            remaining = limit - len(suggestions)
            for i, question in enumerate(common_questions[:remaining]):
                suggestions.append({
                    "question": question,
                    "count": 0,
                    "category": "common"
                })
        
        response_data = {
            "query": query,
            "total_suggestions": len(suggestions),
            "suggestions": suggestions
        }
        
        # 创建JSONResponse并添加禁用缓存的头
        response = JSONResponse(content=response_data)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        
        return response
        
    except Exception as e:
        logger.error(f"获取问题建议失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取问题建议失败: {str(e)}")

@router.post("/batch-ask")
def batch_ask_questions(
    questions: List[str],
    background_tasks: BackgroundTasks,
    category: Optional[str] = Query(None, description="文档类别过滤"),
    session_id: Optional[str] = Query(None, description="会话ID"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    批量提问
    
    适用于需要同时处理多个问题的场景
    """
    try:
        # 验证问题数量
        if len(questions) > 10:
            raise HTTPException(
                status_code=400,
                detail="批量问题数量不能超过10个"
            )
        
        # 验证每个问题
        for i, question in enumerate(questions):
            if len(question.strip()) < 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"第{i+1}个问题长度至少需要2个字符"
                )
            
            if len(question) > 500:
                raise HTTPException(
                    status_code=400,
                    detail=f"第{i+1}个问题长度不能超过500个字符"
                )
        
        # 添加到后台任务
        background_tasks.add_task(
            batch_qa_task,
            db,
            questions,
            current_user.id,
            category,
            session_id
        )
        
        return JSONResponse(
            status_code=202,
            content={
                "message": "批量问答任务已启动",
                "question_count": len(questions),
                "session_id": session_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量提问失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量提问失败: {str(e)}")

# 后台任务函数
def batch_qa_task(
    db: Session,
    questions: List[str],
    user_id: int,
    category: Optional[str],
    session_id: Optional[str]
):
    """
    批量问答的后台任务
    """
    try:
        logger.info(f"开始批量问答任务，问题数量: {len(questions)}")
        
        results = []
        for i, question in enumerate(questions):
            try:
                answer = qa_service.ask_question(
                    db=db,
                    question=question,
                    user_id=user_id,
                    category=category,
                    session_id=session_id
                )
                results.append({
                    "question": question,
                    "success": True,
                    "answer": answer.answer.text
                })
                
                logger.info(f"批量问答进度: {i+1}/{len(questions)}")
                
            except Exception as e:
                logger.error(f"批量问答中的问题处理失败: {e}")
                results.append({
                    "question": question,
                    "success": False,
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r['success'])
        logger.info(
            f"批量问答任务完成: 成功 {success_count}/{len(questions)}"
        )
        
    except Exception as e:
        logger.error(f"批量问答任务执行失败: {e}")

@router.get("/vector-logs")
def get_vector_search_logs(
    question: str = Query(..., description="问题内容"),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    获取向量搜索过程的详细日志
    
    返回向量搜索的实时处理日志，包括：
    - 向量数据库状态
    - 搜索参数
    - 搜索结果统计
    - 处理时间
    """
    try:
        # 获取向量服务实例
        vector_service = qa_service.vector_service
        
        # 模拟向量搜索过程并收集日志
        logs = []
        
        # 1. 向量数据库状态检查
        if vector_service.vector_store is None:
            logs.append({
                "timestamp": datetime.utcnow().isoformat(),
                "level": "WARNING",
                "message": "向量数据库未初始化",
                "details": {}
            })
            return {"logs": logs, "status": "error"}
        
        # 2. 向量数据库统计信息
        stats = vector_service.get_vector_store_stats()
        logs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "level": "INFO",
            "message": f"向量数据库状态检查完成",
            "details": {
                "total_vectors": stats.get('total_vectors', 0),
                "vector_dimension": stats.get('vector_dimension', 0),
                "index_size_mb": stats.get('index_size_mb', 0)
            }
        })
        
        # 3. 搜索参数 - 使用配置文件中的值
        from app.core.config import settings
        search_params = {
            "k": settings.retrieval_k,  # 从配置文件获取
            "similarity_threshold": settings.similarity_threshold  # 从配置文件获取
        }
        logs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "level": "INFO",
            "message": "开始向量搜索",
            "details": {
                "query": question[:100] + "..." if len(question) > 100 else question,
                "search_params": search_params
            }
        })
        
        # 4. 执行搜索并记录结果
        try:
            search_results = vector_service.search_similar_documents(
                query=question,
                k=search_params["k"],
                score_threshold=search_params["similarity_threshold"]
            )
            
            logs.append({
                "timestamp": datetime.utcnow().isoformat(),
                "level": "INFO",
                "message": "向量搜索完成",
                "details": {
                    "results_count": len(search_results),
                    "search_successful": True
                }
            })
            
            # 5. 搜索结果详情
            if search_results:
                for i, (doc, score) in enumerate(search_results[:3]):  # 只显示前3个结果
                    logs.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "level": "DEBUG",
                        "message": f"搜索结果 {i+1}",
                        "details": {
                            "document_id": doc.metadata.get('document_id', 'unknown'),
                            "content_preview": doc.page_content[:100] + "...",
                            "metadata_keys": list(doc.metadata.keys()),
                            "similarity_score": float(score)
                        }
                    })
            
        except Exception as search_error:
            logs.append({
                "timestamp": datetime.utcnow().isoformat(),
                "level": "ERROR",
                "message": "向量搜索失败",
                "details": {
                    "error": str(search_error),
                    "search_successful": False
                }
            })
        
        return {
            "logs": logs,
            "status": "success",
            "total_logs": len(logs)
        }
        
    except Exception as e:
        logger.error(f"获取向量搜索日志失败: {e}")
        return {
            "logs": [{
                "timestamp": datetime.utcnow().isoformat(),
                "level": "ERROR",
                "message": "获取向量搜索日志失败",
                "details": {"error": str(e)}
            }],
            "status": "error",
            "total_logs": 1
        }

@router.get("/health")
def qa_health_check():
    """
    问答服务健康检查
    """
    try:
        # 检查问答链是否可用
        qa_chain_status = "available" if qa_service.qa_chain else "unavailable"
        
        # 检查向量数据库状态
        vector_store_status = "available" if qa_service.vector_service.vector_store else "unavailable"
        
        # 检查Redis连接
        redis_status = "available"
        try:
            redis_client = qa_service.get_redis_client()
            if redis_client:
                redis_client.ping()
            else:
                redis_status = "unavailable"
        except Exception:
            redis_status = "unavailable"
        
        overall_status = "healthy" if all([
            qa_chain_status == "available",
            vector_store_status == "available",
            redis_status == "available"
        ]) else "unhealthy"
        
        return {
            "status": overall_status,
            "components": {
                "qa_chain": qa_chain_status,
                "vector_store": vector_store_status,
                "redis": redis_status
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Kimi文件上传相关API端点
@router.post("/kimi/upload-file")
async def upload_file_to_kimi(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    上传文件到Kimi API
    """
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # 上传文件到Kimi
            result = kimi_file_service.upload_and_extract(temp_file_path)
            
            if result:
                return {
                    "success": True,
                    "message": "文件上传成功",
                    "file_info": result["file_info"],
                    "content_preview": result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"]
                }
            else:
                raise HTTPException(status_code=400, detail="文件上传失败")
                
        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/kimi/files")
async def list_kimi_files(
    current_user: User = Depends(get_current_user)
):
    """
    获取Kimi已上传文件列表
    """
    try:
        files = kimi_file_service.list_files()
        return {
            "success": True,
            "files": files,
            "total": len(files)
        }
        
    except Exception as e:
        logger.error(f"获取文件列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/kimi/files/{file_id}")
async def get_kimi_file_info(
    file_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    获取Kimi文件信息
    """
    try:
        file_info = kimi_file_service.get_file_info(file_id)
        
        if file_info:
            return {
                "success": True,
                "file_info": file_info
            }
        else:
            raise HTTPException(status_code=404, detail="文件不存在")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文件信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/kimi/files/{file_id}/content")
async def get_kimi_file_content(
    file_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    获取Kimi文件内容
    """
    try:
        content = kimi_file_service.get_file_content(file_id)
        
        if content:
            return {
                "success": True,
                "content": content
            }
        else:
            raise HTTPException(status_code=404, detail="文件内容获取失败")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文件内容失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/kimi/files/{file_id}")
async def delete_kimi_file(
    file_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    删除Kimi文件
    """
    try:
        success = kimi_file_service.delete_file(file_id)
        
        if success:
            return {
                "success": True,
                "message": "文件删除成功"
            }
        else:
            raise HTTPException(status_code=400, detail="文件删除失败")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文件失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/kimi/batch-upload")
async def batch_upload_files_to_kimi(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    批量上传文件到Kimi API
    """
    try:
        temp_files = []
        results = []
        
        # 保存所有临时文件
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_files.append(temp_file.name)
        
        try:
            # 批量上传
            upload_results = kimi_file_service.upload_multiple_files(temp_files)
            
            for i, result in enumerate(upload_results):
                results.append({
                    "filename": files[i].filename,
                    "success": True,
                    "file_info": result["file_info"],
                    "content_preview": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
                })
            
            # 处理失败的文件
            for i in range(len(upload_results), len(files)):
                results.append({
                    "filename": files[i].filename,
                    "success": False,
                    "error": "上传失败"
                })
            
            return {
                "success": True,
                "message": f"批量上传完成，成功 {len(upload_results)}/{len(files)} 个文件",
                "results": results
            }
            
        finally:
            # 清理所有临时文件
            for temp_file_path in temp_files:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
    except Exception as e:
        logger.error(f"批量上传失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
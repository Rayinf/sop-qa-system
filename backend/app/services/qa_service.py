from typing import List, Optional, Dict, Any, Tuple
import logging
import json
import uuid
from datetime import datetime, timezone

# from langchain_openai import OpenAI
# from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.database import QALog, User, VectorIndex
from app.models.schemas import (
    QuestionRequest, AnswerResponse, SourceDocument,
    FormattedAnswer, QALogResponse
)
from app.services.vector_service import VectorService
from app.services.advanced_retriever import AdvancedRetrieverService
from app.services.retrieval.unified_retrieval_service import UnifiedRetrievalService
from app.services.retrieval.config_factory import RetrievalConfigFactory
from app.core.database import get_redis_client, get_db
from app.services.document_processor import DocumentProcessor  # 新增导入
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

class QAService:
    """智能问答服务类"""
    
    def __init__(self, model_name: Optional[str] = None):
        # 使用指定模型或默认模型
        self.current_model = model_name or settings.default_model
        
        # 初始化LLM模型
        self.llm_service = LLMService(self.current_model)
        self.llm = self.llm_service.get_llm()
        
        # 向量服务
        self.vector_service = VectorService.get_instance()
        
        # 高级检索器服务（保留兼容性）
        self.advanced_retriever = None
        try:
            self.advanced_retriever = AdvancedRetrieverService(llm=self.llm)
            logger.info("高级检索器服务初始化成功")
        except Exception as e:
            logger.warning(f"高级检索器服务初始化失败: {e}")
        

        
        # 初始化统一检索服务
        try:
            config = RetrievalConfigFactory.create_from_settings()
            self.unified_retrieval_service = UnifiedRetrievalService(config)
            logger.info("✅ 统一检索服务初始化完成")
        except Exception as e:
            logger.error(f"❌ 统一检索服务初始化失败: {e}")
            self.unified_retrieval_service = None
        
        # Redis客户端
        self.redis_client = None
        
        # 问答链
        self._qa_chain = None
        # 记录已知的向量库版本（用于在版本变化时重建问答链）
        self._vector_store_version_seen = None
        
        # 初始化提示模板
        self.setup_prompts()
    
    def _create_llm(self, model_name: str, model_config: Optional[Dict[str, Any]] = None):
        """创建LLM实例（通过LLMService）"""
        service = LLMService(model_name)
        # 如需自定义配置，可在LLMService.switch_model中处理
        return service.get_llm()
    
    def switch_model(self, model_name: str, model_config: Optional[Dict[str, Any]] = None) -> bool:
        """切换模型"""
        try:
            available_models = settings.available_models.split(',')
            if model_name not in available_models:
                logger.error(f"模型 {model_name} 不在可用模型列表中: {available_models}")
                return False
            
            self.current_model = model_name
            if not hasattr(self, 'llm_service') or self.llm_service is None:
                self.llm_service = LLMService(model_name)
            else:
                self.llm_service.switch_model(model_name, model_config)
            self.llm = self.llm_service.get_llm()
            # 重置问答链，使其使用新模型
            self._qa_chain = None
            logger.info(f"已切换到模型: {model_name}")
            return True
        except Exception as e:
            logger.error(f"切换模型失败: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        return settings.available_models.split(',')
    
    def get_current_model(self) -> str:
        """获取当前使用的模型"""
        return self.current_model
    
    def get_redis_client(self):
        """获取Redis客户端"""
        if self.redis_client is None:
            self.redis_client = get_redis_client()
        return self.redis_client

    def _get_current_vector_store_version(self) -> int:
        """获取当前向量库版本号（来自Redis），默认为0"""
        try:
            client = self.get_redis_client()
            if client:
                value = client.get("vector_store:version")
                if value is None:
                    return 0
                try:
                    return int(value)
                except Exception:
                    try:
                        return int(value.decode())
                    except Exception:
                        return 0
            return 0
        except Exception as e:
            logger.warning(f"获取向量库版本失败: {e}")
            return 0
    
    def setup_prompts(self):
        """设置提示模板"""
        # 通用知识库问答提示模板
        self.qa_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
你是一个专业的知识库智能助手。请基于以下提供的文档内容，准确回答用户的问题。

重要指导原则：
1. 严格基于提供的文档内容回答，不要添加文档中没有的信息
2. 如果文档中没有相关信息，请明确说明"根据提供的文档，没有找到相关信息"
3. 回答要准确、具体、有用
4. 如果涉及步骤或流程，请按顺序列出
5. 如果有重要注意事项，请特别强调
6. 保持专业和友好的语调
7. 如果问题涉及列表或分类信息，请仔细整合所有文档片段中的相关信息，确保回答完整
8. 对于列表类问题，请逐一检查所有文档片段，避免遗漏任何项目

文档内容：
{context}

用户问题：{question}

请提供详细、准确的答案：
"""
        )
        
        # 多轮对话提示模板
        self.chat_prompt_template = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template="""
你是一个专业的知识库智能助手，正在与用户进行多轮对话。请基于文档内容和对话历史，回答用户的问题。

文档内容：
{context}

对话历史：
{chat_history}

当前问题：{question}

请提供准确、有用的回答：
"""
        )
    
    @property
    def qa_chain(self) -> Optional[RetrievalQA]:
        """获取问答链"""
        try:
            current_version = self._get_current_vector_store_version()
            if self._vector_store_version_seen is None:
                self._vector_store_version_seen = current_version
            elif current_version != self._vector_store_version_seen:
                logger.info(
                    f"检测到向量库版本变化: {self._vector_store_version_seen} -> {current_version}，重建问答链"
                )
                self._qa_chain = None
                self._vector_store_version_seen = current_version
        except Exception as e:
            logger.warning(f"检查向量库版本失败: {e}")
        if self._qa_chain is None:
            self._qa_chain = self.create_qa_chain()
        return self._qa_chain
    
    def create_qa_chain(self) -> Optional[RetrievalQA]:
        """创建问答链"""
        try:
            if self.vector_service.vector_store is None:
                logger.warning("向量数据库未初始化，无法创建问答链")
                return None
            
            # 选择检索器
            # 如果有统一检索服务，优先使用基础检索器，避免与统一检索服务冲突
            if self.unified_retrieval_service:
                # 使用基础检索器，让统一检索服务处理高级检索逻辑
                retriever = self.vector_service.vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "k": settings.retrieval_k,
                        "score_threshold": 0.1
                    }
                )
                logger.info("使用基础检索器（统一检索服务可用）")
            elif self.advanced_retriever:
                # 使用高级检索器
                try:
                    retriever = self.advanced_retriever.create_ensemble_retriever()
                    logger.info("使用高级Ensemble检索器")
                except Exception as e:
                    logger.warning(f"高级检索器失败，回退到基础检索器: {e}")
                    retriever = self.vector_service.vector_store.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={
                            "k": settings.retrieval_k,
                            "score_threshold": 0.1
                        }
                    )
            else:
                # 使用基础检索器
                retriever = self.vector_service.vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "k": settings.retrieval_k,
                        "score_threshold": 0.1  # 降低阈值以包含更多相关文档
                    }
                )
                logger.info("使用基础检索器（带相似度阈值）")
            
            # 创建问答链
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={
                    "prompt": self.qa_prompt_template
                },
                return_source_documents=True
            )
            
            logger.info("问答链创建成功")
            return qa_chain
            
        except Exception as e:
            logger.error(f"记录问答日志失败: {e}")
            db.rollback()
            return None
    
    def ask_question(self, 
                    db: Session,
                    question: str,
                    user_id: Optional[int] = None,
                    category: Optional[str] = None,
                    session_id: Optional[str] = None,
                    overrides: Optional[Dict[str, Any]] = None,
                    kimi_files: Optional[List[str]] = None,
                    active_kb_ids: Optional[List[uuid.UUID]] = None) -> AnswerResponse:
        """回答问题"""
        try:
            start_time = datetime.now(timezone.utc)
            logger.info(f"🤖 开始处理问答请求: '{question[:100]}{'...' if len(question) > 100 else ''}'") 
            logger.info(f"📋 请求参数 - 用户ID: {user_id}, 类别: {category}, 会话ID: {session_id}")
            
            # 检查是否使用Kimi模型且有文件
            if kimi_files and len(kimi_files) > 0 and 'kimi' in self.current_model.lower():
                logger.info(f"🔍 检测到Kimi模型文件问答: {len(kimi_files)} 个文件")
                return self._handle_kimi_file_question(db, question, kimi_files, user_id, session_id, start_time)
            
            # 检查缓存
            logger.info("🔍 步骤1: 检查答案缓存")
            cached_answer = self.get_cached_answer(question, category)
            if cached_answer:
                logger.info("💾 返回缓存的答案")
                return self.create_answer_response(
                    question=question,
                    answer=cached_answer['answer'],
                    source_documents=cached_answer['sources'],
                    processing_time=0.1,
                    from_cache=True,
                    session_id=session_id,
                    metadata={
                        "retrieval_method": "cache",
                        "category": category
                    }
                )
            else:
                logger.info("❌ 未找到缓存答案，继续处理")
            
            # 特殊处理相关方问题
            logger.info("🔍 步骤2: 检查问题类型")
            if self._is_stakeholder_question(question):
                logger.info("👥 检测到相关方问题，使用专门处理逻辑")
                return self._handle_stakeholder_question(db, question, user_id, session_id, start_time)
            else:
                logger.info("📝 检测到普通问题，使用标准处理逻辑")
            
            # 检查问答链是否可用
            logger.info("🔍 步骤3: 检查问答链状态")
            if self.qa_chain is None:
                logger.error("❌ 问答链未初始化")
                return self.create_error_response(
                    "问答系统暂时不可用，请稍后再试"
                )
            logger.info("✅ 问答链状态正常")
            
            # 构建查询
            logger.info("🔍 步骤4: 构建增强查询")
            query = self.enhance_query(question, category)
            logger.info(f"📝 增强查询: '{query[:100]}{'...' if len(query) > 100 else ''}'")
            
            # 选择检索策略
            if active_kb_ids:
                logger.info("🔍 步骤5: 检测到指定知识库ID，优先走支持知识库过滤的检索路径")
                if self.unified_retrieval_service:
                    logger.info("🔍 使用统一检索服务（带知识库过滤）执行问答")
                    return self.ask_question_with_unified_retrieval(db, question, query, user_id, session_id, start_time, category, overrides, active_kb_ids)
                else:
                    logger.info("🔍 使用传统向量服务（带知识库过滤）执行问答")
                    # 使用向量服务的知识库过滤功能
                    kb_ids_str = [str(kb_id) for kb_id in active_kb_ids]
                    docs_with_scores = self.vector_service.search_similar_documents(
                        query=query,
                        k=settings.retrieval_k,
                        active_kb_ids=kb_ids_str
                    )
                    source_docs = [doc for doc, score in docs_with_scores]
                    
                    # 如果找到相关文档，使用LLM生成答案
                    if source_docs:
                        context = "\n\n".join([doc.page_content for doc in source_docs])
                        prompt = f"""基于以下上下文回答问题：

上下文：
{context}

问题：{question}

请提供准确、详细的答案："""
                        
                        try:
                            answer = self.llm.invoke(prompt).content if hasattr(self.llm.invoke(prompt), 'content') else str(self.llm.invoke(prompt))
                        except Exception as e:
                            logger.error(f"LLM调用失败: {e}")
                            answer = "抱歉，在生成答案时遇到了问题。"
                        
                        result = {
                            "result": answer,
                            "source_documents": source_docs
                        }
                    else:
                        logger.warning("在指定知识库中未找到相关文档")
                        result = {
                            "result": "抱歉，在指定的知识库中没有找到相关信息来回答您的问题。",
                            "source_documents": []
                        }
                    # 继续后续处理
                # 跳过多路由，因为其暂不支持 active_kb_ids
            elif self.unified_retrieval_service:
                logger.info("🔍 步骤5: 使用统一检索服务执行问答")
                return self.ask_question_with_unified_retrieval(db, question, query, user_id, session_id, start_time, category, overrides, active_kb_ids)
            else:
                logger.info("🔍 步骤5: 使用传统问答链执行向量搜索")
                # 如果指定了知识库ID，使用向量服务直接搜索并构建答案
                if active_kb_ids:
                    logger.info(f"🎯 限制搜索范围到知识库: {active_kb_ids}")
                    # 使用向量服务的知识库过滤功能
                    kb_ids_str = [str(kb_id) for kb_id in active_kb_ids]
                    docs_with_scores = self.vector_service.search_similar_documents(
                        query=query,
                        k=settings.retrieval_k,
                        active_kb_ids=kb_ids_str
                    )
                    source_docs = [doc for doc, score in docs_with_scores]
                    
                    # 如果找到相关文档，使用LLM生成答案
                    if source_docs:
                        context = "\n\n".join([doc.page_content for doc in source_docs])
                        prompt = f"""基于以下上下文回答问题：

上下文：
{context}

问题：{question}

请提供准确、详细的答案："""
                        
                        try:
                            answer = self.llm.invoke(prompt).content if hasattr(self.llm.invoke(prompt), 'content') else str(self.llm.invoke(prompt))
                        except Exception as e:
                            logger.error(f"LLM调用失败: {e}")
                            answer = "抱歉，在生成答案时遇到了问题。"
                        
                        result = {
                            "result": answer,
                            "source_documents": source_docs
                        }
                    else:
                        logger.warning("在指定知识库中未找到相关文档")
                        result = {
                            "result": "抱歉，在指定的知识库中没有找到相关信息来回答您的问题。",
                            "source_documents": []
                        }
                else:
                    # 使用传统问答链
                    result = self.qa_chain({"query": query})
            
            source_docs = result.get("source_documents", [])
            logger.info(f"📊 向量搜索完成: {len(source_docs)} 个相关文档")
            
            # 记录找到的文档详情
            for i, doc in enumerate(source_docs[:3]):  # 只记录前3个
                doc_id = doc.metadata.get('document_id', 'unknown')
                content_preview = doc.page_content[:80].replace('\n', ' ')
                logger.debug(f"📄 文档 {i+1}: {doc_id} - '{content_preview}...'")
            
            # 处理结果
            logger.info("🔄 步骤6: 处理搜索结果")
            answer = result.get("result", "")
            logger.info(f"📝 原始答案长度: {len(answer)} 字符")
            
            # 后处理答案
            logger.info("🔄 步骤7: 后处理答案")
            formatted_answer = self.post_process_answer(answer, source_docs)
            logger.info(f"📝 格式化答案长度: {len(formatted_answer)} 字符")
            
            # 转换源文档
            logger.info("🔄 步骤8: 转换源文档格式")
            source_documents = self.convert_source_documents(source_docs)
            logger.info(f"📄 成功转换 {len(source_documents)} 个源文档")
            
            # 计算处理时间
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(f"⏱️ 当前处理时间: {processing_time:.2f}秒")
            
            # 缓存答案
            logger.info("💾 步骤9: 缓存答案")
            self.cache_answer(question, category, formatted_answer, source_documents)
            
            # 记录问答日志
            logger.info("📝 步骤10: 记录问答交互日志")
            qa_log = self.log_qa_interaction(
                db=db,
                question=question,
                answer=formatted_answer,
                source_documents=source_documents,
                user_id=user_id,
                session_id=session_id,
                processing_time=processing_time
            )
            
            # 创建响应
            logger.info("🔄 步骤11: 创建响应对象")
            response = self.create_answer_response(
                question=question,
                answer=formatted_answer,
                source_documents=source_documents,
                processing_time=processing_time,
                session_id=session_id,
                metadata={
                    "retrieval_method": "legacy_retrieval",
                    "category": category
                },
                qa_log_id=qa_log.id if qa_log else None
            )
            
            logger.info(f"✅ 问答处理完成，总处理时间: {processing_time:.2f}秒")
            return response
            
        except Exception as e:
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.error(f"❌ 问答处理失败: {e} (耗时: {processing_time:.2f}秒)")
            return self.create_error_response(f"问答失败: {str(e)}")
    
    def ask_question_with_unified_retrieval(
         self,
         db: Session,
         question: str,
         query: str,
         user_id: Optional[int] = None,
         session_id: Optional[str] = None,
         start_time: Optional[datetime] = None,
         category: Optional[str] = None,
         overrides: Optional[Dict[str, Any]] = None,
         active_kb_ids: Optional[List[uuid.UUID]] = None
     ) -> AnswerResponse:
         """
         使用统一检索服务回答问题
         """
         try:
             if start_time is None:
                 start_time = datetime.now(timezone.utc)
             
             logger.info("🚀 开始统一检索问答")
             
             # 使用统一检索服务获取相关文档
             retrieval_params = overrides.copy() if overrides else {}
             if active_kb_ids:
                 retrieval_params['active_kb_ids'] = active_kb_ids
                 logger.info(f"🎯 限制检索范围到知识库: {active_kb_ids}")
             
             retrieval_result = self.unified_retrieval_service.retrieve(
                 query=query,
                 category=category,
                 **retrieval_params
             )
             
             # 提取检索结果
             source_docs = retrieval_result.documents
             retrieval_metadata = retrieval_result.metadata or {}
             retrieval_mode = retrieval_metadata.get("retrieval_mode", "unknown")
             processing_time_retrieval = retrieval_result.processing_time or 0
             
             logger.info(f"🎯 检索策略: {retrieval_mode}")
             logger.info(f"📄 找到 {len(source_docs)} 个相关文档")
             logger.info(f"⏱️ 检索耗时: {processing_time_retrieval:.3f}秒")
             
             # 使用LLM生成答案
             if source_docs:
                 # 构建上下文
                 context = "\n\n".join([doc.page_content for doc in source_docs[:5]])
                 
                 # 使用问答链生成答案
                 qa_input = {
                     "query": query,
                     "context": context
                 }
                 
                 # 如果有问答链，使用它生成答案
                 if self.qa_chain:
                     # 临时设置检索器为返回固定文档的检索器
                     original_retriever = self.qa_chain.retriever
                     
                     class FixedRetriever:
                        def get_relevant_documents(self, query, *, callbacks=None, **kwargs):
                            # 兼容LangChain Retrieval接口可能传入的callbacks或其他可选参数
                            return source_docs

                        async def aget_relevant_documents(self, query, *, callbacks=None, **kwargs):
                            # 提供异步接口以兼容可能的异步调用
                            return source_docs
                    
                     self.qa_chain.retriever = FixedRetriever()
                     result = self.qa_chain({"query": query})
                     answer = result.get("result", "")
                     
                     # 恢复原始检索器
                     self.qa_chain.retriever = original_retriever
                 else:
                     # 直接使用LLM生成答案
                     prompt = f"""基于以下上下文回答问题：

上下文：
{context}

问题：{query}

请提供准确、详细的答案："""
                     answer = self.llm.predict(prompt)
             else:
                 answer = "抱歉，没有找到相关信息来回答您的问题。"
             
             # 后处理答案
             formatted_answer = self.post_process_answer(answer, source_docs)
             
             # 转换源文档格式
             source_documents = self.convert_source_documents(source_docs)
             
             # 计算总处理时间
             total_processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
             
             # 缓存答案
             self.cache_answer(question, category, formatted_answer, source_documents)
             
             # 先创建QA日志记录，但不提交
             qa_log = None
             try:
                 # 序列化source_documents，确保UUID对象被转换为字符串
                 def serialize_value(value):
                     """递归序列化值，确保UUID和其他非JSON兼容类型被转换为字符串"""
                     if isinstance(value, uuid.UUID):
                         return str(value)
                     elif isinstance(value, dict):
                         return {k: serialize_value(v) for k, v in value.items()}
                     elif isinstance(value, list):
                         return [serialize_value(item) for item in value]
                     elif hasattr(value, '__dict__'):
                         # 对于复杂对象，尝试转换为字符串
                         return str(value)
                     else:
                         return value
                 
                 serialized_docs = []
                 for doc in source_documents:
                     doc_dict = doc.dict()
                     # 递归序列化整个文档字典
                     serialized_doc = serialize_value(doc_dict)
                     serialized_docs.append(serialized_doc)
                 
                 # 创建包含检索信息的元数据
                 log_metadata = {
                     'model': settings.llm_model,
                     'temperature': settings.llm_temperature,
                     'retrieval_k': settings.retrieval_k,
                     'timestamp': datetime.now(timezone.utc).isoformat(),
                     "retrieval_info": {
                        "mode": retrieval_mode,
                        "retrieval_time": processing_time_retrieval,
                        "documents_found": len(source_docs),
                        "auto_selected": retrieval_metadata.get("auto_selected", False)
                    }
                 }
                 
                 qa_log = QALog(
                     question=question,
                     answer=formatted_answer,
                     user_id=user_id,
                     session_id=session_id,
                     retrieved_documents=serialized_docs,
                     response_time=total_processing_time,
                     log_metadata=log_metadata
                 )
                 
                 db.add(qa_log)
                 db.flush()  # 刷新以获取ID，但不提交事务
                 
                 logger.info(f"QA日志已添加到会话，ID: {qa_log.id}")
                 
             except Exception as e:
                 logger.error(f"记录问答日志失败: {e}")
                 qa_log = None
                 # 不在这里处理回滚，让API层统一处理事务
             
             # 创建响应
             response = self.create_answer_response(
                 question=question,
                 answer=formatted_answer,
                 source_documents=source_documents,
                 processing_time=total_processing_time,
                 session_id=session_id,
                 metadata={
                     "retrieval_mode": retrieval_mode,
                     "retrieval_method": "unified_retrieval",
                     "retrieval_time": processing_time_retrieval,
                     "auto_selected": retrieval_metadata.get("auto_selected", False),
                     "category": category
                 },
                 qa_log_id=qa_log.id if qa_log else None
             )
             
             logger.info(f"✅ 统一检索问答完成，总处理时间: {total_processing_time:.2f}秒")
             return response
             
         except Exception as e:
             processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() if start_time else 0
             logger.error(f"❌ 统一检索问答失败: {e} (耗时: {processing_time:.2f}秒)")
             return self.create_error_response(f"统一检索问答失败: {str(e)}")
    

    def enhance_query(self, question: str, category: Optional[str] = None) -> str:
        """增强查询"""
        enhanced_query = question
        
        # 添加类别信息
        if category:
            enhanced_query = f"[{category}] {question}"
        
        # 添加通用关键词增强
        general_keywords = ["步骤", "流程", "方法", "标准", "规范", "指南"]
        if not any(keyword in question for keyword in general_keywords):
            enhanced_query = f"相关信息：{enhanced_query}"
        
        return enhanced_query
    
    def post_process_answer(self, answer: str, source_docs: List[Document]) -> str:
        """后处理答案"""
        try:
            # 清理答案格式
            answer = answer.strip()
            
            # 添加置信度信息
            if len(source_docs) == 0:
                answer += "\n\n⚠️ 注意：此回答基于有限的文档信息，建议进一步确认。"
            elif len(source_docs) < 2:
                answer += "\n\n💡 提示：此回答基于少量文档信息，可能不够全面。"
            
            # 添加相关文档数量信息
            if source_docs:
                answer += f"\n\n📚 参考了 {len(source_docs)} 个相关文档片段"
            
            return answer
            
        except Exception as e:
            logger.error(f"后处理答案失败: {e}")
            return answer
    
    def convert_source_documents(self, source_docs: List[Document]) -> List[SourceDocument]:
        """转换源文档格式并去重"""
        converted_docs = []
        seen_docs = set()  # 用于去重的集合
        
        for doc in source_docs:
            try:
                # 先对 metadata 做安全序列化处理，避免 numpy、UUID 等类型导致 JSON 编码错误
                sanitized_metadata = DocumentProcessor._sanitize_metadata(doc.metadata)
                
                # 生成文档唯一标识用于去重
                doc_id = self._get_document_unique_id(doc, sanitized_metadata)
                
                # 如果已经存在相同的文档，跳过
                if doc_id in seen_docs:
                    logger.debug(f"跳过重复文档: {doc_id}")
                    continue
                
                seen_docs.add(doc_id)
                
                # 提取并转换相似度分数，确保为 float 基础类型
                raw_score = sanitized_metadata.get('score', 0.0)
                try:
                    similarity_score = float(raw_score)
                except Exception:
                    similarity_score = 0.0
                
                # 提取文档标题和来源信息（优先使用更能区分的字段）
                display_title = (
                    sanitized_metadata.get('document_title')
                    or sanitized_metadata.get('filename')
                    or sanitized_metadata.get('title')
                    or '未知文档'
                )
                # Excel/多表格类文档，附加工作表与批次信息，避免标题重复
                sheet_name = sanitized_metadata.get('sheet_name')
                if sheet_name:
                    display_title = f"{display_title} - {sheet_name}"
                if sanitized_metadata.get('batch_type') == 'data':
                    bs = sanitized_metadata.get('batch_start')
                    be = sanitized_metadata.get('batch_end')
                    if bs is not None and be is not None:
                        display_title = f"{display_title} [{bs}-{be}]"
                
                source = (
                    sanitized_metadata.get('source')
                    or sanitized_metadata.get('filename')
                    or '未知来源'
                )
                page_no = sanitized_metadata.get('page_number')
                if page_no is None:
                    page_no = sanitized_metadata.get('page')
                
                source_doc = SourceDocument(
                    content=doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    source=source,
                    title=display_title,
                    metadata=sanitized_metadata,
                    similarity_score=similarity_score,
                    document_id=sanitized_metadata.get('document_id'),
                    page_number=page_no,
                    kb_id=sanitized_metadata.get('kb_id'),
                    kb_name=sanitized_metadata.get('kb_name')
                )
                converted_docs.append(source_doc)
            except Exception as e:
                logger.error(f"转换源文档失败: {e}")
                continue
        
        logger.info(f"源文档去重完成: 原始{len(source_docs)}个，去重后{len(converted_docs)}个")
        return converted_docs

    def _get_document_unique_id(self, doc: Document, metadata: Dict[str, Any]) -> str:
        """生成文档唯一标识用于去重"""
        try:
            # 优先使用chunk_id（最精确的标识）
            if 'chunk_id' in metadata:
                return f"chunk_{metadata['chunk_id']}"
            
            # 其次使用document_id + chunk_index组合
            if 'document_id' in metadata and 'chunk_index' in metadata:
                return f"doc_{metadata['document_id']}_chunk_{metadata['chunk_index']}"
            
            # 使用document_id + page_number组合
            if 'document_id' in metadata and 'page_number' in metadata:
                return f"doc_{metadata['document_id']}_page_{metadata['page_number']}"
            
            # 使用source + title + page_number组合
            if 'source' in metadata and 'title' in metadata and 'page_number' in metadata:
                return f"source_{hash(metadata['source'])}_{hash(metadata['title'])}_page_{metadata['page_number']}"
            
            # 最后使用内容哈希（确保唯一性）
            import hashlib
            content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()[:16]
            return f"content_{content_hash}"
            
        except Exception as e:
            logger.warning(f"生成文档唯一标识失败: {e}")
            # 降级到对象ID
            return f"fallback_{id(doc)}"

    def create_answer_response(self, 
                              question: str,
                              answer: str,
                              source_documents: List[SourceDocument],
                              processing_time: float,
                              from_cache: bool = False,
                              session_id: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None,
                              qa_log_id: Optional[uuid.UUID] = None) -> AnswerResponse:
        """创建答案响应"""
        formatted_answer = FormattedAnswer(
            summary=answer,
            steps=[],
            warnings=[],
            references=[]
        )
        
        # 计算置信度
        confidence = self.calculate_confidence_score(answer, source_documents)
        
        return AnswerResponse(
            question=question,
            answer=answer,
            formatted_answer=formatted_answer,
            source_documents=source_documents,
            confidence=confidence,
            processing_time=processing_time,
            from_cache=from_cache,
            session_id=session_id,
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc),
            qa_log_id=qa_log_id
        )
    
    def create_error_response(self, error_message: str) -> AnswerResponse:
        """创建错误响应"""
        formatted_answer = FormattedAnswer(
            summary=f"抱歉，{error_message}",
            steps=[],
            warnings=[],
            references=[]
        )
        
        return AnswerResponse(
            question="",
            answer=error_message,
            formatted_answer=formatted_answer,
            source_documents=[],
            processing_time=0.0,
            created_at=datetime.now(timezone.utc)
        )
    
    def calculate_confidence_score(self, 
                                  answer: str, 
                                  source_documents: List[SourceDocument]) -> float:
        """计算置信度分数"""
        try:
            # 基础分数
            base_score = 0.5
            
            # 根据源文档数量调整
            doc_count_bonus = min(len(source_documents) * 0.1, 0.3)
            
            # 根据答案长度调整
            length_bonus = min(len(answer) / 1000, 0.1)
            
            # 根据关键词匹配调整
            keyword_bonus = 0.0
            general_keywords = ["步骤", "流程", "方法", "注意", "要求", "说明", "介绍"]
            for keyword in general_keywords:
                if keyword in answer:
                    keyword_bonus += 0.02
            
            # 计算最终分数
            confidence_score = base_score + doc_count_bonus + length_bonus + keyword_bonus
            
            # 限制在0-1范围内
            return min(max(confidence_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"计算置信度失败: {e}")
            return 0.5
    
    def log_qa_interaction(self, 
                          db: Session,
                          question: str,
                          answer: str,
                          source_documents: List[SourceDocument],
                          user_id: Optional[int] = None,
                          session_id: Optional[str] = None,
                          processing_time: float = 0.0) -> Optional[QALog]:
        """记录问答交互"""
        try:
            # 序列化source_documents，确保UUID对象被转换为字符串
            serialized_docs = []
            for doc in source_documents:
                doc_dict = doc.dict()
                # 确保document_id被转换为字符串
                if 'document_id' in doc_dict and doc_dict['document_id'] is not None:
                    doc_dict['document_id'] = str(doc_dict['document_id'])
                # 确保metadata中的UUID也被转换为字符串
                if 'metadata' in doc_dict and isinstance(doc_dict['metadata'], dict):
                    for key, value in doc_dict['metadata'].items():
                        if hasattr(value, '__str__') and 'UUID' in str(type(value)):
                            doc_dict['metadata'][key] = str(value)
                serialized_docs.append(doc_dict)
            
            qa_log = QALog(
                question=question,
                answer=answer,
                user_id=user_id,
                session_id=session_id,
                retrieved_documents=serialized_docs,
                response_time=processing_time,
                log_metadata={
                    'model': settings.llm_model,
                    'temperature': settings.llm_temperature,
                    'retrieval_k': settings.retrieval_k,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            
            db.add(qa_log)
            logger.info("QA日志已添加到会话，准备提交")
            
            db.commit()
            logger.info("QA日志已提交到数据库")
            
            db.refresh(qa_log)
            logger.info(f"问答日志记录成功: {qa_log.id}")
            return qa_log
            
        except Exception as e:
            logger.error(f"记录问答日志失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            db.rollback()
            return None
    
    def get_qa_history(self, 
                      db: Session,
                      user_id: Optional[int] = None,
                      session_id: Optional[str] = None,
                      limit: int = 20) -> List[QALogResponse]:
        """获取问答历史"""
        try:
            query = db.query(QALog)
            
            if user_id:
                query = query.filter(QALog.user_id == user_id)
            
            if session_id:
                query = query.filter(QALog.session_id == session_id)
            
            qa_logs = query.order_by(QALog.created_at.desc()).limit(limit).all()
            
            return [
                QALogResponse(
                    id=log.id,
                    question=log.question,
                    answer=log.answer,
                    user_id=log.user_id,
                    session_id=log.session_id,
                    formatted_answer=log.formatted_answer,
                    retrieved_documents=log.retrieved_documents,
                    response_time=log.response_time,
                    satisfaction_score=log.satisfaction_score or 0,
                    feedback=log.feedback,
                    is_helpful=log.is_helpful,
                    created_at=log.created_at
                )
                for log in qa_logs
            ]
            
        except Exception as e:
            logger.error(f"获取问答历史失败: {e}")
            return []
    
    def submit_feedback(self, 
                       db: Session,
                       qa_log_id: uuid.UUID,
                       score: int,
                       comment: Optional[str] = None,
                       is_helpful: Optional[bool] = None) -> bool:
        """提交反馈"""
        try:
            qa_log = db.query(QALog).filter(QALog.id == qa_log_id).first()
            if not qa_log:
                return False
            
            qa_log.satisfaction_score = score
            qa_log.feedback = comment
            if is_helpful is not None:
                qa_log.is_helpful = is_helpful
            
            db.commit()
            
            logger.info(f"反馈提交成功: QA日志 {qa_log_id}, 评分 {score}")
            return True
            
        except Exception as e:
            logger.error(f"提交反馈失败: {e}")
            db.rollback()
            return False
    
    # 缓存相关方法
    def cache_answer(self, 
                    question: str, 
                    category: Optional[str],
                    answer: str, 
                    source_documents: List[SourceDocument]):
        """缓存答案"""
        try:
            redis_client = self.get_redis_client()
            if redis_client:
                cache_key = self.get_cache_key(question, category)
                cache_data = {
                    'answer': answer,
                    'sources': [doc.dict() for doc in source_documents],
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                redis_client.setex(
                    cache_key, 
                    settings.answer_cache_ttl, 
                    json.dumps(cache_data, ensure_ascii=False)
                )
                
        except Exception as e:
            logger.warning(f"缓存答案失败: {e}")
    
    def get_cached_answer(self, 
                         question: str, 
                         category: Optional[str]) -> Optional[Dict[str, Any]]:
        """获取缓存的答案"""
        try:
            redis_client = self.get_redis_client()
            if redis_client:
                cache_key = self.get_cache_key(question, category)
                cached_data = redis_client.get(cache_key)
                
                if cached_data:
                    data = json.loads(cached_data.decode())
                    # 转换源文档
                    data['sources'] = [
                        SourceDocument(**doc) for doc in data['sources']
                    ]
                    return data
            
            return None
            
        except Exception as e:
            logger.warning(f"获取缓存答案失败: {e}")
            return None
    
    def get_cache_key(self, question: str, category: Optional[str]) -> str:
        """生成缓存键"""
        import hashlib
        
        key_parts = [question.lower().strip()]
        if category:
            key_parts.append(category.lower())
        
        key_string = "|".join(key_parts)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"qa_answer:{key_hash}"
    
    def clear_answer_cache(self, pattern: str = "qa_answer:*") -> int:
        """清除答案缓存"""
        try:
            redis_client = self.get_redis_client()
            if redis_client:
                keys = redis_client.keys(pattern)
                if keys:
                    deleted_count = redis_client.delete(*keys)
                    logger.info(f"清除了 {deleted_count} 个缓存答案")
                    return deleted_count
            return 0
            
        except Exception as e:
            logger.error(f"清除答案缓存失败: {e}")
            return 0
    
    def get_qa_statistics(self, db: Session) -> Dict[str, Any]:
        """获取问答统计信息"""
        try:
            from sqlalchemy import func, desc
            
            # 基本统计
            total_questions = db.query(QALog).count()
            
            # 今日问答数
            today = datetime.now(timezone.utc).date()
            today_questions = db.query(QALog).filter(
                func.date(QALog.created_at) == today
            ).count()
            
            # 平均处理时间
            avg_processing_time = db.query(
                func.avg(QALog.response_time)
            ).scalar() or 0.0
            
            # 反馈统计
            feedback_stats = db.query(
                QALog.satisfaction_score,
                func.count(QALog.id).label('count')
            ).filter(
                QALog.satisfaction_score.isnot(None)
            ).group_by(QALog.satisfaction_score).all()
            
            feedback_distribution = {score: count for score, count in feedback_stats}
            
            # 热门问题
            popular_questions = db.query(
                QALog.question,
                func.count(QALog.id).label('count')
            ).group_by(
                QALog.question
            ).order_by(
                desc('count')
            ).limit(10).all()
            
            return {
                'total_questions': total_questions,
                'today_questions': today_questions,
                'average_processing_time': round(avg_processing_time, 2),
                'feedback_distribution': feedback_distribution,
                'popular_questions': [
                    {'question': q, 'count': c} for q, c in popular_questions
                ],
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取问答统计失败: {e}")
            return {}
    
    def add_documents_to_advanced_retriever(self, documents: List[Document]):
        """向高级检索器添加文档"""
        if self.advanced_retriever:
            try:
                self.advanced_retriever.add_documents_to_parent_retriever(documents)
                logger.info(f"成功向高级检索器添加 {len(documents)} 个文档")
            except Exception as e:
                logger.error(f"向高级检索器添加文档失败: {e}")
    
    def get_advanced_retriever_stats(self) -> Dict[str, Any]:
        """获取高级检索器统计信息"""
        if self.advanced_retriever:
            return self.advanced_retriever.get_retriever_stats()
        return {"advanced_retriever_enabled": False}
    
    def _is_stakeholder_question(self, question: str) -> bool:
        """判断是否为相关方问题"""
        stakeholder_keywords = [
            "相关方", "利益相关方", "stakeholder", "相关者",
            "利益相关者", "关联方", "相关人员", "相关组织"
        ]
        return any(keyword in question.lower() for keyword in stakeholder_keywords)
    
    def _handle_kimi_file_question(self, db: Session, question: str, 
                                  kimi_files: List[str], user_id: Optional[int], 
                                  session_id: Optional[str], start_time: datetime) -> AnswerResponse:
        """处理Kimi模型的文件问答"""
        try:
            logger.info(f"📁 开始处理Kimi文件问答，文件数量: {len(kimi_files)}")
            
            # 获取文件内容
            from app.services.kimi_file_service import KimiFileService
            kimi_file_service = KimiFileService()
            
            file_contents = []
            for file_id in kimi_files:
                try:
                    content = kimi_file_service.get_file_content(file_id)
                    if content:
                        file_contents.append(content)
                        logger.info(f"📄 成功获取文件内容: {file_id}")
                except Exception as e:
                    logger.warning(f"⚠️ 获取文件内容失败: {file_id}, 错误: {str(e)}")
            
            if not file_contents:
                logger.error("❌ 未能获取任何文件内容")
                return self.create_error_response("无法获取文件内容，请重新上传文件")
            
            # 构建包含文件内容的提示
            file_context = "\n\n".join([f"文件内容 {i+1}:\n{content}" for i, content in enumerate(file_contents)])
            enhanced_question = f"基于以下文件内容回答问题：\n\n{file_context}\n\n问题：{question}"
            
            logger.info(f"📝 构建增强问题，总长度: {len(enhanced_question)} 字符")
            
            # 使用LLM服务直接回答
            if self.llm_service:
                try:
                    answer = self.llm_service.generate_response(enhanced_question)
                    logger.info(f"✅ Kimi模型回答生成成功，长度: {len(answer)} 字符")
                    
                    # 计算处理时间
                    processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                    
                    # 创建源文档（基于文件）
                    source_documents = []
                    for i, file_id in enumerate(kimi_files):
                        source_documents.append(SourceDocument(
                            content=file_contents[i][:200] + "..." if len(file_contents[i]) > 200 else file_contents[i],
                            metadata={
                                "source": f"kimi_file_{file_id}",
                                "file_id": file_id,
                                "type": "kimi_file"
                            },
                            score=1.0
                        ))
                    
                    # 记录问答日志
                    qa_log = self.log_qa_interaction(
                        db, question, answer, source_documents, 
                        user_id, session_id, processing_time
                    )
                    
                    return self.create_answer_response(
                        question=question,
                        answer=answer,
                        source_documents=source_documents,
                        processing_time=processing_time,
                        session_id=session_id,
                        metadata={"model": self.current_model, "kimi_files": kimi_files},
                        qa_log_id=qa_log.id if qa_log else None
                    )
                    
                except Exception as e:
                    logger.error(f"❌ Kimi模型回答生成失败: {str(e)}")
                    return self.create_error_response(f"生成回答时出错: {str(e)}")
            else:
                logger.error("❌ LLM服务未初始化")
                return self.create_error_response("语言模型服务不可用")
                
        except Exception as e:
            logger.error(f"❌ 处理Kimi文件问答时出错: {str(e)}")
            return self.create_error_response(f"处理文件问答时出错: {str(e)}")
    
    def _handle_stakeholder_question(self, db: Session, question: str, 
                                   user_id: Optional[int], session_id: Optional[str], 
                                   start_time: datetime) -> AnswerResponse:
        """处理相关方问题"""
        try:
            logger.info("👥 开始处理相关方问题")
            
            # 使用关键词搜索相关方信息
            stakeholder_keywords = ["相关方", "利益相关方", "顾客", "供方", "员工", "股东"]
            logger.info(f"🔑 搜索关键词: {stakeholder_keywords}")
            
            all_docs = []
            for i, keyword in enumerate(stakeholder_keywords):
                logger.info(f"🔍 步骤{i+1}: 搜索关键词 '{keyword}'")
                # 使用向量搜索
                retriever = self.vector_service.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 10}
                )
                docs = retriever.get_relevant_documents(keyword)
                all_docs.extend(docs)
                logger.info(f"📊 关键词 '{keyword}' 搜索结果: {len(docs)} 个文档")
            
            logger.info(f"📋 合并前文档总数: {len(all_docs)}")
            
            # 去重
            logger.info("🔄 开始文档去重处理")
            unique_docs = []
            seen_content = set()
            for doc in all_docs:
                if doc.page_content not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(doc.page_content)
            
            logger.info(f"📋 去重后文档数量: {len(unique_docs)}")
            
            if not unique_docs:
                logger.warning("❌ 未找到任何相关方文档")
                return self.create_error_response("未找到相关方信息")
            
            # 构建上下文
            logger.info("🔄 构建上下文")
            selected_docs = unique_docs[:15]
            context = "\n\n".join([doc.page_content for doc in selected_docs])
            context_length = len(context)
            logger.info(f"📝 上下文构建完成: 使用 {len(selected_docs)} 个文档, 总长度 {context_length} 字符")
            
            # 使用LLM生成答案
            logger.info("🤖 开始使用LLM生成答案")
            prompt = f"""
基于以下文档内容，详细回答关于相关方的问题。请列出所有相关方类型并说明其特点。

文档内容：
{context}

问题：{question}

请提供详细、准确的答案：
"""
            
            logger.info(f"📝 提示词长度: {len(prompt)} 字符")
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"📝 LLM答案生成完成: {len(answer)} 字符")
            
            # 转换源文档
            logger.info("🔄 转换源文档格式")
            source_documents = self.convert_source_documents(unique_docs[:10])
            logger.info(f"📄 成功转换 {len(source_documents)} 个源文档")
            
            # 计算处理时间
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(f"⏱️ 相关方问题处理时间: {processing_time:.2f}秒")
            
            # 缓存答案
            logger.info("💾 缓存相关方答案")
            self.cache_answer(question, "相关方", answer, source_documents)
            
            # 记录问答日志
            logger.info("📝 记录相关方问答日志")
            qa_log = self.log_qa_interaction(
                db=db,
                question=question,
                answer=answer,
                source_documents=source_documents,
                user_id=user_id,
                session_id=session_id,
                processing_time=processing_time
            )
            
            logger.info("✅ 相关方问题处理完成")
            return self.create_answer_response(
                question=question,
                answer=answer,
                source_documents=source_documents,
                processing_time=processing_time,
                session_id=session_id,
                metadata={
                    "retrieval_method": "stakeholder_special",
                    "category": "相关方"
                },
                qa_log_id=qa_log.id if qa_log else None
            )
            
        except Exception as e:
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.error(f"❌ 处理相关方问题失败: {e} (耗时: {processing_time:.2f}秒)")
            return self.create_error_response(f"处理相关方问题失败: {str(e)}")

# QA服务工具函数
def extract_keywords_from_question(question: str) -> List[str]:
    """从问题中提取关键词"""
    import re
    from collections import Counter
    
    # 简单的关键词提取
    words = re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', question)
    
    # 过滤停用词
    stop_words = {'怎么', '如何', '什么', '哪个', '为什么', '是否', '可以', '需要'}
    keywords = [word for word in words if word not in stop_words and len(word) > 1]
    
    # 返回最常见的关键词
    word_counts = Counter(keywords)
    return [word for word, count in word_counts.most_common(5)]

def format_answer_with_steps(answer: str) -> str:
    """格式化包含步骤的答案"""
    try:
        # 检测步骤模式
        step_patterns = [
            r'(\d+[.、])',  # 数字步骤
            r'([一二三四五六七八九十][.、])',  # 中文数字步骤
            r'(第[一二三四五六七八九十]+步)',  # 第X步
        ]
        
        formatted_answer = answer
        
        for pattern in step_patterns:
            if re.search(pattern, answer):
                # 在步骤前添加换行
                formatted_answer = re.sub(
                    pattern, 
                    r'\n\1', 
                    formatted_answer
                )
                break
        
        return formatted_answer.strip()
        
    except Exception as e:
        logger.error(f"格式化答案失败: {e}")
        return answer
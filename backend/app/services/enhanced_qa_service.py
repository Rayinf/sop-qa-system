from typing import List, Dict, Any, Optional, Tuple
import logging
import time
from dataclasses import dataclass

from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever

from app.core.config import settings
from app.services.llm_service import LLMService
from app.services.hybrid_retrieval_service import HybridRetrievalService
from app.services.reranking_service import RerankingService
from app.services.query_expansion_service import QueryExpansionService
from app.services.rag_fusion_service import RAGFusionService
from app.services.answer_quality_service import AnswerQualityService, QualityLevel
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)

@dataclass
class EnhancedQAResult:
    """增强QA结果"""
    answer: str
    source_documents: List[Document]
    confidence_score: float
    quality_score: Optional[Any] = None
    retrieval_method: str = "standard"
    processing_time: float = 0.0
    metadata: Dict[str, Any] = None

class EnhancedQAService:
    """
    增强的问答服务 - 集成所有优化技术
    """
    
    def __init__(self):
        # 初始化各个服务
        self.llm_service = None
        self.vector_service = None
        self.hybrid_retrieval_service = None
        self.reranking_service = None
        self.query_expansion_service = None
        self.rag_fusion_service = None
        self.answer_quality_service = None
        
        self._initialize_services()
        
        # 配置参数
        self.config = {
            'use_hybrid_retrieval': True,
            'use_reranking': True,
            'use_query_expansion': True,
            'use_rag_fusion': True,
            'use_quality_evaluation': True,
            'min_quality_level': QualityLevel.FAIR,
            'retrieval_k': settings.retrieval_k,
            'max_retries': 2
        }
        
        logger.info("增强QA服务初始化完成")
    
    def _initialize_services(self):
        """初始化各个服务"""
        try:
            # LLM服务
            self.llm_service = LLMService()
            logger.info("LLM服务初始化成功")
        except Exception as e:
            logger.error(f"LLM服务初始化失败: {e}")
        
        try:
            # 向量服务
            self.vector_service = VectorService.get_instance()
            logger.info("向量服务初始化成功")
        except Exception as e:
            logger.error(f"向量服务初始化失败: {e}")
        
        try:
            # 混合检索服务
            if self.vector_service:
                self.hybrid_retrieval_service = HybridRetrievalService(self.vector_service)
                logger.info("混合检索服务初始化成功")
        except Exception as e:
            logger.error(f"混合检索服务初始化失败: {e}")
        
        try:
            # 重排序服务
            self.reranking_service = RerankingService()
            logger.info("重排序服务初始化成功")
        except Exception as e:
            logger.error(f"重排序服务初始化失败: {e}")
        
        try:
            # 查询扩展服务
            self.query_expansion_service = QueryExpansionService()
            logger.info("查询扩展服务初始化成功")
        except Exception as e:
            logger.error(f"查询扩展服务初始化失败: {e}")
        
        try:
            # RAG Fusion服务
            self.rag_fusion_service = RAGFusionService()
            logger.info("RAG Fusion服务初始化成功")
        except Exception as e:
            logger.error(f"RAG Fusion服务初始化失败: {e}")
        
        try:
            # 答案质量评估服务
            self.answer_quality_service = AnswerQualityService()
            logger.info("答案质量评估服务初始化成功")
        except Exception as e:
            logger.error(f"答案质量评估服务初始化失败: {e}")
    
    def enhanced_qa(self, 
                   query: str, 
                   category: Optional[str] = None,
                   retrieval_method: str = 'auto',
                   **kwargs) -> EnhancedQAResult:
        """增强的问答处理"""
        start_time = time.time()
        
        try:
            logger.info(f"开始增强QA处理: {query[:50]}... (类别: {category}, 方法: {retrieval_method})")
            
            # 1. 选择最佳检索策略
            retrieval_strategy = self._select_retrieval_strategy(query, category, retrieval_method)
            
            # 2. 执行检索
            documents = self._execute_retrieval(query, category, retrieval_strategy)
            
            # 3. 生成答案
            answer = self._generate_answer(query, documents)
            
            # 4. 评估答案质量
            quality_score = None
            if self.config['use_quality_evaluation'] and self.answer_quality_service:
                quality_score = self.answer_quality_service.evaluate_answer_quality(
                    query, answer, documents
                )
                
                # 如果质量不达标，尝试重新生成
                if (quality_score.level.value in ['poor', 'fair'] and 
                    quality_score.overall_score < self._quality_level_to_score(self.config['min_quality_level'])):
                    logger.info(f"答案质量不达标 ({quality_score.level.value})，尝试重新生成")
                    answer = self._regenerate_answer(query, documents, quality_score)
                    
                    # 重新评估
                    quality_score = self.answer_quality_service.evaluate_answer_quality(
                        query, answer, documents
                    )
            
            # 5. 计算置信度
            confidence_score = self._calculate_confidence(query, answer, documents, quality_score)
            
            # 6. 构建结果
            processing_time = time.time() - start_time
            
            result = EnhancedQAResult(
                answer=answer,
                source_documents=documents,
                confidence_score=confidence_score,
                quality_score=quality_score,
                retrieval_method=retrieval_strategy,
                processing_time=processing_time,
                metadata={
                    'query_length': len(query),
                    'answer_length': len(answer),
                    'source_docs_count': len(documents),
                    'category': category,
                    'config': self.config.copy()
                }
            )
            
            logger.info(f"增强QA处理完成: {processing_time:.2f}s, 置信度: {confidence_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"增强QA处理失败: {e}")
            # 返回降级结果
            return EnhancedQAResult(
                answer=f"抱歉，处理您的问题时出现错误: {str(e)}",
                source_documents=[],
                confidence_score=0.0,
                retrieval_method="error",
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _select_retrieval_strategy(self, query: str, category: Optional[str], method: str) -> str:
        """选择最佳检索策略"""
        try:
            if method != 'auto':
                return method
            
            # 基于查询特征自动选择策略
            query_length = len(query)
            
            # 短查询使用RAG Fusion
            if query_length < 20 and self.rag_fusion_service:
                return 'rag_fusion'
            
            # 中等长度查询使用混合检索
            elif query_length < 100 and self.hybrid_retrieval_service:
                return 'hybrid'
            
            # 长查询使用标准检索
            else:
                return 'standard'
                
        except Exception as e:
            logger.error(f"选择检索策略失败: {e}")
            return 'standard'
    
    def _execute_retrieval(self, query: str, category: Optional[str], strategy: str) -> List[Document]:
        """执行检索"""
        try:
            documents = []
            
            if strategy == 'rag_fusion' and self.rag_fusion_service:
                # RAG Fusion检索
                retriever = self._get_base_retriever(category)
                if retriever:
                    documents = self.rag_fusion_service.rag_fusion_search(
                        query, retriever, 
                        num_queries=4,
                        fusion_method='rrf',
                        k=self.config['retrieval_k']
                    )
                    logger.info(f"RAG Fusion检索完成: {len(documents)} 个文档")
            
            elif strategy == 'hybrid' and self.hybrid_retrieval_service:
                # 混合检索
                hybrid_retriever = self.hybrid_retrieval_service.get_hybrid_retriever(
                    category=category,
                    use_reranking=self.config['use_reranking'],
                    k=self.config['retrieval_k']
                )
                documents = hybrid_retriever.get_relevant_documents(query)
                logger.info(f"混合检索完成: {len(documents)} 个文档")
            
            else:
                # 标准检索
                retriever = self._get_base_retriever(category)
                if retriever:
                    documents = retriever.get_relevant_documents(query)
                    
                    # 应用重排序
                    if self.config['use_reranking'] and self.reranking_service:
                        documents = self.reranking_service.rerank_documents(
                            documents, query, reranker_type='default'
                        )
                    
                    logger.info(f"标准检索完成: {len(documents)} 个文档")
            
            return documents[:self.config['retrieval_k']]
            
        except Exception as e:
            logger.error(f"执行检索失败: {e}")
            return []
    
    def _get_base_retriever(self, category: Optional[str]) -> Optional[BaseRetriever]:
        """获取基础检索器"""
        try:
            if self.vector_service:
                if category:
                    return self.vector_service.get_category_retriever(category)
                else:
                    return self.vector_service.get_retriever()
            return None
        except Exception as e:
            logger.error(f"获取基础检索器失败: {e}")
            return None
    
    def _generate_answer(self, query: str, documents: List[Document]) -> str:
        """生成答案"""
        try:
            if not self.llm_service:
                return "抱歉，LLM服务不可用。"
            
            if not documents:
                return "抱歉，没有找到相关的文档信息。"
            
            # 构建上下文
            context = self._build_context(documents)
            
            # 生成答案
            answer = self.llm_service.generate_answer(query, context)
            
            return answer if answer else "抱歉，无法生成答案。"
            
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return f"抱歉，生成答案时出现错误: {str(e)}"
    
    def _regenerate_answer(self, query: str, documents: List[Document], quality_score) -> str:
        """重新生成答案"""
        try:
            logger.info("尝试重新生成更高质量的答案")
            
            # 使用更详细的提示词
            context = self._build_enhanced_context(documents)
            
            # 添加质量要求到提示词
            enhanced_query = f"""
请基于以下文档内容，详细回答用户的问题。要求：
1. 答案要完整、准确、相关
2. 使用清晰的结构和逻辑
3. 如果信息不足，请明确说明
4. 避免重复和冗余

用户问题：{query}

请提供高质量的答案：
"""
            
            answer = self.llm_service.generate_answer(enhanced_query, context)
            
            return answer if answer else "抱歉，无法生成满意的答案。"
            
        except Exception as e:
            logger.error(f"重新生成答案失败: {e}")
            return "抱歉，重新生成答案时出现错误。"
    
    def _build_context(self, documents: List[Document]) -> str:
        """构建上下文"""
        try:
            context_parts = []
            
            for i, doc in enumerate(documents[:5]):  # 限制文档数量
                title = doc.metadata.get('title', f'文档{i+1}')
                content = doc.page_content[:500]  # 限制长度
                
                context_parts.append(f"【{title}】\n{content}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"构建上下文失败: {e}")
            return ""
    
    def _build_enhanced_context(self, documents: List[Document]) -> str:
        """构建增强上下文"""
        try:
            context_parts = []
            
            for i, doc in enumerate(documents[:3]):  # 使用更少但更相关的文档
                title = doc.metadata.get('title', f'文档{i+1}')
                content = doc.page_content[:800]  # 更多内容
                source = doc.metadata.get('source', '未知来源')
                
                context_parts.append(f"【{title}】(来源: {source})\n{content}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"构建增强上下文失败: {e}")
            return ""
    
    def _calculate_confidence(self, 
                            query: str, 
                            answer: str, 
                            documents: List[Document],
                            quality_score) -> float:
        """计算置信度"""
        try:
            confidence = 0.5  # 基础置信度
            
            # 基于文档数量
            if documents:
                doc_confidence = min(0.3, len(documents) * 0.05)
                confidence += doc_confidence
            
            # 基于答案长度
            if 50 <= len(answer) <= 500:
                confidence += 0.1
            
            # 基于质量评分
            if quality_score:
                quality_confidence = quality_score.overall_score * 0.3
                confidence += quality_confidence
            
            # 基于文档相关性
            if documents and self.answer_quality_service:
                try:
                    relevance = self.answer_quality_service._calculate_source_relevance(query, documents)
                    confidence += relevance * 0.2
                except:
                    pass
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"计算置信度失败: {e}")
            return 0.5
    
    def _quality_level_to_score(self, level: QualityLevel) -> float:
        """质量等级转换为分数"""
        mapping = {
            QualityLevel.EXCELLENT: 0.8,
            QualityLevel.GOOD: 0.6,
            QualityLevel.FAIR: 0.4,
            QualityLevel.POOR: 0.0
        }
        return mapping.get(level, 0.4)
    
    def batch_qa(self, 
                queries: List[str], 
                categories: Optional[List[str]] = None) -> List[EnhancedQAResult]:
        """批量问答处理"""
        try:
            logger.info(f"开始批量QA处理: {len(queries)} 个问题")
            
            results = []
            categories = categories or [None] * len(queries)
            
            for i, query in enumerate(queries):
                category = categories[i] if i < len(categories) else None
                result = self.enhanced_qa(query, category)
                results.append(result)
            
            logger.info(f"批量QA处理完成: {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"批量QA处理失败: {e}")
            return []
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        try:
            self.config.update(new_config)
            logger.info(f"配置已更新: {new_config}")
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            'llm_service': self.llm_service is not None,
            'vector_service': self.vector_service is not None,
            'hybrid_retrieval_service': self.hybrid_retrieval_service is not None,
            'reranking_service': self.reranking_service is not None,
            'query_expansion_service': self.query_expansion_service is not None,
            'rag_fusion_service': self.rag_fusion_service is not None,
            'answer_quality_service': self.answer_quality_service is not None,
            'config': self.config.copy()
        }
    
    def get_performance_stats(self, results: List[EnhancedQAResult]) -> Dict[str, Any]:
        """获取性能统计"""
        try:
            if not results:
                return {'total_queries': 0}
            
            processing_times = [r.processing_time for r in results]
            confidence_scores = [r.confidence_score for r in results]
            
            stats = {
                'total_queries': len(results),
                'avg_processing_time': sum(processing_times) / len(processing_times),
                'max_processing_time': max(processing_times),
                'min_processing_time': min(processing_times),
                'avg_confidence': sum(confidence_scores) / len(confidence_scores),
                'high_confidence_ratio': len([c for c in confidence_scores if c >= 0.7]) / len(confidence_scores),
                'retrieval_methods': {}
            }
            
            # 统计检索方法使用情况
            for result in results:
                method = result.retrieval_method
                stats['retrieval_methods'][method] = stats['retrieval_methods'].get(method, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"获取性能统计失败: {e}")
            return {'error': str(e)}
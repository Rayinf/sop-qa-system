from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
from collections import defaultdict, Counter
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

from app.core.config import settings
from app.services.llm_service import LLMService
from app.services.query_expansion_service import QueryExpansionService

logger = logging.getLogger(__name__)

class RAGFusionService:
    """
    RAG Fusion服务 - 实现多查询生成和结果融合
    """
    
    def __init__(self):
        self.llm_service = None
        self.query_expansion_service = None
        
        # 初始化服务
        try:
            self.llm_service = LLMService()
            logger.info("RAG Fusion的LLM服务初始化成功")
        except Exception as e:
            logger.warning(f"LLM服务初始化失败: {e}")
        
        try:
            self.query_expansion_service = QueryExpansionService()
            logger.info("RAG Fusion的查询扩展服务初始化成功")
        except Exception as e:
            logger.warning(f"查询扩展服务初始化失败: {e}")
        
        logger.info("RAG Fusion服务初始化完成")
    
    def generate_multiple_queries(self, 
                                 original_query: str, 
                                 num_queries: int = 4,
                                 method: str = 'comprehensive') -> List[str]:
        """生成多个查询变体"""
        try:
            logger.info(f"生成多个查询变体: {original_query[:50]}... (数量: {num_queries})")
            
            queries = [original_query]  # 包含原始查询
            
            if method == 'llm' and self.llm_service:
                # 使用LLM生成查询变体
                llm_queries = self._generate_queries_with_llm(original_query, num_queries - 1)
                queries.extend(llm_queries)
            
            elif method == 'expansion' and self.query_expansion_service:
                # 使用查询扩展服务
                expanded_queries = self.query_expansion_service.expand_query(
                    original_query, 
                    expansion_type='comprehensive',
                    max_expansions=num_queries - 1
                )
                queries.extend(expanded_queries[1:])  # 排除原始查询
            
            elif method == 'comprehensive':
                # 综合方法：结合LLM和扩展
                if self.llm_service:
                    llm_queries = self._generate_queries_with_llm(original_query, 2)
                    queries.extend(llm_queries)
                
                if self.query_expansion_service:
                    expanded_queries = self.query_expansion_service.expand_query(
                        original_query,
                        expansion_type='synonym',
                        max_expansions=2
                    )
                    queries.extend(expanded_queries[1:])  # 排除原始查询
            
            # 去重并限制数量
            unique_queries = list(dict.fromkeys(queries))[:num_queries]
            
            logger.info(f"生成查询变体完成: {len(unique_queries)} 个")
            for i, query in enumerate(unique_queries):
                logger.debug(f"查询 {i+1}: {query}")
            
            return unique_queries
            
        except Exception as e:
            logger.error(f"生成多个查询失败: {e}")
            return [original_query]
    
    def _generate_queries_with_llm(self, original_query: str, num_queries: int) -> List[str]:
        """使用LLM生成查询变体"""
        try:
            prompt_template = PromptTemplate(
                input_variables=["query", "num_queries"],
                template="""
你是一个专业的查询重写专家。请为以下查询生成 {num_queries} 个不同的查询变体，这些变体应该能够从不同角度检索到相关信息。

要求：
1. 保持查询的核心意图和语义
2. 使用不同的表达方式、同义词或相关术语
3. 可以从不同角度或层面来表达同一个问题
4. 每个变体应该独立且完整
5. 每行一个查询，不要编号
6. 不要包含原始查询

原始查询：{query}

请生成 {num_queries} 个查询变体：
"""
            )
            
            prompt = prompt_template.format(query=original_query, num_queries=num_queries)
            response = self.llm_service.generate_response(prompt)
            
            if response and response.strip():
                queries = [
                    line.strip() 
                    for line in response.strip().split('\n') 
                    if line.strip() and line.strip() != original_query
                ]
                
                logger.debug(f"LLM生成 {len(queries)} 个查询变体")
                return queries[:num_queries]
            
            return []
            
        except Exception as e:
            logger.error(f"LLM生成查询变体失败: {e}")
            return []
    
    def fusion_retrieve(self, 
                       queries: List[str], 
                       retriever: BaseRetriever,
                       fusion_method: str = 'rrf',
                       k: int = 20) -> List[Document]:
        """融合多个查询的检索结果"""
        try:
            logger.info(f"开始融合检索: {len(queries)} 个查询")
            
            # 并行检索所有查询
            all_results = self._parallel_retrieve(queries, retriever, k)
            
            # 融合结果
            if fusion_method == 'rrf':
                fused_docs = self._reciprocal_rank_fusion(all_results)
            elif fusion_method == 'weighted':
                fused_docs = self._weighted_fusion(all_results, queries)
            elif fusion_method == 'vote':
                fused_docs = self._vote_fusion(all_results)
            else:
                # 默认使用RRF
                fused_docs = self._reciprocal_rank_fusion(all_results)
            
            logger.info(f"融合检索完成: 返回 {len(fused_docs)} 个文档")
            return fused_docs[:k]
            
        except Exception as e:
            logger.error(f"融合检索失败: {e}")
            return []
    
    def _parallel_retrieve(self, 
                          queries: List[str], 
                          retriever: BaseRetriever,
                          k: int) -> List[List[Document]]:
        """并行检索多个查询"""
        try:
            all_results = []
            
            # 使用线程池并行检索
            with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as executor:
                # 提交所有检索任务
                future_to_query = {
                    executor.submit(self._safe_retrieve, retriever, query, k): query 
                    for query in queries
                }
                
                # 收集结果
                for future in as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        docs = future.result(timeout=30)  # 30秒超时
                        all_results.append(docs)
                        logger.debug(f"查询 '{query[:30]}...' 检索到 {len(docs)} 个文档")
                    except Exception as e:
                        logger.warning(f"查询 '{query[:30]}...' 检索失败: {e}")
                        all_results.append([])  # 添加空结果
            
            return all_results
            
        except Exception as e:
            logger.error(f"并行检索失败: {e}")
            return []
    
    def _safe_retrieve(self, retriever: BaseRetriever, query: str, k: int) -> List[Document]:
        """安全的检索方法"""
        try:
            docs = retriever.get_relevant_documents(query)
            return docs[:k]
        except Exception as e:
            logger.warning(f"检索查询 '{query[:30]}...' 失败: {e}")
            return []
    
    def _reciprocal_rank_fusion(self, all_results: List[List[Document]], k: int = 60) -> List[Document]:
        """倒数排名融合 (Reciprocal Rank Fusion)"""
        try:
            doc_scores = defaultdict(float)
            doc_objects = {}
            
            # 为每个文档计算RRF分数
            for results in all_results:
                for rank, doc in enumerate(results):
                    # 使用文档内容作为唯一标识
                    doc_id = self._get_doc_id(doc)
                    
                    # RRF公式: 1 / (k + rank)
                    rrf_score = 1.0 / (k + rank + 1)
                    doc_scores[doc_id] += rrf_score
                    
                    # 保存文档对象
                    if doc_id not in doc_objects:
                        doc_objects[doc_id] = doc
            
            # 按分数排序
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 构建结果列表
            fused_docs = []
            for doc_id, score in sorted_docs:
                doc = doc_objects[doc_id]
                # 添加融合分数到元数据
                doc.metadata['fusion_score'] = score
                doc.metadata['fusion_method'] = 'rrf'
                fused_docs.append(doc)
            
            logger.debug(f"RRF融合: {len(fused_docs)} 个唯一文档")
            return fused_docs
            
        except Exception as e:
            logger.error(f"RRF融合失败: {e}")
            return []
    
    def _weighted_fusion(self, all_results: List[List[Document]], queries: List[str]) -> List[Document]:
        """加权融合"""
        try:
            doc_scores = defaultdict(float)
            doc_objects = {}
            
            # 为不同查询分配权重（原始查询权重更高）
            weights = [1.0] + [0.8] * (len(queries) - 1)
            
            for i, results in enumerate(all_results):
                weight = weights[i] if i < len(weights) else 0.5
                
                for rank, doc in enumerate(results):
                    doc_id = self._get_doc_id(doc)
                    
                    # 加权分数：权重 * (1 / (rank + 1))
                    weighted_score = weight * (1.0 / (rank + 1))
                    doc_scores[doc_id] += weighted_score
                    
                    if doc_id not in doc_objects:
                        doc_objects[doc_id] = doc
            
            # 按分数排序
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 构建结果列表
            fused_docs = []
            for doc_id, score in sorted_docs:
                doc = doc_objects[doc_id]
                doc.metadata['fusion_score'] = score
                doc.metadata['fusion_method'] = 'weighted'
                fused_docs.append(doc)
            
            logger.debug(f"加权融合: {len(fused_docs)} 个唯一文档")
            return fused_docs
            
        except Exception as e:
            logger.error(f"加权融合失败: {e}")
            return []
    
    def _vote_fusion(self, all_results: List[List[Document]]) -> List[Document]:
        """投票融合"""
        try:
            doc_votes = defaultdict(int)
            doc_objects = {}
            doc_positions = defaultdict(list)
            
            # 统计每个文档的出现次数和位置
            for results in all_results:
                for rank, doc in enumerate(results):
                    doc_id = self._get_doc_id(doc)
                    doc_votes[doc_id] += 1
                    doc_positions[doc_id].append(rank)
                    
                    if doc_id not in doc_objects:
                        doc_objects[doc_id] = doc
            
            # 计算综合分数：投票数 + 平均排名的倒数
            doc_scores = {}
            for doc_id, votes in doc_votes.items():
                avg_rank = np.mean(doc_positions[doc_id])
                # 综合分数：投票权重 + 排名权重
                score = votes * 2.0 + (1.0 / (avg_rank + 1))
                doc_scores[doc_id] = score
            
            # 按分数排序
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 构建结果列表
            fused_docs = []
            for doc_id, score in sorted_docs:
                doc = doc_objects[doc_id]
                doc.metadata['fusion_score'] = score
                doc.metadata['fusion_method'] = 'vote'
                doc.metadata['vote_count'] = doc_votes[doc_id]
                fused_docs.append(doc)
            
            logger.debug(f"投票融合: {len(fused_docs)} 个唯一文档")
            return fused_docs
            
        except Exception as e:
            logger.error(f"投票融合失败: {e}")
            return []
    
    def _get_doc_id(self, doc: Document) -> str:
        """获取文档唯一标识"""
        try:
            # 优先使用文档ID
            if 'document_id' in doc.metadata:
                return f"doc_{doc.metadata['document_id']}"
            
            # 使用chunk_id
            if 'chunk_id' in doc.metadata:
                return f"chunk_{doc.metadata['chunk_id']}"
            
            # 使用内容哈希
            import hashlib
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:16]
            return f"hash_{content_hash}"
            
        except Exception as e:
            logger.warning(f"获取文档ID失败: {e}")
            return f"unknown_{id(doc)}"
    
    def rag_fusion_search(self, 
                         query: str, 
                         retriever: BaseRetriever,
                         num_queries: int = 4,
                         fusion_method: str = 'rrf',
                         k: int = 20) -> List[Document]:
        """完整的RAG Fusion搜索流程"""
        try:
            logger.info(f"开始RAG Fusion搜索: {query[:50]}...")
            
            # 1. 生成多个查询
            queries = self.generate_multiple_queries(
                query, 
                num_queries=num_queries,
                method='comprehensive'
            )
            
            # 2. 融合检索
            fused_docs = self.fusion_retrieve(
                queries, 
                retriever,
                fusion_method=fusion_method,
                k=k
            )
            
            # 3. 添加RAG Fusion元数据
            for i, doc in enumerate(fused_docs):
                doc.metadata['rag_fusion_rank'] = i + 1
                doc.metadata['source_queries'] = len(queries)
            
            logger.info(f"RAG Fusion搜索完成: 返回 {len(fused_docs)} 个文档")
            return fused_docs
            
        except Exception as e:
            logger.error(f"RAG Fusion搜索失败: {e}")
            # 降级到普通检索
            try:
                return retriever.get_relevant_documents(query)[:k]
            except:
                return []
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        """获取融合统计信息"""
        return {
            'llm_available': self.llm_service is not None,
            'query_expansion_available': self.query_expansion_service is not None,
            'supported_fusion_methods': ['rrf', 'weighted', 'vote'],
            'supported_generation_methods': ['llm', 'expansion', 'comprehensive']
        }
    
    def analyze_fusion_results(self, docs: List[Document]) -> Dict[str, Any]:
        """分析融合结果"""
        try:
            if not docs:
                return {'total_docs': 0}
            
            # 统计融合信息
            fusion_scores = [doc.metadata.get('fusion_score', 0) for doc in docs]
            vote_counts = [doc.metadata.get('vote_count', 0) for doc in docs if 'vote_count' in doc.metadata]
            
            analysis = {
                'total_docs': len(docs),
                'avg_fusion_score': np.mean(fusion_scores) if fusion_scores else 0,
                'max_fusion_score': max(fusion_scores) if fusion_scores else 0,
                'min_fusion_score': min(fusion_scores) if fusion_scores else 0,
                'fusion_method': docs[0].metadata.get('fusion_method', 'unknown'),
                'source_queries': docs[0].metadata.get('source_queries', 0)
            }
            
            if vote_counts:
                analysis['avg_vote_count'] = np.mean(vote_counts)
                analysis['max_vote_count'] = max(vote_counts)
            
            return analysis
            
        except Exception as e:
            logger.error(f"分析融合结果失败: {e}")
            return {'error': str(e)}
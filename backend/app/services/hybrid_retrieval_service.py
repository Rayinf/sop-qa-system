from typing import List, Optional, Dict, Any, Tuple
import logging
import numpy as np
from collections import defaultdict

from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
# from langchain.retrievers.document_compressors import CrossEncoderReranker
# from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from rank_bm25 import BM25Okapi
import jieba
import re

from app.core.config import settings
from app.services.vector_service import VectorService
from app.services.reranking_service import RerankingService

logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    """
    混合检索器：结合密集向量检索和稀疏BM25检索
    """
    vector_store: FAISS
    documents: List[Document]
    category: Optional[str] = None
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    k: int = 10
    use_reranking: bool = True
    reranking_service: Optional[object] = None
    bm25: Optional[object] = None
    doc_mapping: Dict[int, Document] = {}
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, 
                 vector_store: FAISS,
                 documents: List[Document],
                 category: Optional[str] = None,
                 dense_weight: float = 0.7,
                 sparse_weight: float = 0.3,
                 k: int = 10,
                 use_reranking: bool = True):
        
        # 初始化重排序服务
        reranking_service = None
        if use_reranking:
            try:
                reranking_service = RerankingService()
                logger.info("重排序服务初始化成功")
            except Exception as e:
                logger.warning(f"重排序服务初始化失败: {e}，将不使用重排序")
                use_reranking = False
        
        # 初始化 BM25 和文档映射
        bm25 = None
        doc_mapping = {}
        
        # 初始化父类，传递所有 Pydantic 字段
        super().__init__(
            vector_store=vector_store,
            documents=documents,
            category=category,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            k=k,
            use_reranking=use_reranking,
            reranking_service=reranking_service,
            bm25=bm25,
            doc_mapping=doc_mapping
        )
        
        # 初始化BM25检索器
        self._initialize_bm25()
        
        logger.info(f"初始化混合检索器: category={category}, dense_weight={dense_weight}, sparse_weight={sparse_weight}, reranking={use_reranking}")
    
    def _initialize_bm25(self):
        """初始化BM25检索器"""
        try:
            # 准备文档文本用于BM25
            corpus = []
            self.doc_mapping = {}  # 映射BM25索引到原始文档
            
            for i, doc in enumerate(self.documents):
                # 过滤类别（如果指定）
                if self.category:
                    doc_category = doc.metadata.get('category', '')
                    if doc_category != self.category and not self._is_category_match(doc, self.category):
                        continue
                
                # 分词处理
                text = doc.page_content
                title = doc.metadata.get('title', '')
                combined_text = f"{title} {text}"
                
                # 中文分词
                tokens = self._tokenize_chinese(combined_text)
                corpus.append(tokens)
                self.doc_mapping[len(corpus) - 1] = doc
            
            if corpus:
                self.bm25 = BM25Okapi(corpus)
                logger.info(f"BM25检索器初始化成功，文档数量: {len(corpus)}")
            else:
                self.bm25 = None
                logger.warning("BM25检索器初始化失败：没有可用文档")
                
        except Exception as e:
            logger.error(f"BM25检索器初始化失败: {e}")
            self.bm25 = None
    
    def _tokenize_chinese(self, text: str) -> List[str]:
        """中文分词"""
        # 清理文本
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 使用jieba分词
        tokens = list(jieba.cut(text))
        
        # 过滤停用词和短词
        stop_words = {'的', '是', '在', '了', '和', '有', '为', '与', '等', '及', '或', '但', '而', '也', '都', '要', '可', '能', '会', '将', '应', '应该', '需要', '进行', '实施', '执行', '完成'}
        tokens = [token.strip() for token in tokens if len(token.strip()) > 1 and token.strip() not in stop_words]
        
        return tokens
    
    def _is_category_match(self, doc: Document, target_category: str) -> bool:
        """检查文档是否匹配目标类别"""
        title = doc.metadata.get('title', '').lower()
        content = doc.page_content.lower()
        
        category_keywords = {
            'development': ['开发', '程序', '代码', '技术', '软件', '系统'],
            'manual': ['手册', '规范', '标准', '质量', '管理'],
            'procedure': ['流程', '程序', '操作', '作业', '指导'],
            'policy': ['政策', '规章', '规定', '条例'],
            'guideline': ['指南', '指导', '建议', '推荐']
        }
        
        keywords = category_keywords.get(target_category, [])
        return any(keyword in title or keyword in content for keyword in keywords)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """获取相关文档"""
        try:
            # 1. 密集向量检索
            dense_docs = self._dense_retrieval(query)
            
            # 2. 稀疏BM25检索
            sparse_docs = self._sparse_retrieval(query)
            
            # 3. 融合结果
            hybrid_docs = self._fusion_retrieval(query, dense_docs, sparse_docs)
            
            logger.info(f"混合检索完成: 密集检索{len(dense_docs)}个, 稀疏检索{len(sparse_docs)}个, 融合后{len(hybrid_docs)}个")
            
            # 4. 重排序（如果启用）
            if self.use_reranking and self.reranking_service and hybrid_docs:
                try:
                    reranked_docs = self.reranking_service.rerank_documents(
                        hybrid_docs, query, reranker_type='default'
                    )
                    logger.info(f"重排序后返回 {len(reranked_docs)} 个文档")
                    return reranked_docs[:self.k]
                except Exception as e:
                    logger.warning(f"重排序失败，使用原始结果: {e}")
            
            return hybrid_docs[:self.k]
            
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            return []
    
    def _dense_retrieval(self, query: str) -> List[Tuple[Document, float]]:
        """密集向量检索"""
        try:
            if self.category:
                # 类别过滤的检索
                all_docs = self.vector_store.similarity_search_with_score(query, k=self.k * 2)
                filtered_docs = []
                
                for doc, score in all_docs:
                    doc_category = doc.metadata.get('category', '')
                    if doc_category == self.category or self._is_category_match(doc, self.category):
                        filtered_docs.append((doc, score))
                
                return filtered_docs[:self.k]
            else:
                return self.vector_store.similarity_search_with_score(query, k=self.k)
                
        except Exception as e:
            logger.error(f"密集检索失败: {e}")
            return []
    
    def _sparse_retrieval(self, query: str) -> List[Tuple[Document, float]]:
        """稀疏BM25检索"""
        try:
            if not self.bm25:
                return []
            
            # 查询分词
            query_tokens = self._tokenize_chinese(query)
            if not query_tokens:
                return []
            
            # BM25检索
            scores = self.bm25.get_scores(query_tokens)
            
            # 获取top-k结果
            top_indices = np.argsort(scores)[::-1][:self.k]
            
            results = []
            for idx in top_indices:
                if idx in self.doc_mapping and scores[idx] > 0:
                    doc = self.doc_mapping[idx]
                    # 归一化BM25分数到0-1范围
                    normalized_score = min(scores[idx] / (scores[idx] + 1), 1.0)
                    results.append((doc, normalized_score))
            
            return results
            
        except Exception as e:
            logger.error(f"稀疏检索失败: {e}")
            return []
    
    def _fusion_retrieval(self, query: str, dense_docs: List[Tuple[Document, float]], sparse_docs: List[Tuple[Document, float]]) -> List[Document]:
        """融合密集和稀疏检索结果"""
        try:
            # 使用RRF (Reciprocal Rank Fusion) 算法
            doc_scores = defaultdict(float)
            doc_objects = {}
            
            # 处理密集检索结果
            for rank, (doc, score) in enumerate(dense_docs):
                doc_id = self._get_doc_id(doc)
                doc_objects[doc_id] = doc
                # RRF分数 + 原始分数权重
                rrf_score = 1.0 / (rank + 60)  # k=60是常用参数
                weighted_score = (1 - score) * self.dense_weight  # FAISS返回的是距离，需要转换
                doc_scores[doc_id] += rrf_score * self.dense_weight + weighted_score * 0.3
            
            # 处理稀疏检索结果
            for rank, (doc, score) in enumerate(sparse_docs):
                doc_id = self._get_doc_id(doc)
                doc_objects[doc_id] = doc
                # RRF分数 + 原始分数权重
                rrf_score = 1.0 / (rank + 60)
                weighted_score = score * self.sparse_weight
                doc_scores[doc_id] += rrf_score * self.sparse_weight + weighted_score * 0.3
            
            # 按分数排序
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 返回排序后的文档
            result_docs = []
            for doc_id, score in sorted_docs:
                if doc_id in doc_objects:
                    result_docs.append(doc_objects[doc_id])
            
            return result_docs
            
        except Exception as e:
            logger.error(f"结果融合失败: {e}")
            # 降级到密集检索结果
            return [doc for doc, score in dense_docs]
    
    def _get_doc_id(self, doc: Document) -> str:
        """获取文档唯一标识"""
        return doc.metadata.get('chunk_id', '') or doc.metadata.get('document_id', '') or str(hash(doc.page_content[:100]))

class HybridRetrievalService:
    """
    混合检索服务
    """
    
    def __init__(self, vector_service: Optional[VectorService] = None):
        self.vector_service = vector_service or VectorService.get_instance()
        self.hybrid_retrievers = {}  # 缓存不同类别的混合检索器
        self.reranking_service = None
        
        # 初始化重排序服务
        try:
            self.reranking_service = RerankingService()
            logger.info("混合检索服务的重排序功能初始化成功")
        except Exception as e:
            logger.warning(f"重排序服务初始化失败: {e}")
        
        logger.info("混合检索服务初始化完成")
    
    def get_hybrid_retriever(self, category: Optional[str] = None, **kwargs) -> Optional[HybridRetriever]:
        """获取混合检索器"""
        try:
            use_reranking = kwargs.get('use_reranking', True)
            cache_key = f"{category or 'general'}_{use_reranking}"
            
            # 检查缓存
            if cache_key in self.hybrid_retrievers:
                return self.hybrid_retrievers[cache_key]
            
            # 获取向量存储
            if not self.vector_service.vector_store:
                logger.error("向量存储未初始化")
                return None
            
            # 获取所有文档
            all_docs = self.vector_service.vector_store.similarity_search("测试", k=1000)
            
            if not all_docs:
                logger.warning("没有找到任何文档")
                return None
            
            # 创建混合检索器
            hybrid_retriever = HybridRetriever(
                vector_store=self.vector_service.vector_store,
                documents=all_docs,
                category=category,
                dense_weight=kwargs.get('dense_weight', 0.7),
                sparse_weight=kwargs.get('sparse_weight', 0.3),
                k=kwargs.get('k', settings.retrieval_k),
                use_reranking=use_reranking
            )
            
            # 缓存检索器
            self.hybrid_retrievers[cache_key] = hybrid_retriever
            
            logger.info(f"创建混合检索器成功: category={category}, reranking={use_reranking}")
            return hybrid_retriever
            
        except Exception as e:
            logger.error(f"创建混合检索器失败: {e}")
            return None
    
    def search_hybrid(self, query: str, category: Optional[str] = None, k: int = None) -> List[Document]:
        """混合检索搜索"""
        try:
            k = k or settings.retrieval_k
            
            # 获取混合检索器
            retriever = self.get_hybrid_retriever(category, k=k)
            if not retriever:
                logger.error("无法获取混合检索器")
                return []
            
            # 执行检索
            logger.info(f"开始混合检索: query='{query[:50]}...', category={category}, k={k}")
            docs = retriever.get_relevant_documents(query)
            
            logger.info(f"混合检索完成，返回{len(docs)}个文档")
            return docs
            
        except Exception as e:
            logger.error(f"混合检索搜索失败: {e}")
            return []
    
    def clear_cache(self):
        """清除检索器缓存"""
        self.hybrid_retrievers.clear()
        logger.info("混合检索器缓存已清除")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        return {
            'cached_retrievers': list(self.hybrid_retrievers.keys()),
            'vector_service_stats': self.vector_service.get_vector_store_stats() if self.vector_service else None
        }
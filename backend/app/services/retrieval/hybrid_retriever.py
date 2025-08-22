from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np
from collections import defaultdict
import jieba
import re

from langchain.schema import Document
from rank_bm25 import BM25Okapi

from app.services.retrieval.base_retriever import (
    BaseRetriever, RetrievalStrategy, RetrievalResult, RetrievalConfig
)
from app.services.retrieval.vector_retriever import VectorRetriever, VectorRetrievalConfig
from app.services.reranking_service import RerankingService
from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class HybridRetrievalConfig(RetrievalConfig):
    """混合检索配置"""
    # 向量检索配置
    vector_config: VectorRetrievalConfig = None
    
    # 权重配置
    dense_weight: float = 0.7  # 向量搜索权重
    sparse_weight: float = 0.3  # BM25权重
    
    # BM25配置
    bm25_k: int = 20  # BM25检索数量
    
    # 重排序配置
    use_reranking: bool = True
    reranking_model: str = "bge-reranker-base"
    reranking_top_k: int = 10
    
    # 融合配置
    fusion_method: str = "rrf"  # rrf (Reciprocal Rank Fusion) 或 weighted
    rrf_constant: int = 60
    
    def __post_init__(self):
        if self.vector_config is None:
            self.vector_config = VectorRetrievalConfig(
                k=self.k,
                similarity_threshold=self.similarity_threshold,
                category=self.category,
                enable_cache=self.enable_cache
            )

class HybridRetriever(BaseRetriever):
    """混合检索器：结合向量搜索和BM25关键词搜索"""
    
    def __init__(self, config: HybridRetrievalConfig, documents: Optional[List[Document]] = None):
        super().__init__(config)
        self.hybrid_config = config
        
        # 初始化向量检索器
        self.vector_retriever = VectorRetriever(config.vector_config)
        
        # 初始化重排序服务
        self.reranking_service = None
        if config.use_reranking:
            try:
                self.reranking_service = RerankingService()
                logger.info("重排序服务初始化成功")
            except Exception as e:
                logger.warning(f"重排序服务初始化失败: {e}，将不使用重排序")
                self.hybrid_config.use_reranking = False
        
        # BM25相关
        self.bm25 = None
        self.doc_mapping = {}  # BM25索引到文档的映射
        self.documents = documents or []
        
        # 如果提供了文档，立即初始化BM25
        if self.documents:
            self._initialize_bm25()
        
        logger.info(f"混合检索器初始化成功: dense_weight={config.dense_weight}, sparse_weight={config.sparse_weight}")
    
    def _get_strategy(self) -> RetrievalStrategy:
        return RetrievalStrategy.HYBRID
    
    def set_documents(self, documents: List[Document]):
        """设置文档并初始化BM25"""
        self.documents = documents
        self._initialize_bm25()
    
    def _initialize_bm25(self):
        """初始化BM25检索器"""
        try:
            corpus = []
            self.doc_mapping = {}
            
            for i, doc in enumerate(self.documents):
                # 过滤类别（如果指定）
                if self.hybrid_config.category:
                    doc_category = doc.metadata.get('category', '')
                    if doc_category != self.hybrid_config.category and not self._is_category_match(doc, self.hybrid_config.category):
                        continue
                
                # 准备文本
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
        stop_words = {
            '的', '是', '在', '了', '和', '有', '为', '与', '等', '及', '或', '但', '而', '也', '都', 
            '要', '可', '能', '会', '将', '应', '应该', '需要', '进行', '实施', '执行', '完成', '通过',
            '根据', '按照', '依据', '基于', '关于', '对于', '由于', '因为', '所以', '因此', '然后',
            '如果', '如何', '什么', '怎么', '哪些', '这些', '那些', '这个', '那个', '一个', '每个'
        }
        tokens = [token.strip() for token in tokens if len(token.strip()) > 1 and token.strip() not in stop_words]
        
        return tokens
    
    def _is_category_match(self, doc: Document, target_category: str) -> bool:
        """检查文档是否匹配目标类别"""
        title = doc.metadata.get('title', '').lower()
        content = doc.page_content.lower()
        
        category_keywords = {
            'development': ['开发', '程序', '代码', '技术', '软件', '系统', '编程', '算法'],
            'manual': ['手册', '规范', '标准', '质量', '管理', '体系', '文件', '制度'],
            'procedure': ['流程', '程序', '操作', '作业', '指导', '步骤', '工艺', '方法'],
            'policy': ['政策', '规章', '规定', '条例', '法规', '制度'],
            'guideline': ['指南', '指导', '建议', '推荐', '要求', '准则']
        }
        
        keywords = category_keywords.get(target_category, [])
        return any(keyword in title or keyword in content for keyword in keywords)
    
    def _retrieve_documents(self, query: str, **kwargs) -> RetrievalResult:
        """执行混合检索"""
        k = kwargs.get('k', self.hybrid_config.k)
        
        try:
            # 1. 向量检索
            logger.debug("执行向量检索...")
            vector_result = self.vector_retriever.retrieve(query, k=k * 2)  # 获取更多候选
            dense_docs = list(zip(vector_result.documents, vector_result.scores or [0.0] * len(vector_result.documents)))
            
            # 2. BM25检索
            logger.debug("执行BM25检索...")
            sparse_docs = self._bm25_search(query, k * 2)
            
            # 3. 融合结果
            logger.debug("融合检索结果...")
            fused_docs = self._fusion_retrieval(query, dense_docs, sparse_docs, k)
            
            # 4. 重排序（可选）
            if self.hybrid_config.use_reranking and self.reranking_service and len(fused_docs) > 1:
                logger.debug("执行重排序...")
                fused_docs = self._rerank_documents(query, fused_docs)
            
            # 5. 截取最终结果
            final_docs = fused_docs[:k]
            documents = [doc for doc, score in final_docs]
            scores = [score for doc, score in final_docs]
            
            # 构建元数据
            metadata = {
                "fusion_method": self.hybrid_config.fusion_method,
                "dense_weight": self.hybrid_config.dense_weight,
                "sparse_weight": self.hybrid_config.sparse_weight,
                "dense_results": len(dense_docs),
                "sparse_results": len(sparse_docs),
                "fused_results": len(fused_docs),
                "final_results": len(final_docs),
                "reranking_used": self.hybrid_config.use_reranking and self.reranking_service is not None,
                "vector_retrieval_metadata": vector_result.metadata
            }
            
            return RetrievalResult(
                documents=documents,
                scores=scores,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            raise
    
    def _bm25_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """BM25搜索"""
        if self.bm25 is None:
            logger.warning("BM25检索器未初始化")
            return []
        
        try:
            # 分词查询
            query_tokens = self._tokenize_chinese(query)
            if not query_tokens:
                logger.warning("查询分词结果为空")
                return []
            
            # 获取BM25分数
            scores = self.bm25.get_scores(query_tokens)
            
            # 获取top-k结果
            top_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if idx in self.doc_mapping and scores[idx] > 0:
                    doc = self.doc_mapping[idx]
                    score = float(scores[idx])
                    results.append((doc, score))
            
            logger.debug(f"BM25检索结果: {len(results)} 个文档")
            return results
            
        except Exception as e:
            logger.error(f"BM25搜索失败: {e}")
            return []
    
    def _fusion_retrieval(self, query: str, dense_docs: List[Tuple[Document, float]], 
                         sparse_docs: List[Tuple[Document, float]], k: int) -> List[Tuple[Document, float]]:
        """融合向量检索和BM25检索结果"""
        if self.hybrid_config.fusion_method == "rrf":
            return self._rrf_fusion(dense_docs, sparse_docs, k)
        else:
            return self._weighted_fusion(dense_docs, sparse_docs, k)
    
    def _rrf_fusion(self, dense_docs: List[Tuple[Document, float]], 
                   sparse_docs: List[Tuple[Document, float]], k: int) -> List[Tuple[Document, float]]:
        """Reciprocal Rank Fusion (RRF)融合"""
        doc_scores = defaultdict(float)
        doc_objects = {}
        
        # 处理向量检索结果
        for rank, (doc, score) in enumerate(dense_docs):
            doc_id = self._get_doc_id(doc)
            doc_scores[doc_id] += self.hybrid_config.dense_weight / (self.hybrid_config.rrf_constant + rank + 1)
            doc_objects[doc_id] = doc
        
        # 处理BM25检索结果
        for rank, (doc, score) in enumerate(sparse_docs):
            doc_id = self._get_doc_id(doc)
            doc_scores[doc_id] += self.hybrid_config.sparse_weight / (self.hybrid_config.rrf_constant + rank + 1)
            doc_objects[doc_id] = doc
        
        # 排序并返回结果
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_docs[:k * 2]:  # 获取更多候选用于后续处理
            if doc_id in doc_objects:
                results.append((doc_objects[doc_id], score))
        
        return results
    
    def _weighted_fusion(self, dense_docs: List[Tuple[Document, float]], 
                        sparse_docs: List[Tuple[Document, float]], k: int) -> List[Tuple[Document, float]]:
        """加权融合"""
        doc_scores = defaultdict(float)
        doc_objects = {}
        
        # 归一化分数
        if dense_docs:
            max_dense_score = max(score for _, score in dense_docs)
            min_dense_score = min(score for _, score in dense_docs)
            dense_range = max_dense_score - min_dense_score if max_dense_score != min_dense_score else 1
        
        if sparse_docs:
            max_sparse_score = max(score for _, score in sparse_docs)
            min_sparse_score = min(score for _, score in sparse_docs)
            sparse_range = max_sparse_score - min_sparse_score if max_sparse_score != min_sparse_score else 1
        
        # 处理向量检索结果
        for doc, score in dense_docs:
            doc_id = self._get_doc_id(doc)
            normalized_score = (score - min_dense_score) / dense_range if dense_docs else 0
            doc_scores[doc_id] += self.hybrid_config.dense_weight * normalized_score
            doc_objects[doc_id] = doc
        
        # 处理BM25检索结果
        for doc, score in sparse_docs:
            doc_id = self._get_doc_id(doc)
            normalized_score = (score - min_sparse_score) / sparse_range if sparse_docs else 0
            doc_scores[doc_id] += self.hybrid_config.sparse_weight * normalized_score
            doc_objects[doc_id] = doc
        
        # 排序并返回结果
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_docs[:k * 2]:  # 获取更多候选用于后续处理
            if doc_id in doc_objects:
                results.append((doc_objects[doc_id], score))
        
        return results
    
    def _rerank_documents(self, query: str, docs_with_scores: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """重排序文档"""
        try:
            documents = [doc for doc, score in docs_with_scores]
            reranked_docs = self.reranking_service.rerank_documents(
                query, documents, top_k=self.hybrid_config.reranking_top_k
            )
            
            # 保持原有的分数结构，但使用重排序后的顺序
            reranked_with_scores = []
            for i, doc in enumerate(reranked_docs):
                # 使用重排序后的位置作为新分数（位置越靠前分数越高）
                new_score = 1.0 - (i / len(reranked_docs))
                reranked_with_scores.append((doc, new_score))
            
            logger.debug(f"重排序完成: {len(docs_with_scores)} -> {len(reranked_with_scores)} 个文档")
            return reranked_with_scores
            
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return docs_with_scores
    
    def _get_doc_id(self, doc: Document) -> str:
        """获取文档唯一标识"""
        # 尝试多种方式获取文档ID
        doc_id = doc.metadata.get('document_id')
        if doc_id:
            return str(doc_id)
        
        doc_id = doc.metadata.get('id')
        if doc_id:
            return str(doc_id)
        
        # 使用标题和内容的哈希作为ID
        title = doc.metadata.get('title', '')
        content_hash = hash(doc.page_content[:100])  # 使用前100个字符的哈希
        return f"{title}_{content_hash}"
    
    def get_stats(self) -> Dict[str, Any]:
        """获取混合检索器统计信息"""
        stats = super().get_stats()
        
        # 添加向量检索器统计
        stats["vector_retriever"] = self.vector_retriever.get_stats()
        
        # 添加BM25统计
        stats["bm25"] = {
            "initialized": self.bm25 is not None,
            "document_count": len(self.doc_mapping),
            "total_documents": len(self.documents)
        }
        
        # 添加重排序统计
        stats["reranking"] = {
            "enabled": self.hybrid_config.use_reranking,
            "service_available": self.reranking_service is not None
        }
        
        # 添加配置信息
        stats["config"].update({
            "dense_weight": self.hybrid_config.dense_weight,
            "sparse_weight": self.hybrid_config.sparse_weight,
            "fusion_method": self.hybrid_config.fusion_method,
            "rrf_constant": self.hybrid_config.rrf_constant,
            "reranking_top_k": self.hybrid_config.reranking_top_k
        })
        
        return stats
    
    def reset(self) -> None:
        """重置混合检索器"""
        super().reset()
        self.vector_retriever.reset()
        
        # 重置BM25
        self.bm25 = None
        self.doc_mapping = {}
        
        # 如果有文档，重新初始化BM25
        if self.documents:
            self._initialize_bm25()
        
        logger.info("混合检索器已重置")
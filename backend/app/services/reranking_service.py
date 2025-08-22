from typing import List, Optional, Dict, Any, Tuple
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba

from langchain.schema import Document

from app.core.config import settings

logger = logging.getLogger(__name__)

class TfidfReranker:
    """
    基于TF-IDF的重排序器
    """
    
    def __init__(self, 
                 top_k: int = 10,
                 score_threshold: float = 0.0):
        self.top_k = top_k
        self.score_threshold = score_threshold
        
        logger.info(f"TF-IDF重排序器初始化: top_k={top_k}, threshold={score_threshold}")
    
    def rerank_documents(self, documents: List[Document], query: str) -> List[Document]:
        """重排序文档"""
        try:
            if not documents:
                return documents
            
            if len(documents) <= 1:
                return documents
            
            logger.info(f"开始重排序 {len(documents)} 个文档")
            
            # 使用TF-IDF重排序
            reranked_docs = self._tfidf_rerank(documents, query)
            
            # 应用top_k和分数阈值过滤
            filtered_docs = reranked_docs[:self.top_k]
            
            logger.info(f"重排序完成，返回 {len(filtered_docs)} 个文档")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"文档重排序失败: {e}")
            return documents[:self.top_k]  # 降级返回原始顺序
    
    def _tfidf_rerank(self, documents: List[Document], query: str) -> List[Document]:
        """基于TF-IDF的文本相似度重排序"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import jieba
            
            # 准备文档文本
            doc_texts = []
            for doc in documents:
                title = doc.metadata.get('title', '')
                content = doc.page_content
                doc_text = f"{title} {content}" if title else content
                
                # 中文分词
                tokens = jieba.cut(doc_text)
                doc_texts.append(' '.join(tokens))
            
            # 查询分词
            query_tokens = ' '.join(jieba.cut(query))
            
            # 计算TF-IDF相似度
            vectorizer = TfidfVectorizer(max_features=1000)
            all_texts = [query_tokens] + doc_texts
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # 计算查询与文档的相似度
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            
            # 创建文档-分数对并排序
            doc_scores = list(zip(documents, similarities))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 过滤低分文档
            filtered_docs = [
                doc for doc, score in doc_scores 
                if score >= self.score_threshold
            ]
            
            logger.info(f"TF-IDF重排序完成，平均相似度: {np.mean(similarities):.3f}")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"TF-IDF重排序失败: {e}")
            return documents

class RerankingService:
    """
    重排序服务
    """
    
    def __init__(self):
        self.rerankers = {}
        self._initialize_rerankers()
        
        logger.info("重排序服务初始化完成")
    
    def _initialize_rerankers(self):
        """初始化不同类型的重排序器"""
        try:
            # 默认重排序器
            self.rerankers['default'] = TfidfReranker(
                top_k=settings.retrieval_k,
                score_threshold=0.1
            )
            
            # 高精度重排序器
            self.rerankers['high_precision'] = TfidfReranker(
                top_k=settings.retrieval_k // 2,
                score_threshold=0.3
            )
            
            # 高召回重排序器
            self.rerankers['high_recall'] = TfidfReranker(
                top_k=settings.retrieval_k * 2,
                score_threshold=0.0
            )
            
            logger.info(f"初始化了 {len(self.rerankers)} 个重排序器")
            
        except Exception as e:
            logger.error(f"重排序器初始化失败: {e}")
    
    def rerank_documents(self, 
                        documents: List[Document], 
                        query: str, 
                        reranker_type: str = 'default') -> List[Document]:
        """重排序文档"""
        try:
            if not documents:
                return documents
            
            reranker = self.rerankers.get(reranker_type)
            if not reranker:
                logger.warning(f"重排序器类型 '{reranker_type}' 不存在，使用默认重排序器")
                reranker = self.rerankers.get('default')
            
            if not reranker:
                logger.error("没有可用的重排序器")
                return documents
            
            logger.info(f"使用 {reranker_type} 重排序器处理 {len(documents)} 个文档")
            
            # 执行重排序
            reranked_docs = reranker.rerank_documents(documents, query)
            
            # 添加重排序元数据
            for i, doc in enumerate(reranked_docs):
                doc.metadata['rerank_position'] = i + 1
                doc.metadata['reranker_type'] = reranker_type
            
            logger.info(f"重排序完成，返回 {len(reranked_docs)} 个文档")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"文档重排序失败: {e}")
            return documents
    
    def get_available_rerankers(self) -> List[str]:
        """获取可用的重排序器类型"""
        return list(self.rerankers.keys())
    
    def get_reranker_stats(self) -> Dict[str, Any]:
        """获取重排序器统计信息"""
        stats = {}
        for name, reranker in self.rerankers.items():
            stats[name] = {
                'reranker_type': 'TF-IDF',
                'top_k': getattr(reranker, 'top_k', 0),
                'score_threshold': getattr(reranker, 'score_threshold', 0.0)
            }
        return stats
    
    def update_reranker_config(self, 
                              reranker_type: str, 
                              top_k: Optional[int] = None,
                              score_threshold: Optional[float] = None):
        """更新重排序器配置"""
        try:
            reranker = self.rerankers.get(reranker_type)
            if not reranker:
                logger.error(f"重排序器类型 '{reranker_type}' 不存在")
                return False
            
            if top_k is not None:
                reranker.top_k = top_k
                logger.info(f"更新 {reranker_type} 重排序器 top_k: {top_k}")
            
            if score_threshold is not None:
                reranker.score_threshold = score_threshold
                logger.info(f"更新 {reranker_type} 重排序器 score_threshold: {score_threshold}")
            
            return True
            
        except Exception as e:
            logger.error(f"更新重排序器配置失败: {e}")
            return False
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from app.services.retrieval.base_retriever import (
    BaseRetriever, RetrievalStrategy, RetrievalResult, RetrievalConfig
)
from app.services.vector_service import VectorService
from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class VectorRetrievalConfig(RetrievalConfig):
    """向量检索配置"""
    use_mmr: bool = True
    mmr_fetch_k: int = 100
    mmr_lambda_mult: float = 0.5
    search_type: str = "similarity"  # similarity, similarity_score_threshold, mmr
    filter_dict: Optional[Dict[str, Any]] = None
    score_threshold: Optional[float] = None
    # 新增：类别加权与手动降权配置
    category_weight_mode: str = "weight"  # 可选: "filter" | "weight"
    category_primary_boost: float = 1.25   # 目标类别直接匹配的加权
    category_mapped_boost: float = 1.1     # 智能映射匹配的加权
    category_mismatch_penalty: float = 0.9 # 非目标类别的轻度降权
    manual_downweight_keywords: Optional[Dict[str, float]] = None  # 关键字 -> 降权因子(0~1)

class VectorRetriever(BaseRetriever):
    """向量检索器实现"""
    
    def __init__(self, config: VectorRetrievalConfig, vector_service: Optional[VectorService] = None):
        super().__init__(config)
        self.vector_config = config
        self.vector_service = vector_service or VectorService.get_instance()
        
        # 验证向量服务是否可用
        if self.vector_service.vector_store is None:
            raise ValueError("向量数据库未初始化，无法创建向量检索器")
        
        logger.info(f"向量检索器初始化成功: k={config.k}, use_mmr={config.use_mmr}")
    
    def _get_strategy(self) -> RetrievalStrategy:
        return RetrievalStrategy.VECTOR
    
    def _retrieve_documents(self, query: str, **kwargs) -> RetrievalResult:
        """执行向量检索"""
        # 合并配置参数
        k = kwargs.get('k', self.vector_config.k)
        similarity_threshold = kwargs.get('similarity_threshold', self.vector_config.similarity_threshold)
        filter_dict = kwargs.get('filter_dict', self.vector_config.filter_dict)
        
        try:
            # 获取向量数据库统计信息
            total_vectors = self.vector_service.vector_store.index.ntotal
            logger.debug(f"向量数据库状态: 总计 {total_vectors} 个向量")
            
            # 选择检索方法
            if self.vector_config.use_mmr:
                documents_with_scores = self._mmr_search(query, k, filter_dict)
            else:
                documents_with_scores = self._similarity_search(query, k, similarity_threshold, filter_dict)
            
            # 提取文档和分数
            documents = [doc for doc, score in documents_with_scores]
            scores = [score for doc, score in documents_with_scores]
            
            # 应用相似度阈值过滤
            # 注意：FAISS默认返回L2距离（越小越相似），需要转换为相似度（越大越相似）
            if similarity_threshold is not None:
                filtered_results = []
                filtered_scores = []
                for doc, score in zip(documents, scores):
                    # 将FAISS的L2距离转换为相似度分数
                    # 对于归一化向量，L2距离范围是[0,2]，转换为相似度：similarity = 1 - distance/2
                    similarity_score = 1.0 - min(score / 2.0, 1.0)  # 确保相似度在[0,1]范围内
                    if similarity_score >= similarity_threshold:
                        filtered_results.append(doc)
                        filtered_scores.append(similarity_score)  # 存储转换后的相似度分数
                documents = filtered_results
                scores = filtered_scores
            
            # 类别加权/过滤
            if self.vector_config.category:
                mode = (self.vector_config.category_weight_mode or "weight").lower()
                if mode == "filter":
                    # 保持原有严格过滤逻辑
                    documents, scores = self._filter_by_category(documents, scores, self.vector_config.category)
                else:
                    # 使用加权方式调整分数
                    documents, scores = self._apply_category_weighting(documents, scores, self.vector_config.category)
            
            # 应用手动降权（基于关键词）
            if self.vector_config.manual_downweight_keywords:
                documents, scores = self._apply_manual_downweight(documents, scores, self.vector_config.manual_downweight_keywords)
            
            # 构建元数据
            metadata = {
                "search_type": "mmr" if self.vector_config.use_mmr else "similarity",
                "total_vectors": total_vectors,
                "original_results": len(documents_with_scores),
                "filtered_results": len(documents),
                "similarity_threshold": similarity_threshold,
                "category_filter": self.vector_config.category,
                "filter_dict": filter_dict,
                "category_weight_mode": self.vector_config.category_weight_mode,
            }
            
            if self.vector_config.use_mmr:
                metadata.update({
                    "mmr_fetch_k": self.vector_config.mmr_fetch_k,
                    "mmr_lambda_mult": self.vector_config.mmr_lambda_mult
                })
            
            return RetrievalResult(
                documents=documents,
                scores=scores,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            raise
    
    def _similarity_search(self, query: str, k: int, similarity_threshold: Optional[float], 
                          filter_dict: Optional[Dict[str, Any]]) -> List[Tuple[Document, float]]:
        """相似度搜索"""
        logger.debug(f"执行相似度搜索: k={k}, threshold={similarity_threshold}")
        
        # 根据是否有过滤条件调整搜索数量
        search_k = k
        if filter_dict or self.vector_config.category:
            search_k = min(
                self.vector_service.vector_store.index.ntotal,
                k * settings.filter_search_multiplier
            )
        
        # 执行搜索
        docs_with_scores = self.vector_service.vector_store.similarity_search_with_score(
            query, k=search_k
        )
        
        # 将FAISS的L2距离转换为相似度分数
        converted_docs_with_scores = []
        for doc, distance in docs_with_scores:
            # 对于归一化向量，L2距离范围是[0,2]，转换为相似度：similarity = 1 - distance/2
            similarity_score = 1.0 - min(distance / 2.0, 1.0)
            converted_docs_with_scores.append((doc, similarity_score))
        
        return converted_docs_with_scores
    
    def _mmr_search(self, query: str, k: int, filter_dict: Optional[Dict[str, Any]]) -> List[Tuple[Document, float]]:
        """MMR搜索"""
        logger.debug(f"执行MMR搜索: k={k}, fetch_k={self.vector_config.mmr_fetch_k}, lambda={self.vector_config.mmr_lambda_mult}")
        
        # 根据是否有过滤条件调整fetch_k
        fetch_k = self.vector_config.mmr_fetch_k
        if filter_dict or self.vector_config.category:
            fetch_k = min(
                self.vector_service.vector_store.index.ntotal,
                max(1000, fetch_k * settings.filter_search_multiplier)
            )
        
        # 执行MMR搜索
        documents = self.vector_service.vector_store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=self.vector_config.mmr_lambda_mult
        )
        
        # MMR不直接返回分数，需要单独计算
        # 注意：使用与入库时一致的前缀拼接文本来计算相似度
        docs_with_scores = []
        for doc in documents:
            try:
                # 计算相似度分数 - 使用与入库时一致的前缀拼接文本
                embedding = self.vector_service.create_single_embedding(query)
                # 使用与入库时一致的前缀拼接文本
                prefixed_text = self.vector_service._build_prefixed_text(doc)
                doc_embedding = self.vector_service.create_single_embedding(prefixed_text)
                # 计算余弦相似度
                similarity = np.dot(embedding, doc_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(doc_embedding)
                )
                docs_with_scores.append((doc, float(similarity)))
            except Exception as e:
                logger.warning(f"计算文档相似度失败: {e}")
                docs_with_scores.append((doc, 0.0))
        
        return docs_with_scores
    
    def _filter_by_category(self, documents: List[Document], scores: List[float], 
                           category: str) -> Tuple[List[Document], List[float]]:
        """按类别过滤文档"""
        filtered_docs = []
        filtered_scores = []
        
        for doc, score in zip(documents, scores):
            doc_category = doc.metadata.get('category', '')
            
            # 直接类别匹配
            if doc_category == category:
                filtered_docs.append(doc)
                filtered_scores.append(score)
                continue
            
            # 智能类别映射（对于'other'类别的文档）
            if doc_category == 'other':
                mapped_category = self._map_document_to_category(
                    doc.metadata.get('title', ''), 
                    doc.page_content
                )
                if mapped_category == category:
                    filtered_docs.append(doc)
                    filtered_scores.append(score)
        
        logger.debug(f"类别过滤: {category} - {len(documents)} -> {len(filtered_docs)} 个文档")
        return filtered_docs, filtered_scores
    
    def _apply_category_weighting(self, documents: List[Document], scores: List[float], target_category: str) -> Tuple[List[Document], List[float]]:
        """按类别对分数进行加权而非严格过滤"""
        if not documents:
            return documents, scores
        
        adjusted = []
        for doc, score in zip(documents, scores):
            new_score = score
            doc_category = doc.metadata.get('category', '')
            matched = False
            if doc_category == target_category:
                new_score *= self.vector_config.category_primary_boost
                matched = True
            elif doc_category == 'other':
                mapped_category = self._map_document_to_category(
                    doc.metadata.get('title', ''),
                    doc.page_content
                )
                if mapped_category == target_category:
                    new_score *= self.vector_config.category_mapped_boost
                    matched = True
            if not matched:
                new_score *= self.vector_config.category_mismatch_penalty
            adjusted.append((doc, new_score))
        
        # 按加权后的分数重新排序（降序）
        adjusted.sort(key=lambda x: x[1], reverse=True)
        documents = [d for d, s in adjusted]
        scores = [s for d, s in adjusted]
        return documents, scores
    
    def _apply_manual_downweight(self, documents: List[Document], scores: List[float], rules: Dict[str, float]) -> Tuple[List[Document], List[float]]:
        """根据手动降权关键字对分数进行降权，关键字命中title/source/content时生效"""
        if not documents:
            return documents, scores
        
        adjusted = []
        for doc, score in zip(documents, scores):
            new_score = score
            title = (doc.metadata.get('title') or '').lower()
            source = (doc.metadata.get('source') or '').lower()
            content = (doc.page_content or '').lower()
            for kw, factor in rules.items():
                try:
                    f = float(factor)
                except Exception:
                    f = 1.0
                if f <= 0:
                    f = 0.1
                kw_l = str(kw).lower()
                if kw_l and (kw_l in title or kw_l in source or kw_l in content):
                    new_score *= f
            adjusted.append((doc, new_score))
        
        # 若分数有变化，进行稳定排序（按分数降序）
        adjusted.sort(key=lambda x: x[1], reverse=True)
        documents = [d for d, s in adjusted]
        scores = [s for d, s in adjusted]
        return documents, scores
    
    def _map_document_to_category(self, title: str, content: str) -> str:
        """智能文档类别映射"""
        title_lower = title.lower()
        content_lower = content.lower()
        
        category_keywords = {
            'development': ['开发', '程序', '代码', '技术', '软件', '系统', '编程', '算法'],
            'manual': ['手册', '规范', '标准', '质量', '管理', '体系', '文件', '制度'],
            'procedure': ['流程', '程序', '操作', '作业', '指导', '步骤', '工艺', '方法'],
            'policy': ['政策', '规章', '规定', '条例', '法规', '制度'],
            'guideline': ['指南', '指导', '建议', '推荐', '要求', '准则']
        }
        
        # 计算每个类别的匹配分数
        category_scores = {}
        for cat, keywords in category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in title_lower:
                    score += 3  # 标题匹配权重更高
                if keyword in content_lower:
                    score += 1
            category_scores[cat] = score
        
        # 返回得分最高的类别
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] > 0:
                return best_category
        
        return 'other'
    
    def get_stats(self) -> Dict[str, Any]:
        """获取向量检索器统计信息"""
        stats = super().get_stats()
        
        # 添加向量服务统计信息
        if self.vector_service.vector_store:
            vector_stats = self.vector_service.get_vector_store_stats()
            stats["vector_store"] = vector_stats
        
        # 添加配置信息
        stats["config"].update({
            "use_mmr": self.vector_config.use_mmr,
            "mmr_fetch_k": self.vector_config.mmr_fetch_k,
            "mmr_lambda_mult": self.vector_config.mmr_lambda_mult,
            "search_type": self.vector_config.search_type,
            "score_threshold": self.vector_config.score_threshold,
            "category_weight_mode": self.vector_config.category_weight_mode,
            "category_primary_boost": self.vector_config.category_primary_boost,
            "category_mapped_boost": self.vector_config.category_mapped_boost,
            "category_mismatch_penalty": self.vector_config.category_mismatch_penalty,
            "manual_downweight_keywords": list(self.vector_config.manual_downweight_keywords.keys()) if self.vector_config.manual_downweight_keywords else []
        })
        
        return stats
    
    def reset(self) -> None:
        """重置向量检索器"""
        super().reset()
        # 向量检索器通常不需要额外的重置操作
        logger.info("向量检索器已重置")
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import numpy as np

from langchain.schema import Document

from app.services.retrieval.base_retriever import (
    BaseRetriever, RetrievalStrategy, RetrievalResult, RetrievalConfig
)
from app.services.retrieval.vector_retriever import VectorRetriever, VectorRetrievalConfig
from app.services.retrieval.hybrid_retriever import HybridRetriever, HybridRetrievalConfig
from app.services.retrieval.multi_query_retriever import MultiQueryRetriever, MultiQueryRetrievalConfig
from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class EnsembleRetrievalConfig(RetrievalConfig):
    """集成检索配置"""
    # 检索器配置
    retrievers: List[Dict[str, Any]] = field(default_factory=list)
    
    # 权重配置
    weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])  # 默认权重
    
    # 融合配置
    fusion_method: str = "weighted_sum"  # weighted_sum, rrf, borda_count
    rrf_constant: int = 60
    
    # 并发配置
    max_workers: int = 3
    
    # 结果配置
    enable_score_normalization: bool = True
    min_score_threshold: float = 0.0
    
    def __post_init__(self):
        # 如果没有配置检索器，使用默认配置
        if not self.retrievers:
            self.retrievers = self._get_default_retrievers()
        
        # 确保权重数量与检索器数量匹配
        if len(self.weights) != len(self.retrievers):
            logger.warning(f"权重数量({len(self.weights)})与检索器数量({len(self.retrievers)})不匹配，使用平均权重")
            self.weights = [1.0 / len(self.retrievers)] * len(self.retrievers)
        
        # 归一化权重
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
    
    def _get_default_retrievers(self) -> List[Dict[str, Any]]:
        """获取默认检索器配置"""
        return [
            {
                "type": "vector",
                "config": {
                    "k": self.k,
                    "similarity_threshold": self.similarity_threshold,
                    "category": self.category,
                    "use_mmr": True,
                    "mmr_lambda_mult": 0.5,
                    # 传入默认的向量类别加权与手动降权参数
                    "category_weight_mode": settings.vector_category_weight_mode,
                    "category_primary_boost": settings.vector_category_primary_boost,
                    "category_mapped_boost": settings.vector_category_mapped_boost,
                    "category_mismatch_penalty": settings.vector_category_mismatch_penalty,
                    "manual_downweight_keywords": settings.vector_manual_downweight_keywords,
                }
            },
            {
                "type": "hybrid",
                "config": {
                    "k": self.k,
                    "similarity_threshold": self.similarity_threshold,
                    "category": self.category,
                    "dense_weight": 0.7,
                    "sparse_weight": 0.3,
                    "use_reranking": True,
                    # 让内部向量检索继承类别加权与手动降权参数
                    "category_weight_mode": settings.vector_category_weight_mode,
                    "category_primary_boost": settings.vector_category_primary_boost,
                    "category_mapped_boost": settings.vector_category_mapped_boost,
                    "category_mismatch_penalty": settings.vector_category_mismatch_penalty,
                    "manual_downweight_keywords": settings.vector_manual_downweight_keywords,
                }
            },
            {
                "type": "multi_query",
                "config": {
                    "k": self.k,
                    "similarity_threshold": self.similarity_threshold,
                    "category": self.category,
                    "num_queries": 3,
                    "docs_per_query": 8,
                    # 让内部向量检索继承类别加权与手动降权参数
                    "category_weight_mode": settings.vector_category_weight_mode,
                    "category_primary_boost": settings.vector_category_primary_boost,
                    "category_mapped_boost": settings.vector_category_mapped_boost,
                    "category_mismatch_penalty": settings.vector_category_mismatch_penalty,
                    "manual_downweight_keywords": settings.vector_manual_downweight_keywords,
                }
            }
        ]

class EnsembleRetriever(BaseRetriever):
    """集成检索器：组合多种检索策略并融合结果"""
    
    def __init__(self, config: EnsembleRetrievalConfig, documents: Optional[List[Document]] = None):
        super().__init__(config)
        self.ensemble_config = config
        self.documents = documents or []
        
        # 初始化检索器
        self.retrievers = []
        self._initialize_retrievers()
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        logger.info(f"集成检索器初始化成功: {len(self.retrievers)} 个检索器, 权重: {config.weights}")
    
    def _get_strategy(self) -> RetrievalStrategy:
        return RetrievalStrategy.ENSEMBLE
    
    def _initialize_retrievers(self):
        """初始化所有检索器"""
        for i, retriever_config in enumerate(self.ensemble_config.retrievers):
            try:
                retriever = self._create_retriever(retriever_config)
                if retriever:
                    self.retrievers.append(retriever)
                    logger.debug(f"检索器 {i+1} ({retriever_config['type']}) 初始化成功")
                else:
                    logger.warning(f"检索器 {i+1} ({retriever_config['type']}) 初始化失败")
            except Exception as e:
                logger.error(f"检索器 {i+1} ({retriever_config['type']}) 初始化失败: {e}")
        
        if not self.retrievers:
            raise ValueError("没有可用的检索器")
        
        # 调整权重以匹配实际检索器数量
        if len(self.retrievers) != len(self.ensemble_config.weights):
            self.ensemble_config.weights = [1.0 / len(self.retrievers)] * len(self.retrievers)
            logger.info(f"调整权重以匹配检索器数量: {self.ensemble_config.weights}")
    
    def _create_retriever(self, retriever_config: Dict[str, Any]) -> Optional[BaseRetriever]:
        """创建单个检索器"""
        retriever_type = retriever_config.get("type")
        config_dict = retriever_config.get("config", {})
        
        try:
            if retriever_type == "vector":
                config = VectorRetrievalConfig(**config_dict)
                return VectorRetriever(config)
            
            elif retriever_type == "hybrid":
                # 为混合检索器创建向量配置
                base_vector_params: Dict[str, Any] = {
                    "k": config_dict.get("k", self.ensemble_config.k),
                    "similarity_threshold": config_dict.get("similarity_threshold", self.ensemble_config.similarity_threshold),
                    "category": config_dict.get("category", self.ensemble_config.category),
                    "enable_cache": config_dict.get("enable_cache", self.ensemble_config.enable_cache),
                }
                # 允许的向量参数键
                allowed_vector_keys = {
                    "use_mmr", "mmr_fetch_k", "mmr_lambda_mult", "search_type", "filter_dict", "score_threshold",
                    "category_weight_mode", "category_primary_boost", "category_mapped_boost", "category_mismatch_penalty",
                    "manual_downweight_keywords",
                }
                for key in allowed_vector_keys:
                    if key in config_dict:
                        base_vector_params[key] = config_dict[key]
                vector_config = VectorRetrievalConfig(**base_vector_params)
                # 构造 HybridRetrievalConfig 的参数副本，并移除向量专用键
                hybrid_params: Dict[str, Any] = dict(config_dict)
                for key in allowed_vector_keys:
                    if key in hybrid_params:
                        hybrid_params.pop(key)
                hybrid_params["vector_config"] = vector_config
                config = HybridRetrievalConfig(**hybrid_params)
                retriever = HybridRetriever(config, self.documents)
                return retriever
            
            elif retriever_type == "multi_query":
                # 为多查询检索器创建向量配置
                base_vector_params: Dict[str, Any] = {
                    "k": config_dict.get("docs_per_query", 10),
                    "similarity_threshold": config_dict.get("similarity_threshold", self.ensemble_config.similarity_threshold),
                    "category": config_dict.get("category", self.ensemble_config.category),
                    "enable_cache": config_dict.get("enable_cache", self.ensemble_config.enable_cache),
                }
                allowed_vector_keys = {
                    "use_mmr", "mmr_fetch_k", "mmr_lambda_mult", "search_type", "filter_dict", "score_threshold",
                    "category_weight_mode", "category_primary_boost", "category_mapped_boost", "category_mismatch_penalty",
                    "manual_downweight_keywords",
                }
                for key in allowed_vector_keys:
                    if key in config_dict:
                        base_vector_params[key] = config_dict[key]
                vector_config = VectorRetrievalConfig(**base_vector_params)
                # 构造 MultiQueryRetrievalConfig 的参数副本，并移除向量专用键
                mq_params: Dict[str, Any] = dict(config_dict)
                for key in allowed_vector_keys:
                    if key in mq_params:
                        mq_params.pop(key)
                mq_params["vector_config"] = vector_config
                config = MultiQueryRetrievalConfig(**mq_params)
                return MultiQueryRetriever(config)
            
            else:
                logger.error(f"不支持的检索器类型: {retriever_type}")
                return None
                
        except Exception as e:
            logger.error(f"创建检索器失败 ({retriever_type}): {e}")
            return None
    
    def set_documents(self, documents: List[Document]):
        """设置文档"""
        self.documents = documents
        
        # 更新需要文档的检索器
        for retriever in self.retrievers:
            if hasattr(retriever, 'set_documents'):
                retriever.set_documents(documents)
    
    def _retrieve_documents(self, query: str, **kwargs) -> RetrievalResult:
        """执行集成检索"""
        k = kwargs.get('k', self.ensemble_config.k)
        
        try:
            # 1. 并行检索
            logger.debug(f"并行执行 {len(self.retrievers)} 个检索器...")
            retrieval_results = self._parallel_retrieve(query, k)
            
            # 2. 融合结果
            logger.debug("融合检索结果...")
            fused_docs = self._fuse_results(query, retrieval_results, k)
            
            # 3. 后处理
            final_docs = self._post_process_results(fused_docs, k)
            
            # 构建元数据
            metadata = self._build_metadata(query, retrieval_results, fused_docs, final_docs)
            
            documents = [doc for doc, score in final_docs]
            scores = [score for doc, score in final_docs]
            
            return RetrievalResult(
                documents=documents,
                scores=scores,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"集成检索失败: {e}")
            raise
    
    def _parallel_retrieve(self, query: str, k: int) -> List[RetrievalResult]:
        """并行检索"""
        results = []
        
        # 使用线程池并行检索
        future_to_retriever = {
            self.executor.submit(self._single_retrieve, retriever, query, k): i 
            for i, retriever in enumerate(self.retrievers)
        }
        
        # 按顺序收集结果
        retriever_results = [None] * len(self.retrievers)
        
        for future in as_completed(future_to_retriever):
            retriever_idx = future_to_retriever[future]
            try:
                result = future.result()
                retriever_results[retriever_idx] = result
                logger.debug(f"检索器 {retriever_idx+1} 完成，获得 {len(result.documents)} 个文档")
            except Exception as e:
                logger.error(f"检索器 {retriever_idx+1} 失败: {e}")
                retriever_results[retriever_idx] = RetrievalResult(documents=[], scores=[], metadata={})
        
        return retriever_results
    
    def _single_retrieve(self, retriever: BaseRetriever, query: str, k: int) -> RetrievalResult:
        """单个检索器检索"""
        return retriever.retrieve(query, k=k * 2)  # 获取更多候选用于融合
    
    def _fuse_results(self, query: str, results: List[RetrievalResult], k: int) -> List[Tuple[Document, float]]:
        """融合多个检索结果"""
        if self.ensemble_config.fusion_method == "weighted_sum":
            return self._weighted_sum_fusion(results, k)
        elif self.ensemble_config.fusion_method == "rrf":
            return self._rrf_fusion(results, k)
        elif self.ensemble_config.fusion_method == "borda_count":
            return self._borda_count_fusion(results, k)
        else:
            logger.warning(f"未知的融合方法: {self.ensemble_config.fusion_method}，使用加权求和")
            return self._weighted_sum_fusion(results, k)
    
    def _weighted_sum_fusion(self, results: List[RetrievalResult], k: int) -> List[Tuple[Document, float]]:
        """加权求和融合"""
        doc_scores = defaultdict(float)
        doc_objects = {}
        
        for i, result in enumerate(results):
            weight = self.ensemble_config.weights[i]
            
            # 归一化分数
            scores = result.scores or [0.0] * len(result.documents)
            if scores and self.ensemble_config.enable_score_normalization:
                max_score = max(scores) if scores else 1.0
                min_score = min(scores) if scores else 0.0
                score_range = max_score - min_score if max_score != min_score else 1.0
                scores = [(s - min_score) / score_range for s in scores]
            
            for j, doc in enumerate(result.documents):
                doc_id = self._get_doc_id(doc)
                score = scores[j] if j < len(scores) else 0.0
                
                doc_scores[doc_id] += weight * score
                doc_objects[doc_id] = doc
        
        # 排序并返回
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for doc_id, score in sorted_docs:
            if score >= self.ensemble_config.min_score_threshold:
                fused_results.append((doc_objects[doc_id], score))
        
        return fused_results[:k * 2]  # 返回更多候选用于后处理
    
    def _rrf_fusion(self, results: List[RetrievalResult], k: int) -> List[Tuple[Document, float]]:
        """Reciprocal Rank Fusion"""
        doc_scores = defaultdict(float)
        doc_objects = {}
        
        for i, result in enumerate(results):
            weight = self.ensemble_config.weights[i]
            
            for rank, doc in enumerate(result.documents):
                doc_id = self._get_doc_id(doc)
                rrf_score = weight / (self.ensemble_config.rrf_constant + rank + 1)
                
                doc_scores[doc_id] += rrf_score
                doc_objects[doc_id] = doc
        
        # 排序并返回
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [(doc_objects[doc_id], score) for doc_id, score in sorted_docs[:k * 2]]
    
    def _borda_count_fusion(self, results: List[RetrievalResult], k: int) -> List[Tuple[Document, float]]:
        """Borda Count融合"""
        doc_scores = defaultdict(float)
        doc_objects = {}
        
        for i, result in enumerate(results):
            weight = self.ensemble_config.weights[i]
            num_docs = len(result.documents)
            
            for rank, doc in enumerate(result.documents):
                doc_id = self._get_doc_id(doc)
                borda_score = weight * (num_docs - rank)
                
                doc_scores[doc_id] += borda_score
                doc_objects[doc_id] = doc
        
        # 排序并返回
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [(doc_objects[doc_id], score) for doc_id, score in sorted_docs[:k * 2]]
    
    def _post_process_results(self, fused_docs: List[Tuple[Document, float]], k: int) -> List[Tuple[Document, float]]:
        """后处理结果"""
        # 可以在这里添加额外的过滤、重排序等逻辑
        return fused_docs[:k]
    
    def _build_metadata(self, query: str, retrieval_results: List[RetrievalResult], 
                       fused_docs: List[Tuple[Document, float]], 
                       final_docs: List[Tuple[Document, float]]) -> Dict[str, Any]:
        """构建元数据"""
        metadata = {
            "query": query,
            "num_retrievers": len(self.retrievers),
            "retriever_types": [config["type"] for config in self.ensemble_config.retrievers],
            "weights": self.ensemble_config.weights,
            "fusion_method": self.ensemble_config.fusion_method,
            "retriever_results": [
                {
                    "type": self.ensemble_config.retrievers[i]["type"],
                    "documents_count": len(result.documents),
                    "avg_score": np.mean(result.scores) if result.scores else 0.0,
                    "metadata": result.metadata
                }
                for i, result in enumerate(retrieval_results)
            ],
            "fused_results": len(fused_docs),
            "final_results": len(final_docs),
            "score_normalization": self.ensemble_config.enable_score_normalization,
            "min_score_threshold": self.ensemble_config.min_score_threshold
        }
        
        return metadata
    
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
        """获取集成检索器统计信息"""
        stats = super().get_stats()
        
        # 添加子检索器统计
        stats["retrievers"] = [
            {
                "type": self.ensemble_config.retrievers[i]["type"],
                "stats": retriever.get_stats()
            }
            for i, retriever in enumerate(self.retrievers)
        ]
        
        # 添加配置信息
        stats["config"].update({
            "num_retrievers": len(self.retrievers),
            "weights": self.ensemble_config.weights,
            "fusion_method": self.ensemble_config.fusion_method,
            "rrf_constant": self.ensemble_config.rrf_constant,
            "score_normalization": self.ensemble_config.enable_score_normalization,
            "min_score_threshold": self.ensemble_config.min_score_threshold,
            "max_workers": self.ensemble_config.max_workers
        })
        
        return stats
    
    def reset(self) -> None:
        """重置集成检索器"""
        super().reset()
        
        # 重置所有子检索器
        for retriever in self.retrievers:
            retriever.reset()
        
        logger.info("集成检索器已重置")
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
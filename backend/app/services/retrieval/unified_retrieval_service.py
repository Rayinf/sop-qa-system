from typing import List, Dict, Any, Optional, Type, Union
from dataclasses import dataclass
import logging
from enum import Enum

from langchain.schema import Document

from app.services.retrieval.base_retriever import (
    BaseRetriever, RetrievalStrategy, RetrievalResult, RetrievalConfig
)
from app.services.retrieval.vector_retriever import VectorRetriever, VectorRetrievalConfig
from app.services.retrieval.hybrid_retriever import HybridRetriever, HybridRetrievalConfig
from app.services.retrieval.multi_query_retriever import MultiQueryRetriever, MultiQueryRetrievalConfig
from app.services.retrieval.ensemble_retriever import EnsembleRetriever, EnsembleRetrievalConfig
from app.core.config import settings

logger = logging.getLogger(__name__)

class RetrievalMode(Enum):
    """检索模式"""
    VECTOR = "vector"  # 向量检索
    HYBRID = "hybrid"  # 混合检索
    MULTI_QUERY = "multi_query"  # 多查询检索
    ENSEMBLE = "ensemble"  # 集成检索
    AUTO = "auto"  # 自动选择

@dataclass
class UnifiedRetrievalConfig:
    """统一检索配置"""
    # 基础配置
    mode: RetrievalMode = RetrievalMode.AUTO
    k: int = 5
    similarity_threshold: float = 0.0
    category: Optional[str] = None
    enable_cache: bool = True
    
    # 向量检索配置
    vector_config: Optional[Dict[str, Any]] = None
    
    # 混合检索配置
    hybrid_config: Optional[Dict[str, Any]] = None
    
    # 多查询检索配置
    multi_query_config: Optional[Dict[str, Any]] = None
    
    # 集成检索配置
    ensemble_config: Optional[Dict[str, Any]] = None
    
    # 自动选择配置
    auto_selection_rules: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        # 设置默认配置
        if self.vector_config is None:
            self.vector_config = self._get_default_vector_config()
        
        if self.hybrid_config is None:
            self.hybrid_config = self._get_default_hybrid_config()
        
        if self.multi_query_config is None:
            self.multi_query_config = self._get_default_multi_query_config()
        
        if self.ensemble_config is None:
            self.ensemble_config = self._get_default_ensemble_config()
        
        if self.auto_selection_rules is None:
            self.auto_selection_rules = self._get_default_auto_rules()
    
    def _get_default_vector_config(self) -> Dict[str, Any]:
        return {
            "use_mmr": True,
            "mmr_lambda_mult": 0.5,
            "mmr_fetch_k": 20,
            # 类别加权与手动降权的默认值（来自全局设置）
            "category_weight_mode": settings.vector_category_weight_mode,
            "category_primary_boost": settings.vector_category_primary_boost,
            "category_mapped_boost": settings.vector_category_mapped_boost,
            "category_mismatch_penalty": settings.vector_category_mismatch_penalty,
            "manual_downweight_keywords": settings.vector_manual_downweight_keywords,
        }
    
    def _get_default_hybrid_config(self) -> Dict[str, Any]:
        return {
            "dense_weight": 0.7,
            "sparse_weight": 0.3,
            "use_reranking": True,
            "fusion_method": "rrf"
        }
    
    def _get_default_multi_query_config(self) -> Dict[str, Any]:
        return {
            "num_queries": 3,
            "docs_per_query": 8,
            "enable_deduplication": True
        }
    
    def _get_default_ensemble_config(self) -> Dict[str, Any]:
        return {
            "weights": [0.4, 0.3, 0.3],
            "fusion_method": "weighted_sum",
            "enable_score_normalization": True
        }
    
    def _get_default_auto_rules(self) -> Dict[str, Any]:
        return {
            "query_length_threshold": 10,  # 查询长度阈值
            "complex_query_keywords": ["如何", "怎么", "步骤", "流程", "方法"],
            "technical_keywords": ["开发", "代码", "技术", "系统", "算法"],
            "prefer_hybrid_for_short_queries": True,
            "prefer_multi_query_for_complex_queries": True,
            "prefer_ensemble_for_important_queries": True
        }

class UnifiedRetrievalService:
    """统一检索服务：管理所有检索策略并提供统一接口"""
    
    def __init__(self, config: UnifiedRetrievalConfig, documents: Optional[List[Document]] = None):
        self.config = config
        self.documents = documents or []
        
        # 检索器缓存
        self._retrievers: Dict[RetrievalMode, BaseRetriever] = {}
        
        # 统计信息
        self.stats = {
            "total_queries": 0,
            "mode_usage": {mode.value: 0 for mode in RetrievalMode},
            "avg_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(f"统一检索服务初始化成功，默认模式: {config.mode.value}")
    
    def set_documents(self, documents: List[Document]):
        """设置文档"""
        self.documents = documents
        
        # 更新所有已初始化的检索器
        for retriever in self._retrievers.values():
            if hasattr(retriever, 'set_documents'):
                retriever.set_documents(documents)
        
        logger.info(f"已更新文档，共 {len(documents)} 个文档")
    
    def retrieve(self, query: str, mode: Optional[RetrievalMode] = None, **kwargs) -> RetrievalResult:
        """统一检索接口"""
        import time
        start_time = time.time()
        
        try:
            # 从 kwargs 中提取 retrievalMode 参数
            retrieval_mode_str = kwargs.pop('retrievalMode', None)
            if retrieval_mode_str and not mode:
                try:
                    mode = RetrievalMode(retrieval_mode_str)
                except ValueError:
                    logger.warning(f"无效的检索模式: {retrieval_mode_str}，使用默认模式")
            
            # 确定检索模式
            actual_mode = mode or self.config.mode
            if actual_mode == RetrievalMode.AUTO:
                actual_mode = self._auto_select_mode(query)
            
            # 获取检索器
            retriever = self._get_retriever(actual_mode)
            
            # 执行检索
            result = retriever.retrieve(query, **kwargs)
            
            # 更新统计信息
            self._update_stats(actual_mode, time.time() - start_time)
            
            # 添加模式信息到元数据
            if result.metadata is None:
                result.metadata = {}
            result.metadata["retrieval_mode"] = actual_mode.value
            result.metadata["auto_selected"] = mode is None or mode == RetrievalMode.AUTO
            
            logger.debug(f"检索完成: 模式={actual_mode.value}, 结果数={len(result.documents)}, 耗时={time.time() - start_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            raise
    
    def _auto_select_mode(self, query: str) -> RetrievalMode:
        """自动选择检索模式"""
        rules = self.config.auto_selection_rules
        
        # 分析查询特征
        query_lower = query.lower()
        query_length = len(query)
        
        # 检查是否包含复杂查询关键词
        has_complex_keywords = any(
            keyword in query_lower 
            for keyword in rules.get("complex_query_keywords", [])
        )
        
        # 检查是否包含技术关键词
        has_technical_keywords = any(
            keyword in query_lower 
            for keyword in rules.get("technical_keywords", [])
        )
        
        # 根据规则选择模式
        if has_complex_keywords and rules.get("prefer_multi_query_for_complex_queries", True):
            selected_mode = RetrievalMode.MULTI_QUERY
        elif query_length < rules.get("query_length_threshold", 10) and rules.get("prefer_hybrid_for_short_queries", True):
            selected_mode = RetrievalMode.HYBRID
        elif has_technical_keywords:
            selected_mode = RetrievalMode.VECTOR
        elif rules.get("prefer_ensemble_for_important_queries", True):
            selected_mode = RetrievalMode.ENSEMBLE
        else:
            selected_mode = RetrievalMode.VECTOR  # 默认
        
        logger.debug(f"自动选择检索模式: {selected_mode.value} (查询长度: {query_length}, 复杂关键词: {has_complex_keywords}, 技术关键词: {has_technical_keywords})")
        
        return selected_mode
    
    def _get_retriever(self, mode: RetrievalMode) -> BaseRetriever:
        """获取检索器（带缓存）"""
        if mode not in self._retrievers:
            self._retrievers[mode] = self._create_retriever(mode)
        
        return self._retrievers[mode]
    
    def _create_retriever(self, mode: RetrievalMode) -> BaseRetriever:
        """创建检索器"""
        try:
            if mode == RetrievalMode.VECTOR:
                config = VectorRetrievalConfig(
                    k=self.config.k,
                    similarity_threshold=self.config.similarity_threshold,
                    category=self.config.category,
                    enable_cache=self.config.enable_cache,
                    **self.config.vector_config
                )
                return VectorRetriever(config)
            
            elif mode == RetrievalMode.HYBRID:
                vector_config = VectorRetrievalConfig(
                    k=self.config.k,
                    similarity_threshold=self.config.similarity_threshold,
                    category=self.config.category,
                    enable_cache=self.config.enable_cache,
                    **self.config.vector_config
                )
                config = HybridRetrievalConfig(
                    k=self.config.k,
                    similarity_threshold=self.config.similarity_threshold,
                    category=self.config.category,
                    enable_cache=self.config.enable_cache,
                    vector_config=vector_config,
                    **self.config.hybrid_config
                )
                return HybridRetriever(config, self.documents)
            
            elif mode == RetrievalMode.MULTI_QUERY:
                vector_config = VectorRetrievalConfig(
                    k=self.config.multi_query_config.get("docs_per_query", 8),
                    similarity_threshold=self.config.similarity_threshold,
                    category=self.config.category,
                    enable_cache=self.config.enable_cache,
                    **self.config.vector_config
                )
                config = MultiQueryRetrievalConfig(
                    k=self.config.k,
                    similarity_threshold=self.config.similarity_threshold,
                    category=self.config.category,
                    enable_cache=self.config.enable_cache,
                    vector_config=vector_config,
                    **self.config.multi_query_config
                )
                return MultiQueryRetriever(config)
            
            elif mode == RetrievalMode.ENSEMBLE:
                config = EnsembleRetrievalConfig(
                    k=self.config.k,
                    similarity_threshold=self.config.similarity_threshold,
                    category=self.config.category,
                    enable_cache=self.config.enable_cache,
                    **self.config.ensemble_config
                )
                return EnsembleRetriever(config, self.documents)
            
            else:
                raise ValueError(f"不支持的检索模式: {mode}")
                
        except Exception as e:
            logger.error(f"创建检索器失败 ({mode.value}): {e}")
            raise
    
    def _update_stats(self, mode: RetrievalMode, response_time: float):
        """更新统计信息"""
        self.stats["total_queries"] += 1
        self.stats["mode_usage"][mode.value] += 1
        
        # 更新平均响应时间
        total_queries = self.stats["total_queries"]
        current_avg = self.stats["avg_response_time"]
        self.stats["avg_response_time"] = (current_avg * (total_queries - 1) + response_time) / total_queries
    
    def get_available_modes(self) -> List[RetrievalMode]:
        """获取可用的检索模式"""
        return list(RetrievalMode)
    
    def get_mode_description(self, mode: RetrievalMode) -> str:
        """获取检索模式描述"""
        descriptions = {
            RetrievalMode.VECTOR: "基于向量相似度的语义检索，适合语义相关的查询",
            RetrievalMode.HYBRID: "结合向量检索和关键词检索，平衡语义和字面匹配",
            RetrievalMode.MULTI_QUERY: "生成多个相关查询进行检索，提高召回率",
            RetrievalMode.ENSEMBLE: "组合多种检索策略，获得最佳检索效果",
            RetrievalMode.AUTO: "根据查询特征自动选择最适合的检索模式"
        }
        return descriptions.get(mode, "未知模式")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 添加检索器统计
        stats["retrievers"] = {}
        for mode, retriever in self._retrievers.items():
            stats["retrievers"][mode.value] = retriever.get_stats()
        
        # 添加配置信息
        stats["config"] = {
            "default_mode": self.config.mode.value,
            "k": self.config.k,
            "similarity_threshold": self.config.similarity_threshold,
            "category": self.config.category,
            "enable_cache": self.config.enable_cache,
            "documents_count": len(self.documents)
        }
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_queries": 0,
            "mode_usage": {mode.value: 0 for mode in RetrievalMode},
            "avg_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        logger.info("统计信息已重置")
    
    def reset_retrievers(self):
        """重置所有检索器"""
        for retriever in self._retrievers.values():
            retriever.reset()
        
        self._retrievers.clear()
        logger.info("所有检索器已重置")
    
    def reset(self):
        """完全重置服务"""
        self.reset_stats()
        self.reset_retrievers()
        logger.info("统一检索服务已重置")
    
    def warmup(self, sample_queries: Optional[List[str]] = None):
        """预热检索器"""
        if sample_queries is None:
            sample_queries = [
                "测试查询",
                "如何操作",
                "技术文档",
                "质量管理流程"
            ]
        
        logger.info("开始预热检索器...")
        
        for mode in [RetrievalMode.VECTOR, RetrievalMode.HYBRID, RetrievalMode.MULTI_QUERY]:
            try:
                retriever = self._get_retriever(mode)
                for query in sample_queries[:2]:  # 每个模式用2个查询预热
                    retriever.retrieve(query, k=1)
                logger.debug(f"检索器 {mode.value} 预热完成")
            except Exception as e:
                logger.warning(f"检索器 {mode.value} 预热失败: {e}")
        
        logger.info("检索器预热完成")
    
    def __del__(self):
        """清理资源"""
        for retriever in self._retrievers.values():
            if hasattr(retriever, '__del__'):
                retriever.__del__()
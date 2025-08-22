from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timezone

from langchain.schema import Document

logger = logging.getLogger(__name__)

class RetrievalStrategy(Enum):
    """检索策略枚举"""
    VECTOR = "vector"
    HYBRID = "hybrid"
    PARENT = "parent"
    MULTI_QUERY = "multi_query"
    COMPRESSION = "compression"
    ENSEMBLE = "ensemble"
    ROUTING = "routing"

@dataclass
class RetrievalResult:
    """检索结果数据类"""
    documents: List[Document]
    scores: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    strategy_used: Optional[str] = None
    fallback_used: bool = False

@dataclass
class RetrievalConfig:
    """检索配置基类"""
    k: int = 10
    similarity_threshold: float = 0.3
    category: Optional[str] = None
    enable_cache: bool = True
    cache_ttl: int = 3600  # 缓存时间（秒）

class BaseRetriever(ABC):
    """统一检索器基类"""
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.strategy = self._get_strategy()
        self._stats = {
            "total_queries": 0,
            "total_documents_retrieved": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "last_query_time": None
        }
        self._cache = {} if config.enable_cache else None
        
    @abstractmethod
    def _get_strategy(self) -> RetrievalStrategy:
        """获取检索策略类型"""
        pass
    
    @abstractmethod
    def _retrieve_documents(self, query: str, **kwargs) -> RetrievalResult:
        """实际的文档检索逻辑"""
        pass
    
    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        """检索文档的统一入口"""
        start_time = datetime.now(timezone.utc)
        
        try:
            # 更新统计信息
            self._stats["total_queries"] += 1
            self._stats["last_query_time"] = start_time.isoformat()
            
            # 检查缓存
            cache_key = self._get_cache_key(query, **kwargs)
            if self._cache and cache_key in self._cache:
                self._stats["cache_hits"] += 1
                cached_result = self._cache[cache_key]
                logger.info(f"🎯 缓存命中: {self.strategy.value} - '{query[:50]}{'...' if len(query) > 50 else ''}'")
                return cached_result
            
            if self._cache:
                self._stats["cache_misses"] += 1
            
            # 执行检索
            logger.info(f"🔍 开始检索: {self.strategy.value} - '{query[:50]}{'...' if len(query) > 50 else ''}'")
            result = self._retrieve_documents(query, **kwargs)
            
            # 计算处理时间
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.processing_time = processing_time
            result.strategy_used = self.strategy.value
            
            # 更新统计信息
            self._stats["total_documents_retrieved"] += len(result.documents)
            self._update_average_processing_time(processing_time)
            
            # 缓存结果
            if self._cache and cache_key:
                self._cache[cache_key] = result
                # 简单的缓存清理策略
                if len(self._cache) > 1000:
                    # 删除最旧的一半缓存
                    keys_to_delete = list(self._cache.keys())[:500]
                    for key in keys_to_delete:
                        del self._cache[key]
            
            logger.info(f"✅ 检索完成: {self.strategy.value} - 找到 {len(result.documents)} 个文档，耗时 {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ 检索失败: {self.strategy.value} - {e}")
            # 返回空结果而不是抛出异常
            return RetrievalResult(
                documents=[],
                metadata={"error": str(e)},
                processing_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
                strategy_used=self.strategy.value
            )
    
    def _get_cache_key(self, query: str, **kwargs) -> Optional[str]:
        """生成缓存键"""
        if not self.config.enable_cache:
            return None
        
        # 创建基于查询和参数的缓存键
        key_parts = [query, str(self.config.k), str(self.config.similarity_threshold)]
        if self.config.category:
            key_parts.append(self.config.category)
        
        # 添加其他关键参数
        for key, value in sorted(kwargs.items()):
            if key in ['k', 'category', 'filter_dict']:
                key_parts.append(f"{key}:{value}")
        
        return f"{self.strategy.value}:" + ":".join(key_parts)
    
    def _update_average_processing_time(self, processing_time: float):
        """更新平均处理时间"""
        current_avg = self._stats["average_processing_time"]
        total_queries = self._stats["total_queries"]
        
        if total_queries == 1:
            self._stats["average_processing_time"] = processing_time
        else:
            # 计算移动平均
            self._stats["average_processing_time"] = (
                (current_avg * (total_queries - 1) + processing_time) / total_queries
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        stats = self._stats.copy()
        stats.update({
            "strategy": self.strategy.value,
            "config": {
                "k": self.config.k,
                "similarity_threshold": self.config.similarity_threshold,
                "category": self.config.category,
                "enable_cache": self.config.enable_cache
            },
            "cache_size": len(self._cache) if self._cache else 0
        })
        return stats
    
    def reset(self) -> None:
        """重置检索器状态"""
        self._stats = {
            "total_queries": 0,
            "total_documents_retrieved": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "last_query_time": None
        }
        if self._cache:
            self._cache.clear()
        logger.info(f"🔄 检索器已重置: {self.strategy.value}")
    
    def clear_cache(self) -> int:
        """清空缓存"""
        if not self._cache:
            return 0
        
        cache_size = len(self._cache)
        self._cache.clear()
        logger.info(f"🗑️ 缓存已清空: {self.strategy.value} - 清除了 {cache_size} 个缓存项")
        return cache_size
    
    def warm_up(self, queries: List[str]) -> Dict[str, Any]:
        """预热检索器"""
        logger.info(f"🔥 开始预热检索器: {self.strategy.value} - {len(queries)} 个查询")
        
        start_time = datetime.now(timezone.utc)
        results = []
        
        for query in queries:
            try:
                result = self.retrieve(query)
                results.append({
                    "query": query,
                    "documents_count": len(result.documents),
                    "processing_time": result.processing_time,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })
        
        total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        successful_queries = sum(1 for r in results if r["success"])
        
        warmup_stats = {
            "total_queries": len(queries),
            "successful_queries": successful_queries,
            "failed_queries": len(queries) - successful_queries,
            "total_time": total_time,
            "average_time_per_query": total_time / len(queries) if queries else 0,
            "results": results
        }
        
        logger.info(f"✅ 预热完成: {self.strategy.value} - {successful_queries}/{len(queries)} 成功，耗时 {total_time:.3f}s")
        return warmup_stats
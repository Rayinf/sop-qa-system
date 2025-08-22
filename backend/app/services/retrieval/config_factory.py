from typing import Dict, Any, List
import logging

from app.services.retrieval.unified_retrieval_service import (
    UnifiedRetrievalConfig, RetrievalMode
)
from app.core.config import settings

logger = logging.getLogger(__name__)

class RetrievalConfigFactory:
    """检索配置工厂：根据设置创建统一检索配置"""
    
    @staticmethod
    def create_config(category: str = None, **overrides) -> UnifiedRetrievalConfig:
        """创建统一检索配置
        
        Args:
            category: 文档类别
            **overrides: 覆盖的配置参数
        
        Returns:
            UnifiedRetrievalConfig: 统一检索配置
        """
        try:
            # 解析检索模式
            mode_str = overrides.get('mode', settings.retrieval_mode)
            mode = RetrievalConfigFactory._parse_mode(mode_str)
            
            # 基础配置
            base_config = {
                'mode': mode,
                'k': overrides.get('k', settings.retrieval_k),
                'similarity_threshold': overrides.get('similarity_threshold', settings.retrieval_similarity_threshold),
                'category': category,
                'enable_cache': overrides.get('enable_cache', settings.retrieval_enable_cache)
            }
            
            # 向量检索配置
            vector_config = RetrievalConfigFactory._create_vector_config(overrides)
            
            # 混合检索配置
            hybrid_config = RetrievalConfigFactory._create_hybrid_config(overrides)
            
            # 多查询检索配置
            multi_query_config = RetrievalConfigFactory._create_multi_query_config(overrides)
            
            # 集成检索配置
            ensemble_config = RetrievalConfigFactory._create_ensemble_config(overrides)
            
            # 自动选择配置
            auto_selection_rules = RetrievalConfigFactory._create_auto_selection_rules(overrides)
            
            config = UnifiedRetrievalConfig(
                **base_config,
                vector_config=vector_config,
                hybrid_config=hybrid_config,
                multi_query_config=multi_query_config,
                ensemble_config=ensemble_config,
                auto_selection_rules=auto_selection_rules
            )
            
            logger.debug(f"创建检索配置成功: mode={mode.value}, category={category}")
            return config
            
        except Exception as e:
            logger.error(f"创建检索配置失败: {e}")
            raise
    
    @staticmethod
    def _parse_mode(mode_str: str) -> RetrievalMode:
        """解析检索模式"""
        mode_mapping = {
            'vector': RetrievalMode.VECTOR,
            'hybrid': RetrievalMode.HYBRID,
            'multi_query': RetrievalMode.MULTI_QUERY,
            'ensemble': RetrievalMode.ENSEMBLE,
            'auto': RetrievalMode.AUTO
        }
        
        mode = mode_mapping.get(mode_str.lower())
        if mode is None:
            logger.warning(f"未知的检索模式: {mode_str}，使用默认模式 AUTO")
            return RetrievalMode.AUTO
        
        return mode
    
    @staticmethod
    def _create_vector_config(overrides: Dict[str, Any]) -> Dict[str, Any]:
        """创建向量检索配置"""
        return {
            'use_mmr': overrides.get('vector_use_mmr', settings.vector_use_mmr),
            'mmr_fetch_k': overrides.get('vector_mmr_fetch_k', settings.vector_mmr_fetch_k),
            'mmr_lambda_mult': overrides.get('vector_mmr_lambda_mult', settings.vector_mmr_lambda_mult),
            # 新增：类别加权与手动降权
            'category_weight_mode': overrides.get('vector_category_weight_mode', settings.vector_category_weight_mode),
            'category_primary_boost': overrides.get('vector_category_primary_boost', settings.vector_category_primary_boost),
            'category_mapped_boost': overrides.get('vector_category_mapped_boost', settings.vector_category_mapped_boost),
            'category_mismatch_penalty': overrides.get('vector_category_mismatch_penalty', settings.vector_category_mismatch_penalty),
            'manual_downweight_keywords': overrides.get('vector_manual_downweight_keywords', settings.vector_manual_downweight_keywords),
        }
    
    @staticmethod
    def _create_hybrid_config(overrides: Dict[str, Any]) -> Dict[str, Any]:
        """创建混合检索配置"""
        return {
            'dense_weight': overrides.get('hybrid_dense_weight', settings.hybrid_dense_weight),
            'sparse_weight': overrides.get('hybrid_sparse_weight', settings.hybrid_sparse_weight),
            'use_reranking': overrides.get('hybrid_use_reranking', settings.hybrid_use_reranking),
            'fusion_method': overrides.get('hybrid_fusion_method', settings.hybrid_fusion_method),
            'rrf_constant': overrides.get('hybrid_rrf_constant', settings.hybrid_rrf_constant)
        }
    
    @staticmethod
    def _create_multi_query_config(overrides: Dict[str, Any]) -> Dict[str, Any]:
        """创建多查询检索配置"""
        return {
            'num_queries': overrides.get('multi_query_num_queries', settings.multi_query_num_queries),
            'docs_per_query': overrides.get('multi_query_docs_per_query', settings.multi_query_docs_per_query),
            'enable_deduplication': overrides.get('multi_query_enable_deduplication', settings.multi_query_enable_deduplication),
            'max_workers': overrides.get('multi_query_max_workers', settings.multi_query_max_workers)
        }
    
    @staticmethod
    def _create_ensemble_config(overrides: Dict[str, Any]) -> Dict[str, Any]:
        """创建集成检索配置"""
        # 处理权重配置
        weights = overrides.get('ensemble_weights', settings.ensemble_weights)
        if isinstance(weights, str):
            try:
                # 尝试解析字符串格式的权重
                weights = [float(w.strip()) for w in weights.split(',')]
            except ValueError:
                logger.warning(f"无法解析权重字符串: {weights}，使用默认权重")
                weights = [0.4, 0.3, 0.3]
        
        return {
            'weights': weights,
            'fusion_method': overrides.get('ensemble_fusion_method', settings.ensemble_fusion_method),
            'enable_score_normalization': overrides.get('ensemble_enable_score_normalization', settings.ensemble_enable_score_normalization),
            'min_score_threshold': overrides.get('ensemble_min_score_threshold', settings.ensemble_min_score_threshold),
            'max_workers': overrides.get('ensemble_max_workers', settings.ensemble_max_workers)
        }
    
    @staticmethod
    def _create_auto_selection_rules(overrides: Dict[str, Any]) -> Dict[str, Any]:
        """创建自动选择规则"""
        return {
            'query_length_threshold': overrides.get('auto_query_length_threshold', settings.auto_query_length_threshold),
            'complex_query_keywords': overrides.get('auto_complex_query_keywords', [
                "如何", "怎么", "步骤", "流程", "方法", "操作", "程序", "过程",
                "指导", "指南", "教程", "说明", "详细", "具体", "完整"
            ]),
            'technical_keywords': overrides.get('auto_technical_keywords', [
                "开发", "代码", "技术", "系统", "算法", "编程", "软件", "架构",
                "API", "接口", "数据库", "服务器", "网络", "安全", "性能"
            ]),
            'prefer_hybrid_for_short_queries': overrides.get('auto_prefer_hybrid_for_short', settings.auto_prefer_hybrid_for_short),
            'prefer_multi_query_for_complex_queries': overrides.get('auto_prefer_multi_query_for_complex', settings.auto_prefer_multi_query_for_complex),
            'prefer_ensemble_for_important_queries': overrides.get('auto_prefer_ensemble_for_important', settings.auto_prefer_ensemble_for_important)
        }
    
    @staticmethod
    def create_category_specific_config(category: str) -> UnifiedRetrievalConfig:
        """为特定类别创建优化的检索配置"""
        category_configs = {
            'development': {
                'mode': 'vector',
                'vector_use_mmr': True,
                'vector_mmr_lambda_mult': 0.7,  # 更注重相关性
                'k': 8
            },
            'manual': {
                'mode': 'hybrid',
                'hybrid_dense_weight': 0.6,
                'hybrid_sparse_weight': 0.4,  # 更注重关键词匹配
                'k': 6
            },
            'procedure': {
                'mode': 'multi_query',
                'multi_query_num_queries': 4,  # 生成更多查询
                'multi_query_docs_per_query': 6,
                'k': 5
            },
            'policy': {
                'mode': 'ensemble',
                'ensemble_weights': [0.3, 0.4, 0.3],  # 平衡各种策略
                'k': 7
            }
        }
        
        overrides = category_configs.get(category, {})
        return RetrievalConfigFactory.create_config(category=category, **overrides)
    
    @staticmethod
    def create_performance_optimized_config(performance_level: str = 'balanced') -> UnifiedRetrievalConfig:
        """创建性能优化的配置
        
        Args:
            performance_level: 性能级别 ('fast', 'balanced', 'quality')
        """
        performance_configs = {
            'fast': {
                'mode': 'vector',
                'k': 3,
                'vector_use_mmr': False,  # 关闭MMR以提高速度
                'enable_cache': True
            },
            'balanced': {
                'mode': 'hybrid',
                'k': 5,
                'hybrid_use_reranking': False,  # 关闭重排序以平衡速度和质量
                'enable_cache': True
            },
            'quality': {
                'mode': 'ensemble',
                'k': 8,
                'ensemble_weights': [0.3, 0.3, 0.4],  # 更注重多查询
                'multi_query_num_queries': 4,
                'hybrid_use_reranking': True,
                'enable_cache': True
            }
        }
        
        overrides = performance_configs.get(performance_level, performance_configs['balanced'])
        return RetrievalConfigFactory.create_config(**overrides)
    
    @staticmethod
    def get_config_summary(config: UnifiedRetrievalConfig) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            'mode': config.mode.value,
            'k': config.k,
            'similarity_threshold': config.similarity_threshold,
            'category': config.category,
            'enable_cache': config.enable_cache,
            'vector_config': {
                'use_mmr': config.vector_config.get('use_mmr'),
                'mmr_lambda_mult': config.vector_config.get('mmr_lambda_mult')
            },
            'hybrid_config': {
                'dense_weight': config.hybrid_config.get('dense_weight'),
                'sparse_weight': config.hybrid_config.get('sparse_weight'),
                'use_reranking': config.hybrid_config.get('use_reranking')
            },
            'multi_query_config': {
                'num_queries': config.multi_query_config.get('num_queries'),
                'docs_per_query': config.multi_query_config.get('docs_per_query')
            },
            'ensemble_config': {
                'weights': config.ensemble_config.get('weights'),
                'fusion_method': config.ensemble_config.get('fusion_method')
            }
        }
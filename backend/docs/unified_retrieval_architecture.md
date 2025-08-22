# 统一检索架构设计

## 当前问题分析

### 现有检索策略混乱的问题

1. **多个重叠的检索服务**
   - `VectorService`: 基础向量搜索
   - `AdvancedRetrieverService`: 高级检索策略（Parent、MultiQuery、Compression、Ensemble）
   - `MultiRetrievalService`: 多知识库路由
   - `HybridRetrievalService`: 混合检索（向量+BM25）

2. **配置参数分散且冲突**
   - 检索器类型配置：`retriever_type`
   - 集成权重配置：`ensemble_weights`
   - MMR配置：`use_mmr`, `mmr_fetch_k`, `mmr_lambda_mult`
   - 高级检索器配置：`use_advanced_retriever`
   - 多知识库配置：`use_multi_retrieval`

3. **检索策略选择逻辑复杂**
   - QAService中有多个条件判断
   - 不同服务之间缺乏统一的接口
   - 检索器实例化和缓存机制不一致

## 统一检索架构设计

### 核心设计原则

1. **单一职责原则**：每个检索器只负责一种特定的检索策略
2. **统一接口**：所有检索器实现相同的接口
3. **配置驱动**：通过配置文件统一管理所有检索策略
4. **分层架构**：基础检索 → 高级检索 → 路由检索
5. **可扩展性**：易于添加新的检索策略

### 架构层次

```
┌─────────────────────────────────────────────────────────────┐
│                    检索路由层 (Routing Layer)                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           MultiRetrievalService                         │ │
│  │  - 问题分类和知识库路由                                    │ │
│  │  - 降级策略处理                                          │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   检索策略层 (Strategy Layer)                │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │   VectorSearch  │ │  HybridSearch   │ │  AdvancedSearch │ │
│  │  - 基础向量检索  │ │  - 向量+BM25    │ │  - 复合检索策略  │ │
│  │  - MMR检索      │ │  - 重排序       │ │  - Parent/Child │ │
│  │  - 相似度过滤    │ │  - 融合算法     │ │  - MultiQuery   │ │
│  └─────────────────┘ └─────────────────┘ │  - Compression  │ │
│                                          │  - Ensemble     │ │
│                                          └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    基础服务层 (Base Layer)                   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │  VectorStore    │ │   Embeddings    │ │   TextSplitter  │ │
│  │  - FAISS索引    │ │  - 向量化模型    │ │  - 文档分块     │ │
│  │  - 向量存储     │ │  - 相似度计算    │ │  - 重叠处理     │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 统一检索接口

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain.schema import Document

class BaseRetriever(ABC):
    """统一检索器基类"""
    
    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> List[Document]:
        """检索文档"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """重置检索器状态"""
        pass
```

### 检索策略分类

#### 1. 基础检索策略 (Basic Retrieval)

**VectorRetriever**
- **职责**: 基础向量相似度搜索
- **适用场景**: 简单的语义搜索
- **配置参数**: `k`, `similarity_threshold`, `use_mmr`

#### 2. 混合检索策略 (Hybrid Retrieval)

**HybridRetriever**
- **职责**: 结合向量搜索和BM25关键词搜索
- **适用场景**: 需要兼顾语义和关键词匹配
- **配置参数**: `dense_weight`, `sparse_weight`, `use_reranking`

#### 3. 高级检索策略 (Advanced Retrieval)

**ParentDocumentRetriever**
- **职责**: 父子文档检索，提供更完整的上下文
- **适用场景**: 需要完整文档上下文的问答
- **配置参数**: `parent_chunk_ratio`, `child_chunk_ratio`

**MultiQueryRetriever**
- **职责**: 生成多个查询角度，提高召回率
- **适用场景**: 复杂问题需要多角度搜索
- **配置参数**: `multi_query_count`

**CompressionRetriever**
- **职责**: 压缩和过滤检索结果，提高精确度
- **适用场景**: 需要高精度答案的场景
- **配置参数**: `compression_enabled`

**EnsembleRetriever**
- **职责**: 组合多种检索策略
- **适用场景**: 需要平衡多种检索优势
- **配置参数**: `ensemble_weights`

#### 4. 路由检索策略 (Routing Retrieval)

**MultiRetrievalService**
- **职责**: 根据问题类型路由到不同知识库
- **适用场景**: 多领域知识库系统
- **配置参数**: 类别映射、降级策略

### 配置统一化

#### 新的配置结构

```python
class RetrievalConfig:
    # 检索策略选择
    primary_strategy: str = "hybrid"  # vector, hybrid, advanced, routing
    fallback_strategy: str = "vector"  # 降级策略
    
    # 基础检索配置
    vector_config: VectorConfig = VectorConfig()
    
    # 混合检索配置
    hybrid_config: HybridConfig = HybridConfig()
    
    # 高级检索配置
    advanced_config: AdvancedConfig = AdvancedConfig()
    
    # 路由检索配置
    routing_config: RoutingConfig = RoutingConfig()

class VectorConfig:
    k: int = 10
    similarity_threshold: float = 0.3
    use_mmr: bool = True
    mmr_fetch_k: int = 100
    mmr_lambda_mult: float = 0.5

class HybridConfig:
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    use_reranking: bool = True
    reranking_model: str = "bge-reranker-base"

class AdvancedConfig:
    strategy: str = "ensemble"  # parent, multi_query, compression, ensemble
    ensemble_weights: List[float] = [0.4, 0.3, 0.3]
    parent_chunk_ratio: float = 1.5
    child_chunk_ratio: float = 0.8
    multi_query_count: int = 3
    compression_enabled: bool = True

class RoutingConfig:
    enabled: bool = True
    classification_model: str = "text-classification"
    confidence_threshold: float = 0.7
    fallback_retriever: str = "manual"
```

### 实现计划

#### 阶段1: 重构基础架构
1. 创建统一的检索器基类
2. 重构VectorService为VectorRetriever
3. 统一配置管理

#### 阶段2: 整合检索策略
1. 重构AdvancedRetrieverService
2. 整合HybridRetrievalService
3. 创建检索器工厂

#### 阶段3: 优化路由层
1. 简化MultiRetrievalService
2. 实现统一的检索管理器
3. 优化缓存机制

#### 阶段4: 测试和优化
1. 性能测试
2. 准确性评估
3. 参数调优

### 预期收益

1. **代码简化**: 减少重复代码，提高可维护性
2. **配置统一**: 集中管理所有检索相关配置
3. **性能优化**: 统一缓存和实例管理
4. **扩展性**: 易于添加新的检索策略
5. **可测试性**: 每个组件职责明确，便于单元测试

### 迁移策略

1. **向后兼容**: 保持现有API不变
2. **渐进式迁移**: 逐步替换现有实现
3. **配置映射**: 自动映射旧配置到新配置
4. **性能监控**: 监控迁移过程中的性能变化
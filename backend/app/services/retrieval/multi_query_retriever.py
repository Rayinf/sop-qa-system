from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM

from app.services.retrieval.base_retriever import (
    BaseRetriever, RetrievalStrategy, RetrievalResult, RetrievalConfig
)
from app.services.retrieval.vector_retriever import VectorRetriever, VectorRetrievalConfig
from app.services.llm_service import LLMService
from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class MultiQueryRetrievalConfig(RetrievalConfig):
    """多查询检索配置"""
    # 向量检索配置
    vector_config: VectorRetrievalConfig = None
    
    # 查询生成配置
    num_queries: int = 3  # 生成查询数量
    query_generation_prompt: str = None  # 自定义查询生成提示
    
    # 检索配置
    docs_per_query: int = 10  # 每个查询检索的文档数
    
    # 去重配置
    enable_deduplication: bool = True
    similarity_threshold_for_dedup: float = 0.95  # 去重相似度阈值
    
    # 并发配置
    max_workers: int = 3  # 最大并发数
    
    def __post_init__(self):
        if self.vector_config is None:
            self.vector_config = VectorRetrievalConfig(
                k=self.docs_per_query,
                similarity_threshold=self.similarity_threshold,
                category=self.category,
                enable_cache=self.enable_cache
            )
        
        if self.query_generation_prompt is None:
            self.query_generation_prompt = self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """获取默认的查询生成提示"""
        return """
你是一个专业的查询扩展助手。给定一个用户查询，你需要生成{num_queries}个相关但不同角度的查询，以帮助更全面地检索相关文档。

要求：
1. 生成的查询应该与原查询语义相关但表达方式不同
2. 每个查询应该从不同角度或层面来理解原查询
3. 保持查询的专业性和准确性
4. 每行一个查询，不要添加编号或其他格式
5. 查询应该简洁明了，避免过于复杂的表述

原查询: {question}

生成的相关查询：
"""

class MultiQueryRetriever(BaseRetriever):
    """多查询检索器：通过生成多个相关查询来提高检索效果"""
    
    def __init__(self, config: MultiQueryRetrievalConfig):
        super().__init__(config)
        self.multi_query_config = config
        
        # 初始化向量检索器
        self.vector_retriever = VectorRetriever(config.vector_config)
        
        # 初始化LLM服务
        self.llm_service = LLMService()
        
        # 查询生成提示模板
        self.query_prompt = PromptTemplate(
            template=config.query_generation_prompt,
            input_variables=["question", "num_queries"]
        )
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        logger.info(f"多查询检索器初始化成功: num_queries={config.num_queries}, docs_per_query={config.docs_per_query}")
    
    def _get_strategy(self) -> RetrievalStrategy:
        return RetrievalStrategy.MULTI_QUERY
    
    def _retrieve_documents(self, query: str, **kwargs) -> RetrievalResult:
        """执行多查询检索"""
        k = kwargs.get('k', self.multi_query_config.k)
        
        try:
            # 1. 生成多个查询
            logger.debug("生成多个相关查询...")
            queries = self._generate_queries(query)
            
            # 2. 并行检索
            logger.debug(f"并行检索 {len(queries)} 个查询...")
            all_results = self._parallel_retrieve(queries)
            
            # 3. 合并和去重
            logger.debug("合并和去重检索结果...")
            merged_docs = self._merge_and_deduplicate(all_results)
            
            # 4. 重新排序和截取
            logger.debug("重新排序检索结果...")
            final_docs = self._rerank_merged_results(query, merged_docs, k)
            
            # 构建元数据
            metadata = {
                "original_query": query,
                "generated_queries": queries,
                "num_queries": len(queries),
                "docs_per_query": self.multi_query_config.docs_per_query,
                "total_retrieved": sum(len(result.documents) for result in all_results),
                "after_deduplication": len(merged_docs),
                "final_results": len(final_docs),
                "deduplication_enabled": self.multi_query_config.enable_deduplication
            }
            
            documents = [doc for doc, score in final_docs]
            scores = [score for doc, score in final_docs]
            
            return RetrievalResult(
                documents=documents,
                scores=scores,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"多查询检索失败: {e}")
            raise
    
    def _generate_queries(self, original_query: str) -> List[str]:
        """生成多个相关查询"""
        try:
            # 构建提示
            prompt = self.query_prompt.format(
                question=original_query,
                num_queries=self.multi_query_config.num_queries
            )
            
            # 调用LLM生成查询
            response = self.llm_service.generate_response(prompt)
            
            # 解析生成的查询
            queries = self._parse_generated_queries(response, original_query)
            
            logger.debug(f"成功生成 {len(queries)} 个查询: {queries}")
            return queries
            
        except Exception as e:
            logger.error(f"查询生成失败: {e}，使用原查询")
            return [original_query]
    
    def _parse_generated_queries(self, response: str, original_query: str) -> List[str]:
        """解析LLM生成的查询"""
        queries = [original_query]  # 始终包含原查询
        
        if not response:
            return queries
        
        # 按行分割并清理
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 跳过空行、编号、标题等
            if not line or len(line) < 5:
                continue
            
            # 移除可能的编号前缀
            line = self._clean_query_line(line)
            
            # 检查是否与原查询或已有查询重复
            if line and line != original_query and line not in queries:
                if not self._is_query_similar_to_existing(line, queries):
                    queries.append(line)
        
        # 确保查询数量不超过配置
        if len(queries) > self.multi_query_config.num_queries + 1:  # +1 for original query
            queries = queries[:self.multi_query_config.num_queries + 1]
        
        return queries
    
    def _clean_query_line(self, line: str) -> str:
        """清理查询行"""
        import re
        
        # 移除编号前缀 (1. 2. 3. 或 1) 2) 3) 等)
        line = re.sub(r'^\d+[.)、]\s*', '', line)
        
        # 移除其他可能的前缀
        prefixes = ['查询:', '问题:', 'Q:', 'Query:', '-', '•', '*']
        for prefix in prefixes:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        
        # 移除引号
        line = line.strip('"\'')
        
        return line.strip()
    
    def _is_query_similar_to_existing(self, new_query: str, existing_queries: List[str]) -> bool:
        """检查新查询是否与现有查询过于相似"""
        for existing in existing_queries:
            # 简单的相似度检查：计算共同词汇比例
            new_words = set(new_query.lower().split())
            existing_words = set(existing.lower().split())
            
            if len(new_words) == 0 or len(existing_words) == 0:
                continue
            
            intersection = len(new_words & existing_words)
            union = len(new_words | existing_words)
            
            similarity = intersection / union if union > 0 else 0
            
            if similarity > 0.8:  # 80%相似度阈值
                return True
        
        return False
    
    def _parallel_retrieve(self, queries: List[str]) -> List[RetrievalResult]:
        """并行检索多个查询"""
        results = []
        
        # 使用线程池并行检索
        future_to_query = {
            self.executor.submit(self._single_retrieve, query): query 
            for query in queries
        }
        
        for future in as_completed(future_to_query):
            query = future_to_query[future]
            try:
                result = future.result()
                results.append(result)
                logger.debug(f"查询 '{query}' 检索完成，获得 {len(result.documents)} 个文档")
            except Exception as e:
                logger.error(f"查询 '{query}' 检索失败: {e}")
                # 添加空结果以保持一致性
                results.append(RetrievalResult(documents=[], scores=[], metadata={}))
        
        return results
    
    def _single_retrieve(self, query: str) -> RetrievalResult:
        """单个查询检索"""
        return self.vector_retriever.retrieve(
            query, 
            k=self.multi_query_config.docs_per_query
        )
    
    def _merge_and_deduplicate(self, results: List[RetrievalResult]) -> List[tuple[Document, float]]:
        """合并和去重检索结果"""
        all_docs = []
        seen_docs = set()
        
        for result in results:
            for i, doc in enumerate(result.documents):
                score = result.scores[i] if result.scores and i < len(result.scores) else 0.0
                
                if self.multi_query_config.enable_deduplication:
                    doc_id = self._get_doc_id(doc)
                    if doc_id in seen_docs:
                        continue
                    
                    # 检查内容相似度
                    if self._is_duplicate_content(doc, all_docs):
                        continue
                    
                    seen_docs.add(doc_id)
                
                all_docs.append((doc, score))
        
        return all_docs
    
    def _is_duplicate_content(self, new_doc: Document, existing_docs: List[tuple[Document, float]]) -> bool:
        """检查文档内容是否重复"""
        if not self.multi_query_config.enable_deduplication:
            return False
        
        new_content = new_doc.page_content.lower().strip()
        
        for existing_doc, _ in existing_docs:
            existing_content = existing_doc.page_content.lower().strip()
            
            # 计算内容相似度
            similarity = self._calculate_content_similarity(new_content, existing_content)
            
            if similarity > self.multi_query_config.similarity_threshold_for_dedup:
                return True
        
        return False
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """计算内容相似度"""
        if content1 == content2:
            return 1.0
        
        # 简单的Jaccard相似度
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _rerank_merged_results(self, original_query: str, merged_docs: List[tuple[Document, float]], k: int) -> List[tuple[Document, float]]:
        """重新排序合并后的结果"""
        if not merged_docs:
            return []
        
        # 按分数排序
        sorted_docs = sorted(merged_docs, key=lambda x: x[1], reverse=True)
        
        # 可以在这里添加更复杂的重排序逻辑
        # 例如：基于与原查询的相关性重新计算分数
        
        return sorted_docs[:k]
    
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
        """获取多查询检索器统计信息"""
        stats = super().get_stats()
        
        # 添加向量检索器统计
        stats["vector_retriever"] = self.vector_retriever.get_stats()
        
        # 添加配置信息
        stats["config"].update({
            "num_queries": self.multi_query_config.num_queries,
            "docs_per_query": self.multi_query_config.docs_per_query,
            "deduplication_enabled": self.multi_query_config.enable_deduplication,
            "similarity_threshold_for_dedup": self.multi_query_config.similarity_threshold_for_dedup,
            "max_workers": self.multi_query_config.max_workers
        })
        
        return stats
    
    def reset(self) -> None:
        """重置多查询检索器"""
        super().reset()
        self.vector_retriever.reset()
        logger.info("多查询检索器已重置")
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
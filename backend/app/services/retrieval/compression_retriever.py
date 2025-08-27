from typing import List, Optional, Dict, Any
import logging

from langchain.schema import Document

from app.services.retrieval.base_retriever import (
    BaseRetriever, RetrievalStrategy, RetrievalResult, RetrievalConfig
)
from app.services.advanced_retriever import AdvancedRetrieverService

logger = logging.getLogger(__name__)


class CompressionRetriever(BaseRetriever):
    """上下文压缩检索封装器
    
    该检索器遵循统一的 BaseRetriever 接口，内部委托 AdvancedRetrieverService
    使用 LangChain 的 ContextualCompressionRetriever 完成压缩检索。
    """

    def __init__(self, config: RetrievalConfig):
        super().__init__(config)
        self._advanced = AdvancedRetrieverService()

    def _get_strategy(self) -> RetrievalStrategy:
        return RetrievalStrategy.COMPRESSION

    def _retrieve_documents(self, query: str, **kwargs) -> RetrievalResult:
        """执行上下文压缩检索"""
        try:
            docs = self._advanced.search_with_advanced_retriever(
                query=query,
                retriever_type="compression",
                k=self.config.k
            )
            metadata: Dict[str, Any] = {
                "note": "contextual compression retriever via AdvancedRetrieverService"
            }
            return RetrievalResult(documents=docs, metadata=metadata)
        except Exception as e:
            logger.error(f"CompressionRetriever: 检索失败 - {e}")
            return RetrievalResult(documents=[], metadata={"error": str(e)})

    def __del__(self):
        try:
            if hasattr(self._advanced, "reset_retrievers"):
                self._advanced.reset_retrievers()
        except Exception:
            pass
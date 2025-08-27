from typing import List, Optional, Dict, Any
import logging

from langchain.schema import Document

from app.services.retrieval.base_retriever import (
    BaseRetriever, RetrievalStrategy, RetrievalResult, RetrievalConfig
)
from app.services.advanced_retriever import AdvancedRetrieverService

logger = logging.getLogger(__name__)


class ParentRetriever(BaseRetriever):
    """父文档检索封装器
    
    该检索器遵循统一的 BaseRetriever 接口，内部委托 AdvancedRetrieverService
    使用 LangChain 的 ParentDocumentRetriever 完成父子文档检索。
    """

    def __init__(self, config: RetrievalConfig):
        super().__init__(config)
        self._advanced = AdvancedRetrieverService()
        self._has_initialized_docs = False

    def _get_strategy(self) -> RetrievalStrategy:
        return RetrievalStrategy.PARENT

    def set_documents(self, documents: List[Document]) -> None:
        """注入文档到父文档检索器
        
        注意：AdvancedRetrieverService 内部会创建 ParentDocumentRetriever 并将文档加入其 docstore。
        """
        try:
            if documents:
                self._advanced.add_documents_to_parent_retriever(documents)
                self._has_initialized_docs = True
                logger.info(f"ParentRetriever: 已注入 {len(documents)} 个文档到父文档检索器")
        except Exception as e:
            logger.error(f"ParentRetriever: 注入文档失败 - {e}")
            # 不抛出，让检索时返回空结果由上层处理

    def _retrieve_documents(self, query: str, **kwargs) -> RetrievalResult:
        """执行父文档检索"""
        try:
            # 使用高级检索服务执行父文档检索
            docs = self._advanced.search_with_advanced_retriever(
                query=query,
                retriever_type="parent",
                k=self.config.k
            )
            metadata: Dict[str, Any] = {
                "note": "parent document retriever via AdvancedRetrieverService"
            }
            return RetrievalResult(documents=docs, metadata=metadata)
        except Exception as e:
            logger.error(f"ParentRetriever: 检索失败 - {e}")
            return RetrievalResult(documents=[], metadata={"error": str(e)})

    def __del__(self):
        try:
            # 释放 AdvancedRetrieverService 内部资源（若有）
            if hasattr(self._advanced, "reset_retrievers"):
                self._advanced.reset_retrievers()
        except Exception:
            pass
from typing import List, Optional, Dict, Any, Tuple
import logging
from datetime import datetime

from langchain.retrievers import (
    ParentDocumentRetriever,
    MultiQueryRetriever,
    ContextualCompressionRetriever,
    EnsembleRetriever
)
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    LLMChainFilter,
    EmbeddingsFilter
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain.schema import Document, BaseRetriever
# from langchain.chat_models import ChatOpenAI  # removed
from langchain.prompts import PromptTemplate

from app.core.config import settings
from app.services.vector_service import VectorService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

class AdvancedRetrieverService:
    """高级检索器服务类，整合多种检索策略"""
    
    def __init__(self, llm: Optional[object] = None):
        self.vector_service = VectorService.get_instance()
        # 使用与向量库一致的嵌入模型，确保检索空间一致
        self.embeddings = self.vector_service._langchain_embeddings
        
        # LLM用于查询生成和压缩
        if llm is not None:
            self.llm = llm
            self.llm_service = None
        else:
            self.llm_service = LLMService()
            self.llm = self.llm_service.get_llm()
        
        # 存储服务
        self.docstore = InMemoryStore()
        
        # 文本分割器 - 使用配置文件中的比例
        parent_ratio = getattr(settings, "parent_chunk_ratio", 1.5)
        child_ratio = getattr(settings, "child_chunk_ratio", 0.8)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(settings.chunk_size * parent_ratio),  # 父文档大小
            chunk_overlap=settings.chunk_overlap
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(settings.chunk_size * child_ratio),  # 子文档大小
            chunk_overlap=int(settings.chunk_overlap * child_ratio)
        )
        
        # 缓存检索器实例
        self._parent_retriever = None
        self._multi_query_retriever = None
        self._compression_retriever = None
        self._ensemble_retriever = None
    
    def _create_default_llm(self):
        """创建默认LLM实例（通过LLMService）"""
        service = LLMService()
        return service.get_llm()
    
    def create_parent_document_retriever(self) -> ParentDocumentRetriever:
        """创建父文档检索器"""
        if self._parent_retriever is not None:
            return self._parent_retriever
        
        try:
            # 创建子文档向量存储
            child_vectorstore = FAISS.from_texts(
                texts=["初始化文本"],
                embedding=self.embeddings
            )
            
            # 创建父文档检索器
            self._parent_retriever = ParentDocumentRetriever(
                vectorstore=child_vectorstore,
                docstore=self.docstore,
                child_splitter=self.child_splitter,
                parent_splitter=self.parent_splitter,
                search_kwargs={"k": settings.retrieval_k // 2}
            )
            
            logger.info("父文档检索器创建成功")
            return self._parent_retriever
            
        except Exception as e:
            logger.error(f"创建父文档检索器失败: {e}")
            raise
    
    def create_multi_query_retriever(self, base_retriever: BaseRetriever) -> MultiQueryRetriever:
        """创建多查询检索器"""
        if self._multi_query_retriever is not None:
            return self._multi_query_retriever
        
        try:
            # 自定义查询生成提示 - 根据配置动态生成
            query_count = settings.multi_query_num_queries
            query_lines = "\n".join([f"查询{i+1}:" for i in range(query_count)])
            
            query_prompt = PromptTemplate(
                input_variables=["question"],
                template=f"""你是一个AI助手，需要为给定的问题生成多个不同的搜索查询。
                
原始问题: {{question}}

请生成{query_count}个不同角度的搜索查询，这些查询应该能够帮助找到回答原始问题所需的信息。
每个查询应该从不同的角度或使用不同的关键词来表达相同的信息需求。

{query_lines}"""
            )
            
            self._multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=self.llm,
                prompt=query_prompt
            )
            
            logger.info("多查询检索器创建成功")
            return self._multi_query_retriever
            
        except Exception as e:
            logger.error(f"创建多查询检索器失败: {e}")
            raise
    
    def create_contextual_compression_retriever(self, base_retriever: BaseRetriever) -> ContextualCompressionRetriever:
        """创建上下文压缩检索器"""
        if self._compression_retriever is not None:
            return self._compression_retriever
        
        try:
            # 创建LLM链提取器
            compressor = LLMChainExtractor.from_llm(
                llm=self.llm,
                prompt=PromptTemplate(
                    input_variables=["question", "context"],
                    template="""给定以下上下文和问题，请提取与问题最相关的信息。
                    只返回直接相关的内容，去除无关信息。
                    
                    问题: {question}
                    
                    上下文: {context}
                    
                    相关信息:"""
                )
            )
            
            self._compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            
            logger.info("上下文压缩检索器创建成功")
            return self._compression_retriever
            
        except Exception as e:
            logger.error(f"创建上下文压缩检索器失败: {e}")
            raise
    
    def create_ensemble_retriever(self) -> EnsembleRetriever:
        """创建集成检索器，组合多种检索策略"""
        if self._ensemble_retriever is not None:
            return self._ensemble_retriever
        
        try:
            # 获取基础检索器
            base_retriever = self.vector_service.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": settings.retrieval_k}
            )
            
            # 创建各种检索器
            parent_retriever = self.create_parent_document_retriever()
            multi_query_retriever = self.create_multi_query_retriever(base_retriever)
            compression_retriever = self.create_contextual_compression_retriever(base_retriever)
            
            # 创建集成检索器
            self._ensemble_retriever = EnsembleRetriever(
                retrievers=[
                    base_retriever,
                    parent_retriever,
                    multi_query_retriever,
                    compression_retriever
                ],
                weights=settings.ensemble_weights  # 使用配置文件中的权重
            )
            
            logger.info("集成检索器创建成功")
            return self._ensemble_retriever
            
        except Exception as e:
            logger.error(f"创建集成检索器失败: {e}")
            raise
    
    def add_documents_to_parent_retriever(self, documents: List[Document]):
        """向父文档检索器添加文档"""
        try:
            parent_retriever = self.create_parent_document_retriever()
            parent_retriever.add_documents(documents)
            logger.info(f"成功向父文档检索器添加 {len(documents)} 个文档")
        except Exception as e:
            logger.error(f"向父文档检索器添加文档失败: {e}")
            raise
    
    def search_with_advanced_retriever(self, 
                                     query: str, 
                                     retriever_type: str = "ensemble",
                                     k: int = None) -> List[Document]:
        """使用高级检索器进行搜索
        
        Args:
            query: 搜索查询
            retriever_type: 检索器类型 (ensemble, parent, multi_query, compression)
            k: 返回文档数量
        
        Returns:
            检索到的文档列表
        """
        try:
            k = k or settings.retrieval_k
            
            logger.info(f"🔍 高级检索器开始搜索: '{query[:50]}{'...' if len(query) > 50 else ''}'")
            logger.info(f"📊 检索参数: k={k}, 检索器类型={retriever_type}")
            
            if retriever_type == "ensemble":
                logger.info("🔧 创建集成检索器...")
                retriever = self.create_ensemble_retriever()
            elif retriever_type == "parent":
                logger.info("🔧 创建父文档检索器...")
                retriever = self.create_parent_document_retriever()
            elif retriever_type == "multi_query":
                logger.info("🔧 创建多查询检索器...")
                base_retriever = self.vector_service.vector_store.as_retriever(
                    search_kwargs={"k": k}
                )
                retriever = self.create_multi_query_retriever(base_retriever)
            elif retriever_type == "compression":
                logger.info("🔧 创建上下文压缩检索器...")
                base_retriever = self.vector_service.vector_store.as_retriever(
                    search_kwargs={"k": k}
                )
                retriever = self.create_contextual_compression_retriever(base_retriever)
            else:
                raise ValueError(f"不支持的检索器类型: {retriever_type}")
            
            # 执行检索
            logger.info("🔄 执行高级检索...")
            documents = retriever.get_relevant_documents(query)
            
            logger.info(f"📋 原始检索结果: {len(documents)} 个文档")
            
            # 记录检索结果详情
            for i, doc in enumerate(documents[:5]):  # 只记录前5个结果
                doc_id = doc.metadata.get('document_id', 'unknown')
                content_preview = doc.page_content[:100].replace('\n', ' ')
                logger.debug(f"📄 结果 {i+1}: doc_id={doc_id}, content='{content_preview}...'")
            
            logger.info(f"✅ 使用 {retriever_type} 检索器找到 {len(documents)} 个相关文档")
            if len(documents) == 0:
                logger.warning("⚠️ 未找到任何相关文档")
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ 高级检索器搜索失败: {e}")
            return []
    
    def get_retriever_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        return {
            "parent_retriever_initialized": self._parent_retriever is not None,
            "multi_query_retriever_initialized": self._multi_query_retriever is not None,
            "compression_retriever_initialized": self._compression_retriever is not None,
            "ensemble_retriever_initialized": self._ensemble_retriever is not None,
            "docstore_size": len(self.docstore.mget(list(self.docstore.yield_keys()))),
            "vector_store_stats": self.vector_service.get_vector_store_stats()
        }
    
    def reset_retrievers(self):
        """重置所有检索器实例"""
        self._parent_retriever = None
        self._multi_query_retriever = None
        self._compression_retriever = None
        self._ensemble_retriever = None
        self.docstore = InMemoryStore()
        logger.info("所有检索器已重置")
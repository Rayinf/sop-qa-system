from typing import List, Optional, Dict, Any, Tuple
import os
import logging
import numpy as np
from datetime import datetime, timezone
import threading
import inspect

# 禁用tqdm进度条以避免在向量化过程中卡住
os.environ['TQDM_DISABLE'] = '1'
try:
    import tqdm
    # 全局禁用进度条，防止在批处理时卡住
    from functools import partial
    tqdm.tqdm = partial(tqdm.tqdm, disable=True)
except ImportError:
    pass

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document, BaseRetriever
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.database import VectorIndex, Document as DocumentModel
from app.core.database import get_redis_client

logger = logging.getLogger(__name__)

class CategoryFilteredRetriever(BaseRetriever):
    """
    基于类别过滤的检索器
    """
    
    def __init__(self, vector_store: FAISS, category: str, map_function):
        super().__init__()
        self._vector_store = vector_store
        self._category = category
        self._map_function = map_function
    
    @property
    def vector_store(self):
        return self._vector_store
    
    @property
    def category(self):
        return self._category
    
    @property
    def map_function(self):
        return self._map_function
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """
        获取相关文档，根据类别进行过滤
        """
        try:
            logger.info(f"🔍 CategoryFilteredRetriever开始搜索: '{query[:50]}{'...' if len(query) > 50 else ''}', 类别: {self.category}")
            # 从向量数据库中搜索相关文档并获取相似度分数
            all_scored_docs = self.vector_store.similarity_search_with_score(query, k=settings.retrieval_k * settings.category_search_multiplier)
            logger.info(f"📊 CategoryFilteredRetriever原始结果: {len(all_scored_docs)} 个文档")
            
            similarity_threshold = settings.similarity_threshold
            
            # 根据类别与相似度过滤文档
            filtered_docs = []
            for doc, score in all_scored_docs:
                # 当向量库返回的是距离（越小越相似）或相似度（越大越相似）时，简单判断
                is_relevant = False
                if score is None:
                    is_relevant = True  # 若无分数信息则默认相关
                else:
                    # 经验判断：若分数大于阈值则认为相关；若存储的是距离，可以改成 score <= (1 - similarity_threshold)
                    # 若 score 越小越相似（距离），或 score 越大越相似（相似度），两种情况均做兼容判断
                    is_relevant = (score >= similarity_threshold) or (score <= (1 - similarity_threshold))
                if not is_relevant:
                    continue
                
                doc_category = doc.metadata.get('category', '通用文档')
                doc_title = doc.metadata.get('title', '').lower()
                
                # 如果文档类别是'other'，使用智能映射
                if doc_category == 'other':
                    mapped_category = self.map_function(doc_title, doc.page_content)
                    if mapped_category == self.category:
                        filtered_docs.append(doc)
                # 如果文档类别直接匹配
                elif doc_category == self.category:
                    filtered_docs.append(doc)
            
            # 限制返回结果数量
            final_docs = filtered_docs[:10]
            logger.info(f"✅ CategoryFilteredRetriever过滤完成: 返回 {len(final_docs)} 个文档")
            return final_docs
            
        except Exception as e:
            logger.error(f"CategoryFilteredRetriever检索失败: {e}")
            return []
    
    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """
        异步获取相关文档
        """
        return self._get_relevant_documents(query, run_manager=run_manager)

class VectorService:
    """向量化服务类"""
    
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> 'VectorService':
        """获取单例实例"""
        if cls._instance is None:
            cls()
        return cls._instance
    
    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        # 版本指纹（用于判断是否加载了新代码）
        self.version_fingerprint = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # 初始化嵌入模型 - 根据配置选择本地或API模式
        self.embedding_mode = settings.embedding_mode
        
        if self.embedding_mode == "api":
            # 使用 Qwen3 Embedding API
            import requests
            # 从环境变量读取API密钥
            self.api_key = os.getenv('DASHSCOPE_API_KEY', settings.embedding_api_key)
            self.api_base_url = settings.embedding_base_url
            self.api_headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            logger.info(f"🆔 VectorService 启动，指纹: {self.version_fingerprint}，使用API模式: {settings.embedding_base_url}")
        else:
            # 使用本地嵌入模型
            logger.info(f"🆔 VectorService 启动，指纹: {self.version_fingerprint}，使用本地模式: {settings.local_embedding_model}")
        
        # 包装为 LangChain 兼容的嵌入模型
        self._langchain_embeddings = self._create_langchain_wrapper()
        
        # 向量数据库路径
        self.vector_db_path = settings.vector_path
        self.ensure_vector_db_directory()
        
        # Redis客户端用于缓存
        self.redis_client = None
        
        # 向量数据库实例
        self._vector_store = None
        
        # 标记已初始化，避免重复初始化
        self._initialized = True
    
    def _create_langchain_wrapper(self):
        """创建LangChain兼容的嵌入包装器"""
        from langchain.embeddings.base import Embeddings
        
        if self.embedding_mode == "local":
            # 使用本地HuggingFace嵌入模型
            return HuggingFaceEmbeddings(
                model_name=settings.local_embedding_model,
                model_kwargs={'device': settings.local_embedding_device},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        # API模式 - 创建API包装器
        class Qwen3APIEmbeddingWrapper(Embeddings):
            def __init__(self, api_base_url, api_headers):
                self.api_base_url = api_base_url
                self.api_headers = api_headers
                import requests
                self.requests = requests
            
            def embed_documents(self, texts):
                """嵌入文档列表（分批处理并打印进度）"""
                total = len(texts)
                if total == 0:
                    return []
                logger.info(f"🔄 开始嵌入 {total} 个文本（API调用）")
                
                # API调用批处理大小，避免单次请求过大
                batch_size = 10
                
                all_embeddings = []
                for start in range(0, total, batch_size):
                    end = min(start + batch_size, total)
                    batch = texts[start:end]
                    
                    try:
                        # 调用API获取嵌入向量
                        payload = {
                            "model": settings.embedding_model_name,
                            "input": batch,
                            "dimensions": settings.embedding_dimensions,
                            "encoding_format": settings.embedding_encoding_format
                        }
                        
                        response = self.requests.post(
                            f"{self.api_base_url}/embeddings",
                            headers=self.api_headers,
                            json=payload,
                            timeout=30
                        )
                        response.raise_for_status()
                        
                        # 提取嵌入向量
                        result = response.json()
                        batch_embeddings = [data["embedding"] for data in result["data"]]
                        all_embeddings.extend(batch_embeddings)
                        
                        logger.info(f"⏳ 嵌入进度: {end}/{total}")
                        
                    except Exception as e:
                        logger.error(f"❌ API调用失败，批次 {start}-{end}: {e}")
                        # 如果批处理失败，尝试逐个处理
                        for single_text in batch:
                            try:
                                single_payload = {
                                    "model": settings.embedding_model_name,
                                    "input": [single_text],
                                    "dimensions": settings.embedding_dimensions,
                                    "encoding_format": settings.embedding_encoding_format
                                }
                                
                                single_response = self.requests.post(
                                    f"{self.api_base_url}/embeddings",
                                    headers=self.api_headers,
                                    json=single_payload,
                                    timeout=30
                                )
                                single_response.raise_for_status()
                                
                                single_result = single_response.json()
                                all_embeddings.append(single_result["data"][0]["embedding"])
                            except Exception as single_e:
                                logger.error(f"❌ 单个文本嵌入失败: {single_e}")
                                # 返回零向量作为fallback
                                all_embeddings.append([0.0] * settings.embedding_dimensions)
                
                logger.info(f"✅ 嵌入完成: {len(all_embeddings)} 个向量")
                
                # 验证返回的嵌入数量与输入文本数量匹配
                if len(all_embeddings) != total:
                    logger.error(f"❌ 嵌入数量不匹配: 期望 {total}, 实际 {len(all_embeddings)}")
                    # 补齐缺失的嵌入向量
                    while len(all_embeddings) < total:
                        all_embeddings.append([0.0] * settings.embedding_dimensions)
                    # 截断多余的嵌入向量
                    all_embeddings = all_embeddings[:total]
                    logger.info(f"🔧 已修正嵌入数量为: {len(all_embeddings)}")
                
                return all_embeddings
            
            def embed_query(self, text):
                """嵌入查询文本"""
                try:
                    payload = {
                        "model": settings.embedding_model_name,
                        "input": [text],
                        "dimensions": settings.embedding_dimensions,
                        "encoding_format": settings.embedding_encoding_format
                    }
                    
                    response = self.requests.post(
                        f"{self.api_base_url}/embeddings",
                        headers=self.api_headers,
                        json=payload,
                        timeout=30
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    return result["data"][0]["embedding"]
                except Exception as e:
                    logger.error(f"❌ 查询文本嵌入失败: {e}")
                    # 返回零向量作为fallback
                    return [0.0] * settings.embedding_dimensions
        
        # API模式下返回API包装器
        return Qwen3APIEmbeddingWrapper(self.api_base_url, self.api_headers)
    
    def switch_embedding_mode(self, new_mode: str):
        """动态切换embedding模式"""
        if new_mode not in ["api", "local"]:
            raise ValueError("embedding_mode must be 'api' or 'local'")
        
        if new_mode != self.embedding_mode:
            logger.info(f"🔄 切换embedding模式: {self.embedding_mode} -> {new_mode}")
            self.embedding_mode = new_mode
            
            # 重新初始化API相关属性
            if new_mode == "api":
                import requests
                # 从环境变量读取API密钥
                self.api_key = os.getenv('DASHSCOPE_API_KEY', settings.embedding_api_key)
                self.api_base_url = settings.embedding_base_url
                self.api_headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            
            # 重新创建嵌入包装器
            self._langchain_embeddings = self._create_langchain_wrapper()
            
            # 清空向量存储缓存，强制重新加载
            self._vector_store = None
            logger.info(f"✅ embedding模式切换完成: {new_mode}")
            # 通知向量库相关组件刷新（触发问答链重建、清理缓存）
            try:
                self._notify_vector_store_updated()
            except Exception as e:
                logger.warning(f"通知向量库更新事件失败: {e}")
    
    def _get_vector_path_for_mode(self, mode: str) -> str:
        """根据embedding模式获取向量存储路径"""
        if mode == "api":
            return os.path.join(self.vector_db_path, "api_text_embedding_v4")
        else:  # local mode
            return os.path.join(self.vector_db_path, "local_text_embedding_ada_002")
    
    def ensure_vector_db_directory(self):
        """确保向量数据库目录存在"""
        os.makedirs(self.vector_db_path, exist_ok=True)
    
    def get_redis_client(self):
        """获取Redis客户端"""
        if self.redis_client is None:
            self.redis_client = get_redis_client()
        return self.redis_client

    def _bump_vector_store_version(self):
        """增加向量库版本号（用于通知其他服务刷新检索器/缓存）"""
        try:
            client = self.get_redis_client()
            if client:
                client.incr("vector_store:version")
        except Exception as e:
            logger.warning(f"更新向量库版本号失败: {e}")

    def _clear_qa_answer_cache(self, pattern: str = "qa_answer:*") -> int:
        """清除QA答案缓存"""
        try:
            client = self.get_redis_client()
            if client:
                keys = client.keys(pattern)
                if keys:
                    return client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"清除QA答案缓存失败: {e}")
            return 0

    def _notify_vector_store_updated(self):
        """向量库更新后通知：提升版本并清理QA缓存"""
        self._bump_vector_store_version()
        deleted = self._clear_qa_answer_cache()
        if deleted:
            logger.info(f"向量库更新后已清理 {deleted} 个QA答案缓存")
    
    @property
    def vector_store(self) -> Optional[FAISS]:
        """获取向量数据库实例"""
        if self._vector_store is None:
            self._vector_store = self.load_vector_store()
        return self._vector_store
    
    def load_vector_store(self) -> Optional[FAISS]:
        """加载向量数据库"""
        try:
            logger.info("🔄 开始加载向量数据库...")
            # 根据当前embedding模式获取正确的向量存储路径
            mode_specific_path = self._get_vector_path_for_mode(self.embedding_mode)
            index_path = os.path.join(mode_specific_path, "faiss_index")
            index_file = os.path.join(index_path, "index.faiss")
            
            logger.info(f"📁 检查向量数据库路径: {index_path} (模式: {self.embedding_mode})")
            
            if os.path.exists(index_file):
                logger.info(f"📄 找到向量索引文件: {index_file}")
                logger.info("⚙️ 正在反序列化向量数据库...")
                
                # 兼容不同版本的 LangChain：老版本没有 allow_dangerous_deserialization 参数
                load_kwargs = {}
                try:
                    sig = inspect.signature(FAISS.load_local)
                    if "allow_dangerous_deserialization" in sig.parameters:
                        load_kwargs["allow_dangerous_deserialization"] = True
                except Exception:
                    # 签名检查失败则不传该参数
                    pass

                try:
                    vector_store = FAISS.load_local(
                        index_path,
                        self._langchain_embeddings,
                        **load_kwargs,
                    )
                except TypeError as te:
                    # 向后兼容：如果报 unexpected keyword argument，则回退为不带该参数
                    if "allow_dangerous_deserialization" in str(te):
                        logger.warning("当前 LangChain 版本不支持 allow_dangerous_deserialization，自动回退为安全加载模式")
                        vector_store = FAISS.load_local(index_path, self._langchain_embeddings)
                    else:
                        raise
                
                # 获取向量数据库统计信息
                total_vectors = vector_store.index.ntotal
                vector_dimension = vector_store.index.d
                
                # 维度一致性检查：防止旧索引与新嵌入模型维度不一致
                try:
                    test_dim = len(self._langchain_embeddings.embed_query("test"))
                except Exception:
                    test_dim = None
                if test_dim and test_dim != vector_dimension:
                    logger.warning(
                        f"⚠️ 向量索引维度({vector_dimension})与当前嵌入模型维度({test_dim})不一致，建议重建索引；暂不加载旧索引以避免运行时错误"
                    )
                    return None
                
                logger.info(f"✅ 向量数据库加载成功")
                logger.info(f"📊 向量数据库统计: {total_vectors} 个向量, 维度: {vector_dimension}")
                
                return vector_store
            else:
                logger.info(f"⚠️ 向量数据库不存在: {index_file}")
                logger.info("💡 将在首次添加文档时创建新的向量数据库")
                return None
        except Exception as e:
            logger.error(f"❌ 加载向量数据库失败: {e}")
            return None
    
    def save_vector_store(self, vector_store: FAISS):
        """保存向量数据库"""
        try:
            logger.info("💾 开始保存向量数据库...")
            # 根据当前embedding模式获取正确的向量存储路径
            mode_specific_path = self._get_vector_path_for_mode(self.embedding_mode)
            index_path = os.path.join(mode_specific_path, "faiss_index")
            
            # 获取保存前的统计信息
            total_vectors = vector_store.index.ntotal
            vector_dimension = vector_store.index.d
            
            logger.info(f"📊 准备保存: {total_vectors} 个向量, 维度: {vector_dimension}")
            logger.info(f"📁 保存路径: {index_path}")
            
            # 确保目录存在
            os.makedirs(index_path, exist_ok=True)
            
            # 保存向量数据库
            vector_store.save_local(index_path)
            
            # 更新当前实例的向量存储
            self._vector_store = vector_store
            
            # 验证保存结果
            index_file = os.path.join(index_path, "index.faiss")
            if os.path.exists(index_file):
                file_size = os.path.getsize(index_file)
                logger.info(f"✅ 向量数据库保存成功")
                logger.info(f"📄 索引文件大小: {file_size / 1024 / 1024:.2f} MB")
                
                # 强制重新加载向量数据库以确保最新状态
                logger.info("🔄 重新加载向量数据库以确保最新状态...")
                self._vector_store = None  # 清除缓存
                reloaded_store = self.load_vector_store()  # 重新加载
                if reloaded_store:
                    self._vector_store = reloaded_store
                    logger.info(f"✅ 向量数据库重新加载成功，包含 {reloaded_store.index.ntotal} 个向量")
                else:
                    logger.warning("⚠️ 向量数据库重新加载失败，使用当前实例")
                    self._vector_store = vector_store

                # 通知相关组件向量库已更新（提升版本并清理QA答案缓存）
                try:
                    self._notify_vector_store_updated()
                except Exception as e:
                    logger.warning(f"向量库更新通知失败: {e}")
            else:
                logger.warning("⚠️ 保存完成但未找到索引文件")
                
        except Exception as e:
            logger.error(f"❌ 保存向量数据库失败: {e}")

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """创建文本嵌入向量"""
        try:
            logger.info(f"🧮 开始创建嵌入向量: {len(texts)} 个文本片段")
            
            # 记录文本统计信息
            total_chars = sum(len(text) for text in texts)
            avg_length = total_chars / len(texts) if texts else 0
            
            logger.info(f"📝 文本统计: 总字符数 {total_chars}, 平均长度 {avg_length:.1f}")
            
            # 创建嵌入向量（内部已分批）
            embeddings = self._langchain_embeddings.embed_documents(texts)
            
            # 验证结果
            if embeddings:
                vector_dim = len(embeddings[0]) if embeddings[0] else 0
                logger.info(f"✅ 嵌入向量创建成功: {len(embeddings)} 个向量, 维度: {vector_dim}")
            else:
                logger.warning("⚠️ 未生成任何嵌入向量")
                
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ 创建嵌入向量失败: {e}")
            raise

    def create_single_embedding(self, text: str) -> List[float]:
        """创建单个文本的嵌入向量"""
        try:
            embedding = self._langchain_embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"创建单个嵌入向量失败: {e}")
            raise

    def _build_prefixed_text(self, doc: Document) -> str:
        """构造用于嵌入的前置拼接文本（标题/类别 + 正文）"""
        title = (doc.metadata or {}).get('title') or ''
        category = (doc.metadata or {}).get('category') or ''
        parts = []
        if title:
            parts.append(f"标题: {title}")
        if category:
            parts.append(f"类别: {category}")
        if parts:
            prefix = " | ".join(parts)
            return f"{prefix}\n{doc.page_content}"
        return doc.page_content

    def add_documents_to_vector_store(self, 
                                     documents: List[Document],
                                     document_id: str) -> Tuple[bool, Optional[List[List[float]]]]:
        """增量将新文档片段添加到向量数据库（只计算新片段嵌入，不重建索引）"""
        try:
            logger.info(f"📚 开始增量添加文档到向量数据库: {document_id}")
            logger.info(f"📄 原始片段数量: {len(documents)}")

            # 进度：准备元数据
            self.update_vectorization_progress(document_id, {
                "document_id": document_id,
                "status": "processing",
                "progress": 20,
                "current_step": "准备文档数据",
                "total_steps": 4,
                "current_step_index": 1,
                "message": f"正在准备 {len(documents)} 个文档片段的元数据",
                "error": None
            })

            # 确保每个片段都带有 document_id
            for i, doc in enumerate(documents):
                doc.metadata['document_id'] = document_id
                if i % 1000 == 0:
                    logger.info(f"📝 处理文档片段进度: {i+1}/{len(documents)}")

            # 构造前缀文本
            prefixed_docs = [
                Document(page_content=self._build_prefixed_text(doc), metadata=doc.metadata)
                for doc in documents
            ]

            # 加载（或懒加载）现有向量库
            vector_store = self.vector_store  # 触发加载

            # 过滤已存在的片段（根据 chunk_id 去重，只添加新块）
            existing_chunk_ids: set = set()
            if vector_store is not None:
                try:
                    if hasattr(vector_store, "docstore") and hasattr(vector_store.docstore, "_dict"):
                        for _id, _doc in vector_store.docstore._dict.items():
                            cid = (_doc.metadata or {}).get('chunk_id')
                            if cid:
                                existing_chunk_ids.add(cid)
                except Exception as e:
                    logger.warning(f"⚠️ 读取现有片段失败，跳过去重: {e}")

            new_docs: List[Document] = []
            skipped = 0
            for d in prefixed_docs:
                cid = d.metadata.get('chunk_id')
                if cid and cid in existing_chunk_ids:
                    skipped += 1
                    continue
                new_docs.append(d)

            logger.info(f"📊 新片段数量: {len(new_docs)}（跳过已存在: {skipped}）")

            if len(new_docs) == 0:
                logger.info("✅ 没有需要新增的片段，向量库保持不变")
                return True, []

            # 进度：计算新片段嵌入
            self.update_vectorization_progress(document_id, {
                "document_id": document_id,
                "status": "processing",
                "progress": 45,
                "current_step": "计算新片段嵌入",
                "total_steps": 4,
                "current_step_index": 2,
                "message": f"正在计算 {len(new_docs)} 个新文档片段的嵌入向量",
                "error": None
            })

            new_texts = [d.page_content for d in new_docs]
            new_metas = [d.metadata for d in new_docs]
            new_embeddings = self.create_embeddings(new_texts)

            if len(new_embeddings) != len(new_texts):
                raise ValueError(f"新文本数量({len(new_texts)})与嵌入数量({len(new_embeddings)})不匹配")
            if not new_embeddings:
                raise ValueError("未生成任何新嵌入向量")

            # 进度：写入向量索引
            self.update_vectorization_progress(document_id, {
                "document_id": document_id,
                "status": "processing",
                "progress": 70,
                "current_step": "写入向量索引",
                "total_steps": 4,
                "current_step_index": 3,
                "message": f"正在将 {len(new_docs)} 个新片段写入向量索引",
                "error": None
            })

            if vector_store is None:
                # 首次创建索引，仅使用新片段
                logger.info("🆕 创建新的向量索引（仅包含新片段）...")
                text_embedding_pairs = list(zip(new_texts, new_embeddings))
                vector_store = FAISS.from_embeddings(
                    text_embedding_pairs,
                    self._langchain_embeddings,
                    metadatas=new_metas
                )
                logger.info(f"✅ 新索引创建成功，包含 {len(new_docs)} 个片段")
            else:
                # 增量追加嵌入
                logger.info("➕ 向现有索引增量追加新片段...")
                text_embedding_pairs = list(zip(new_texts, new_embeddings))
                # 使用 add_embeddings 避免对新片段再次计算嵌入
                vector_store.add_embeddings(
                    text_embedding_pairs,
                    metadatas=new_metas
                )
                logger.info(f"✅ 追加完成，当前索引总向量数: {vector_store.index.ntotal}")

            # 进度：保存向量库
            self.update_vectorization_progress(document_id, {
                "document_id": document_id,
                "status": "processing",
                "progress": 85,
                "current_step": "保存向量数据库",
                "total_steps": 4,
                "current_step_index": 4,
                "message": "正在保存向量数据库到磁盘",
                "error": None
            })

            # 保存并刷新内存中的引用
            self.save_vector_store(vector_store)

            # 返回本批新文档的嵌入
            return True, new_embeddings

        except Exception as e:
            logger.error(f"❌ 增量添加文档到向量数据库失败: {e}")
            return False, None


    def update_vector_indices_in_db(self, 
                                   db: Session, 
                                   document_id: str, 
                                   documents: List[Document],
                                   embeddings: Optional[List[List[float]]] = None) -> bool:
        """更新数据库中的向量索引记录（可复用已计算的嵌入）"""
        try:
            # 创建/复用嵌入向量
            texts = [self._build_prefixed_text(doc) for doc in documents]
            if embeddings is None:
                embeddings = self.create_embeddings(texts)
            
            # 更新数据库中的向量索引
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                chunk_id = doc.metadata.get('chunk_id')
                
                vector_index = db.query(VectorIndex).filter(
                    VectorIndex.document_id == document_id,
                    VectorIndex.chunk_id == chunk_id
                ).first()
                
                if vector_index:
                    vector_index.embedding_vector = embedding
                    vector_index.vector_created_at = datetime.now(timezone.utc)
                    vector_index.status = "vectorized"
            
            db.commit()
            logger.info(f"更新 {len(documents)} 个向量索引记录")
            return True
            
        except Exception as e:
            logger.error(f"更新向量索引记录失败: {e}")
            db.rollback()
            return False

    def vectorize_document(self, 
                          db: Session, 
                          document_id: str, 
                          documents: List[Document]) -> bool:
        """向量化文档的主要方法"""
        try:
            logger.info(f"🚀 开始向量化文档: {document_id}")
            logger.info(f"📊 文档统计: {len(documents)} 个片段")
            
            # 计算文档总字符数
            total_chars = sum(len(doc.page_content) for doc in documents)
            logger.info(f"📝 文档内容: 总计 {total_chars} 字符")
            
            # 初始化进度
            self.update_vectorization_progress(document_id, {
                "document_id": document_id,
                "status": "processing",
                "progress": 0,
                "current_step": "开始向量化",
                "total_steps": 4,
                "current_step_index": 0,
                "message": "正在准备向量化任务",
                "error": None
            })
            
            # 步骤1: 添加到向量数据库（并取得嵌入）
            logger.info("🔄 步骤1: 添加文档到向量数据库")
            self.update_vectorization_progress(document_id, {
                "document_id": document_id,
                "status": "processing",
                "progress": 25,
                "current_step": "生成向量嵌入",
                "total_steps": 4,
                "current_step_index": 1,
                "message": f"正在处理 {len(documents)} 个文档片段",
                "error": None
            })
            
            success, embeddings = self.add_documents_to_vector_store(documents, document_id)
            if not success:
                logger.error("❌ 步骤1失败: 无法添加到向量数据库")
                self.update_vectorization_progress(document_id, {
                    "document_id": document_id,
                    "status": "error",
                    "progress": 25,
                    "current_step": "向量化失败",
                    "total_steps": 4,
                    "current_step_index": 1,
                    "message": "向量数据库添加失败",
                    "error": "向量数据库添加失败"
                })
                return False
            logger.info("✅ 步骤1完成: 向量数据库更新成功")
            
            # 步骤2: 更新数据库中的向量索引（复用同批嵌入）
            logger.info("🔄 步骤2: 更新数据库向量索引记录")
            self.update_vectorization_progress(document_id, {
                "document_id": document_id,
                "status": "processing",
                "progress": 50,
                "current_step": "更新数据库索引",
                "total_steps": 4,
                "current_step_index": 2,
                "message": "正在更新向量索引记录",
                "error": None
            })
            
            success = self.update_vector_indices_in_db(db, document_id, documents, embeddings=embeddings)
            if not success:
                logger.error("❌ 步骤2失败: 无法更新数据库索引")
                self.update_vectorization_progress(document_id, {
                    "document_id": document_id,
                    "status": "error",
                    "progress": 50,
                    "current_step": "索引更新失败",
                    "total_steps": 4,
                    "current_step_index": 2,
                    "message": "数据库索引更新失败",
                    "error": "数据库索引更新失败"
                })
                return False
            logger.info("✅ 步骤2完成: 数据库索引更新成功")
            
            # 步骤3: 更新文档状态
            logger.info("🔄 步骤3: 更新文档状态")
            self.update_vectorization_progress(document_id, {
                "document_id": document_id,
                "status": "processing",
                "progress": 75,
                "current_step": "更新文档状态",
                "total_steps": 4,
                "current_step_index": 3,
                "message": "正在更新文档状态",
                "error": None
            })
            
            document = db.query(DocumentModel).filter(
                DocumentModel.id == document_id
            ).first()
            
            if document:
                old_status = document.status
                document.status = "vectorized"
                document.updated_at = datetime.now(timezone.utc)
                db.commit()
                logger.info(f"📄 文档状态更新: {old_status} → vectorized")
            else:
                logger.warning(f"⚠️ 未找到文档记录: {document_id}")
            
            # 步骤4: 缓存向量化状态
            logger.info("🔄 步骤4: 缓存向量化状态")
            self.update_vectorization_progress(document_id, {
                "document_id": document_id,
                "status": "completed",
                "progress": 100,
                "current_step": "向量化完成",
                "total_steps": 4,
                "current_step_index": 4,
                "message": f"成功处理 {len(documents)} 个片段，共 {total_chars} 字符",
                "error": None
            })
            
            self.cache_vectorization_status(document_id, "completed")
            logger.info("✅ 步骤4完成: 状态缓存成功")
            
            logger.info(f"🎉 文档向量化完成: {document_id}")
            logger.info(f"📈 处理结果: {len(documents)} 个片段, {total_chars} 字符")
            return True
            
        except Exception as e:
            logger.error(f"❌ 文档向量化失败 {document_id}: {e}")
            # 更新错误状态
            self.update_vectorization_progress(document_id, {
                "document_id": document_id,
                "status": "error",
                "progress": 0,
                "current_step": "向量化失败",
                "total_steps": 4,
                "current_step_index": 0,
                "message": f"向量化过程中发生错误: {str(e)}",
                "error": str(e)
            })
            return False

    def search_similar_documents(self, 
                                query: str, 
                                k: int = None,
                                score_threshold: float = None,
                                filter_dict: Optional[Dict[str, Any]] = None,
                                active_kb_ids: Optional[List[str]] = None) -> List[Tuple[Document, float]]:
        """搜索相似文档
        Args:
            query: 查询文本
            k: 返回文档数量，默认读取 settings.retrieval_k
            score_threshold: 相似度阈值，低于该阈值的文档将被过滤（注意：此处的分数定义为"相似度"，越大越相似）
            filter_dict: 额外过滤条件（例如 {"category": "manual"}），会直接传递给 FAISS 的 filter 参数
            active_kb_ids: 激活的知识库ID列表，用于限制搜索范围
        Returns:
            (Document, similarity) 列表，分数为相似度（cosine，相似度越大越相关）
        """
        try:
            # 使用配置文件中的默认值
            if k is None:
                k = settings.retrieval_k
            if score_threshold is None:
                score_threshold = settings.similarity_threshold
                
            logger.info(f"🔍 开始向量搜索: '{query[:50]}{'...' if len(query) > 50 else ''}'")
            logger.info(f"📊 搜索参数: k={k}, similarity_threshold={score_threshold}")
            
            if self.vector_store is None:
                logger.warning("⚠️ 向量数据库未初始化")
                return []
            
            # 获取向量数据库统计信息
            total_vectors = self.vector_store.index.ntotal
            logger.info(f"📚 向量数据库状态: 总计 {total_vectors} 个向量")
            
            # 先召回候选
            if settings.vector_use_mmr:
                logger.info("🔄 执行MMR检索（最大边际相关性）...")
                # 如果有过滤条件，需要搜索更多结果然后手动过滤
                fetch_k = min(total_vectors, max(1000, settings.vector_mmr_fetch_k * settings.filter_search_multiplier)) if filter_dict else settings.vector_mmr_fetch_k
                k_for_mmr = min(total_vectors, max(1000, k * settings.filter_search_multiplier)) if filter_dict else k
                docs = self.vector_store.max_marginal_relevance_search(
                    query,
                    k=k_for_mmr,
                    fetch_k=fetch_k,
                    lambda_mult=settings.vector_mmr_lambda_mult
                )
                # 与非MMR分支保持一致的结构：(doc, 原始距离[MMR无则为None])
                docs_with_scores = [(doc, None) for doc in docs]
                logger.info(f"📋 MMR检索候选: {len(docs_with_scores)} 个文档")
            else:
                logger.info("🔄 执行向量相似度搜索（基础召回）...")
                # 如果有过滤条件，需要搜索更多结果然后手动过滤
                search_k = min(total_vectors, max(1000, k * settings.filter_search_multiplier)) if filter_dict else k
                docs_with_scores = self.vector_store.similarity_search_with_score(
                    query,
                    k=search_k
                )
                logger.info(f"📋 基础召回: {len(docs_with_scores)} 个文档")
            
            # 手动应用过滤器（基于metadata）
            filters_applied = []
            
            # 应用知识库过滤
            if active_kb_ids:
                logger.info(f"🎯 应用知识库过滤器: {active_kb_ids}")
                kb_filtered_docs = []
                for doc, raw_score in docs_with_scores:
                    doc_kb_id = doc.metadata.get('kb_id')
                    if doc_kb_id and str(doc_kb_id) in [str(kb_id) for kb_id in active_kb_ids]:
                        kb_filtered_docs.append((doc, raw_score))
                docs_with_scores = kb_filtered_docs
                filters_applied.append(f"知识库: {len(active_kb_ids)}个")
                logger.info(f"📋 知识库过滤后候选: {len(docs_with_scores)} 个文档")
            
            # 应用其他过滤器
            if filter_dict:
                logger.info(f"🔄 应用其他过滤器: {filter_dict}")
                filtered_docs_with_scores = []
                for doc, raw_score in docs_with_scores:
                    match = True
                    for key, value in filter_dict.items():
                        if doc.metadata.get(key) != value:
                            match = False
                            break
                    if match:
                        filtered_docs_with_scores.append((doc, raw_score))
                docs_with_scores = filtered_docs_with_scores
                filters_applied.append(f"其他: {len(filter_dict)}个条件")
                logger.info(f"📋 其他过滤后候选: {len(docs_with_scores)} 个文档")
            
            if filters_applied:
                logger.info(f"✅ 过滤器应用完成: {', '.join(filters_applied)}")
            
            # 限制候选数量到原始k值
            docs_with_scores = docs_with_scores[:k]
            
            # 统一计算相似度（使用与入库一致的前缀拼接文本 + 归一化向量点积）
            if not docs_with_scores:
                return []
            
            query_embedding = np.array(self.create_single_embedding(query), dtype=np.float32)
            prefixed_texts = [self._build_prefixed_text(doc) for doc, _ in docs_with_scores]
            doc_embeddings = self._langchain_embeddings.embed_documents(prefixed_texts)
            
            docs_with_similarities: List[Tuple[Document, float]] = []
            for (doc, raw_score), doc_emb in zip(docs_with_scores, doc_embeddings):
                doc_vec = np.array(doc_emb, dtype=np.float32)
                # 由于我们的嵌入已做 normalize_embeddings=True，内积即为余弦相似度
                similarity = float(np.dot(query_embedding, doc_vec))
                # 将分数信息写回 metadata
                # - 最终语义分数：metadata['score']（越大越相似）
                # - 原始FAISS距离（若存在）：metadata['faiss_distance'] 以便排查问题
                doc.metadata['score'] = similarity
                if raw_score is not None:
                    try:
                        doc.metadata['faiss_distance'] = float(raw_score)
                    except Exception:
                        doc.metadata['faiss_distance'] = None
                else:
                    doc.metadata['faiss_distance'] = None
                docs_with_similarities.append((doc, similarity))
            
            # 按相似度降序排序
            docs_with_similarities.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"原始搜索结果（按相似度降序）: {len(docs_with_similarities)} 个文档")
            for i, (doc, sim) in enumerate(docs_with_similarities):
                logger.info(f"  - 结果 {i+1}: similarity={sim:.4f}, title='{doc.metadata.get('title', 'N/A')}', filename='{doc.metadata.get('filename', 'N/A')}', doc_id='{doc.metadata.get('document_id', 'N/A')}'")
            
            # 应用阈值过滤（基于相似度）
            logger.info(f"🔄 应用相似度阈值过滤: >= {score_threshold}")
            filtered_docs_with_scores: List[Tuple[Document, float]] = [
                (doc, sim) for doc, sim in docs_with_similarities if sim >= score_threshold
            ]
            
            filtered_count = len(filtered_docs_with_scores)
            filtered_out = len(docs_with_similarities) - filtered_count
            logger.info(f"✅ 初步过滤后结果: {filtered_count} 个片段")
            if filtered_out > 0:
                logger.info(f"🚫 过滤掉 {filtered_out} 个低分片段")

            # —— 去重：同一文档只保留最高相似度片段，但确保至少返回3种不同类别的文档 ——
            logger.info("🔄 去重同一文档，保留最高相似度片段…")
            doc_best_map: Dict[str, Tuple[Document, float]] = {}
            for doc, sim in filtered_docs_with_scores:
                doc_id = doc.metadata.get("document_id", "unknown")
                if doc_id not in doc_best_map or sim > doc_best_map[doc_id][1]:
                    doc_best_map[doc_id] = (doc, sim)
            unique_docs_with_scores = list(doc_best_map.values())
            
            # 按相似度降序排序
            unique_docs_with_scores.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"📋 去重后剩余 {len(unique_docs_with_scores)} 个文档")
            
            # 确保多样性：尝试从不同类别中选择文档
            category_docs: Dict[str, List[Tuple[Document, float]]] = {}
            for doc, sim in unique_docs_with_scores:
                category = doc.metadata.get("category", "unknown")
                if category not in category_docs:
                    category_docs[category] = []
                category_docs[category].append((doc, sim))
            
            for category, docs in category_docs.items():
                logger.info(f"类别 '{category}' 有 {len(docs)} 个文档")
            
            if len(category_docs) > 1:
                diverse_results: List[Tuple[Document, float]] = []
                # 首先从每个类别中取最高分文档
                for category, docs in category_docs.items():
                    if docs:
                        diverse_results.append(docs[0])
                # 然后按相似度排序，添加剩余文档直到达到k
                remaining_slots = k - len(diverse_results)
                if remaining_slots > 0:
                    remaining_docs: List[Tuple[Document, float]] = []
                    for category, docs in category_docs.items():
                        if len(docs) > 1:
                            remaining_docs.extend(docs[1:])
                    remaining_docs.sort(key=lambda x: x[1], reverse=True)
                    diverse_results.extend(remaining_docs[:remaining_slots])
                diverse_results.sort(key=lambda x: x[1], reverse=True)
                logger.info(f"📋 多样化后返回 {len(diverse_results)} 个文档")
                return diverse_results
            
            logger.info(f"📋 仅有单一类别，返回 {min(len(unique_docs_with_scores), k)} 个文档")
            return unique_docs_with_scores[:k]
            
        except Exception as e:
            logger.error(f"❌ 向量搜索失败: {e}")
            return []
    
    def search_by_document_id(self, 
                             document_id: str, 
                             k: int = 10) -> List[Document]:
        """根据文档ID搜索文档片段"""
        try:
            if self.vector_store is None:
                return []
            
            # 使用过滤器搜索特定文档的片段
            filter_dict = {"document_id": document_id}
            
            # 获取所有匹配的文档
            all_docs = self.vector_store.similarity_search(
                "",  # 空查询
                k=k,
                filter=filter_dict
            )
            
            return all_docs
            
        except Exception as e:
            logger.error(f"根据文档ID搜索失败: {e}")
            return []
    
    def delete_document_from_vector_store(self, document_id: str) -> bool:
        """从向量数据库中删除文档"""
        try:
            if self.vector_store is None:
                return True
            
            # FAISS不支持直接删除，需要重建索引
            # 这里我们标记为删除，在重建时排除
            self.cache_deleted_document(document_id)
            
            logger.info(f"标记文档为删除: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False
    
    def rebuild_vector_store(self, db: Session) -> bool:
        """重建向量数据库（排除已删除的文档）"""
        try:
            logger.info("🔄 开始重建向量数据库")
            
            # 获取所有有效的文档片段
            logger.info("📋 查询向量索引记录...")
            vector_indices = db.query(VectorIndex).all()

            if not vector_indices:
                logger.info("⚠️ 没有需要重建的向量数据")
                return True

            logger.info(f"📊 找到 {len(vector_indices)} 个向量索引记录")

            # 获取已删除的文档ID
            logger.info("🗑️ 检查已删除的文档...")
            try:
                deleted_docs = self.get_deleted_documents()
                if deleted_docs:
                    logger.info(f"📋 已删除文档数量: {len(deleted_docs)}")
                else:
                    logger.info("✅ 没有已删除的文档")
            except Exception as e:
                logger.warning(f"⚠️ 无法获取删除文档列表: {e}")
                deleted_docs = set()  # 如果Redis不可用，假设没有删除的文档

            # 过滤掉已删除的文档
            logger.info("🔄 过滤有效的向量索引...")
            valid_indices = [
                vi for vi in vector_indices 
                if vi.document_id not in deleted_docs
            ]

            filtered_count = len(vector_indices) - len(valid_indices)
            if filtered_count > 0:
                logger.info(f"🚫 过滤掉 {filtered_count} 个已删除文档的索引")
            
            logger.info(f"✅ 有效向量索引: {len(valid_indices)} 个")

            if not valid_indices:
                # 创建空的向量数据库
                logger.info("🗑️ 所有文档都已删除，创建空向量数据库")
                self._vector_store = None
                return True

            # 重建文档列表
            logger.info("📄 重建文档列表...")
            documents = []
            total_chars = 0
            
            for i, vi in enumerate(valid_indices):
                # 确保metadata是字典类型
                metadata = vi.vector_metadata if vi.vector_metadata else {}
                if not isinstance(metadata, dict):
                    metadata = {}
                
                doc = Document(
                    page_content=vi.chunk_text,
                    metadata=metadata
                )
                documents.append(doc)
                total_chars += len(vi.chunk_text)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"📝 已处理 {i + 1}/{len(valid_indices)} 个文档片段")

            logger.info(f"📚 文档重建完成: {len(documents)} 个片段, 总计 {total_chars} 字符")

            # 获取重建前的统计信息
            old_vector_count = 0
            if self._vector_store is not None:
                old_vector_count = self._vector_store.index.ntotal
                logger.info(f"📊 重建前向量数量: {old_vector_count}")

            # 使用前缀文本创建新的向量数据库
            logger.info("🔄 创建新的向量数据库（使用前缀文本）...")
            prefixed_docs = [
                Document(page_content=self._build_prefixed_text(doc), metadata=doc.metadata)
                for doc in documents
            ]
            new_vector_store = FAISS.from_documents(prefixed_docs, self._langchain_embeddings)
            
            new_vector_count = new_vector_store.index.ntotal
            vector_dimension = new_vector_store.index.d
            logger.info(f"✅ 新向量数据库创建成功: {new_vector_count} 个向量, 维度: {vector_dimension}")

            # 保存新的向量数据库
            logger.info("💾 保存新的向量数据库...")
            self.save_vector_store(new_vector_store)

            # 清除删除标记
            logger.info("🧹 清除删除标记...")
            self.clear_deleted_documents()
            logger.info("✅ 删除标记已清除")

            logger.info("🎉 向量数据库重建完成")
            logger.info(f"📈 重建统计:")
            logger.info(f"   - 原向量数量: {old_vector_count}")
            logger.info(f"   - 新向量数量: {new_vector_count}")
            logger.info(f"   - 文档片段数: {len(documents)}")
            logger.info(f"   - 总字符数: {total_chars}")
            
            return True

        except Exception as e:
            logger.error(f"❌ 重建向量数据库失败: {e}")
            return False
    
    def rebuild_vector_store_from_documents(self, documents: List[Document]) -> bool:
        """从文档列表重建向量数据库"""
        try:
            if not documents:
                logger.info("没有文档需要重建向量数据库")
                return True
            
            # 使用前缀文本创建新的向量数据库
            prefixed_docs = [
                Document(page_content=self._build_prefixed_text(doc), metadata=doc.metadata)
                for doc in documents
            ]
            new_vector_store = FAISS.from_documents(prefixed_docs, self._langchain_embeddings)
            
            # 保存新的向量数据库
            self.save_vector_store(new_vector_store)
            
            logger.info(f"向量数据库重建完成，包含 {len(documents)} 个文档")
            return True
            
        except Exception as e:
            logger.error(f"从文档列表重建向量数据库失败: {e}")
            return False
    
    def get_category_retriever(self, category: str) -> Optional[Any]:
        """
        为指定类别创建检索器
        
        Args:
            category: 文档类别
            
        Returns:
            该类别的检索器，如果没有该类别的文档则返回None
        """
        try:
            if self.vector_store is None:
                logger.warning(f"向量数据库未初始化，无法为类别 '{category}' 创建检索器")
                return None
            
            # 检查该类别是否有文档
            test_docs = self.vector_store.similarity_search(
                "测试", k=settings.retrieval_k * settings.category_check_multiplier  # 获取更多文档用于检查
            )
            
            # 智能类别映射：基于文档标题和内容进行分类
            category_docs = []
            for doc in test_docs:
                doc_category = doc.metadata.get('category', '通用文档')
                doc_title = doc.metadata.get('title', '').lower()
                
                # 特殊处理: 当请求类别为 'other' 时，直接包含标注为 'other' 的文档
                if category == 'other':
                    if doc_category == 'other':
                        category_docs.append(doc)
                    continue
                
                # 如果文档类别是'other'，根据标题进行智能映射
                if doc_category == 'other':
                    mapped_category = self._map_document_to_category(doc_title, doc.page_content)
                    if mapped_category == category:
                        category_docs.append(doc)
                # 如果文档类别直接匹配
                elif doc_category == category:
                    category_docs.append(doc)
            
            if not category_docs:
                logger.warning(f"类别 '{category}' 没有找到相关文档")
                return None
            
            logger.info(f"为类别 '{category}' 创建检索器，找到 {len(category_docs)} 个文档")
            
            # 使用已定义的CategoryFilteredRetriever类
            return CategoryFilteredRetriever(self.vector_store, category, self._map_document_to_category)
            
        except Exception as e:
            logger.error(f"为类别 '{category}' 创建检索器失败: {e}")
            return None
    
    def _map_document_to_category(self, title: str, content: str) -> str:
        """
        根据文档标题和内容智能映射到合适的类别
        
        Args:
            title: 文档标题
            content: 文档内容
            
        Returns:
            映射的类别名称
        """
        title_lower = title.lower()
        content_lower = content.lower()[:500]  # 只检查前500字符
        
        # 质量手册类别关键词
        manual_keywords = ['质量手册', '管理制度', '规范', '标准', '体系', '方针', '政策']
        if any(keyword in title_lower for keyword in manual_keywords):
            return 'manual'
        
        # 开发类别关键词
        development_keywords = ['设计开发', '开发', '技术', '编程', '系统设计', '软件', '代码']
        if any(keyword in title_lower for keyword in development_keywords):
            return 'development'
        
        # 程序类别关键词
        procedure_keywords = ['程序', '流程', '操作', '作业指导', '工作流程', '步骤']
        if any(keyword in title_lower for keyword in procedure_keywords):
            return 'procedure'
        
        # 记录类别关键词
        record_keywords = ['记录', '表单', '清单', '报告', '模板', '检查']
        if any(keyword in title_lower for keyword in record_keywords):
            return 'record'
        
        # 如果标题无法确定，检查内容
        if '开发' in content_lower or '设计' in content_lower:
            return 'development'
        elif '质量' in content_lower or '管理' in content_lower:
            return 'manual'
        elif '程序' in content_lower or '流程' in content_lower:
            return 'procedure'
        elif '记录' in content_lower or '表单' in content_lower:
            return 'record'
        
        # 默认返回manual类别
        return 'manual'
    
    def get_retriever(self, k: int = 5) -> Optional[Any]:
        """
        获取通用检索器
        
        Args:
            k: 返回的文档数量
            
        Returns:
            通用检索器，如果向量数据库未初始化则返回None
        """
        try:
            if self.vector_store is None:
                logger.warning("向量数据库未初始化，无法创建通用检索器")
                return None
            
            # 创建基础检索器
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            
            logger.info(f"创建通用检索器成功，k={k}")
            return retriever
            
        except Exception as e:
            logger.error(f"创建通用检索器失败: {e}")
            return None
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """获取向量数据库统计信息"""
        try:
            stats = {
                'total_vectors': 0,
                'vector_dimension': 0,
                'index_size_mb': 0,
                'last_updated': None
            }
            
            if self.vector_store is not None:
                stats['total_vectors'] = self.vector_store.index.ntotal
                stats['vector_dimension'] = self.vector_store.index.d
            
            # 获取索引文件大小
            index_path = os.path.join(self.vector_db_path, "faiss_index.faiss")
            if os.path.exists(index_path):
                file_size = os.path.getsize(index_path)
                stats['index_size_mb'] = round(file_size / (1024 * 1024), 2)
                stats['last_updated'] = datetime.fromtimestamp(
                    os.path.getmtime(index_path)
                ).isoformat()
            
            return stats
            
        except Exception as e:
            logger.error(f"获取向量数据库统计信息失败: {e}")
            return {}
    
    # Redis缓存相关方法
    def cache_vectorization_status(self, document_id: str, status: str):
        """缓存向量化状态"""
        try:
            redis_client = self.get_redis_client()
            if redis_client:
                key = f"vectorization_status:{document_id}"
                redis_client.setex(key, 3600, status)  # 缓存1小时
        except Exception as e:
            logger.warning(f"缓存向量化状态失败: {e}")
    
    def get_vectorization_status(self, document_id: str) -> Optional[str]:
        """获取向量化状态"""
        try:
            redis_client = self.get_redis_client()
            if redis_client:
                key = f"vectorization_status:{document_id}"
                status = redis_client.get(key)
                return status.decode() if status else None
        except Exception as e:
            logger.warning(f"获取向量化状态失败: {e}")
            return None
    
    def update_vectorization_progress(self, document_id: str, progress_data: dict):
        """更新向量化进度"""
        try:
            redis_client = self.get_redis_client()
            if redis_client:
                import json
                key = f"vectorization_progress:{document_id}"
                value = json.dumps(progress_data, ensure_ascii=False)
                redis_client.setex(key, 3600, value)  # 缓存1小时
        except Exception as e:
            logger.warning(f"更新向量化进度失败: {e}")
    
    def get_vectorization_progress(self, document_id: str) -> Optional[dict]:
        """获取向量化进度"""
        try:
            redis_client = self.get_redis_client()
            if redis_client:
                import json
                key = f"vectorization_progress:{document_id}"
                progress = redis_client.get(key)
                if progress:
                    return json.loads(progress.decode('utf-8'))
            return None
        except Exception as e:
            logger.warning(f"获取向量化进度失败: {e}")
            return None
    
    def cache_deleted_document(self, document_id: str):
        """缓存已删除的文档ID"""
        try:
            redis_client = self.get_redis_client()
            if redis_client:
                key = "deleted_documents"
                redis_client.sadd(key, document_id)
        except Exception as e:
            logger.warning(f"缓存删除文档失败: {e}")
    
    def get_deleted_documents(self) -> set:
        """获取已删除的文档ID列表"""
        try:
            redis_client = self.get_redis_client()
            if redis_client:
                key = "deleted_documents"
                deleted_docs = redis_client.smembers(key)
                return {doc.decode() for doc in deleted_docs}
            return set()
        except Exception as e:
            logger.warning(f"获取删除文档列表失败: {e}")
            return set()
    
    def clear_deleted_documents(self):
        """清除已删除文档的缓存"""
        try:
            redis_client = self.get_redis_client()
            if redis_client:
                key = "deleted_documents"
                redis_client.delete(key)
        except Exception as e:
            logger.warning(f"清除删除文档缓存失败: {e}")
    
    def cache_search_result(self, query: str, results: List[Dict[str, Any]]):
        """缓存搜索结果"""
        try:
            redis_client = self.get_redis_client()
            if redis_client:
                import json
                key = f"search_result:{hash(query)}"
                value = json.dumps(results, ensure_ascii=False)
                redis_client.setex(key, 300, value)  # 缓存5分钟
        except Exception as e:
            logger.warning(f"缓存搜索结果失败: {e}")
    
    def get_cached_search_result(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """获取缓存的搜索结果"""
        try:
            redis_client = self.get_redis_client()
            if redis_client:
                import json
                key = f"search_result:{hash(query)}"
                cached_result = redis_client.get(key)
                if cached_result:
                    return json.loads(cached_result.decode())
            return None
        except Exception as e:
            logger.warning(f"获取缓存搜索结果失败: {e}")
            return None

# 向量服务工具函数
def calculate_similarity(vector1: List[float], vector2: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    try:
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        # 计算余弦相似度
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        similarity = dot_product / (norm_v1 * norm_v2)
        return float(similarity)
        
    except Exception as e:
        logger.error(f"计算向量相似度失败: {e}")
        return 0.0

def normalize_vector(vector: List[float]) -> List[float]:
    """归一化向量"""
    try:
        v = np.array(vector)
        norm = np.linalg.norm(v)
        if norm == 0:
            return vector
        return (v / norm).tolist()
    except Exception as e:
        logger.error(f"向量归一化失败: {e}")
        return vector
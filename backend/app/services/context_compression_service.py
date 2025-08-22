from typing import List, Dict, Any, Optional, Tuple
import logging
import re
import numpy as np
from dataclasses import dataclass

from langchain.schema import Document
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.config import settings
from app.services.llm_service import LLMService
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)

@dataclass
class CompressionMetrics:
    """压缩指标"""
    original_length: int
    compressed_length: int
    compression_ratio: float
    relevance_score: float
    information_density: float
    quality_score: float

class ContextCompressionService:
    """
    上下文压缩服务 - 过滤无关信息并保留最相关的文档片段
    """
    
    def __init__(self):
        self.embeddings = None
        self.llm_service = None
        self.tfidf_vectorizer = None
        self.vector_service = None
        
        self._initialize_services()
        
        # 压缩配置
        self.config = {
            'max_context_length': 4000,
            'target_compression_ratio': 0.6,
            'relevance_threshold': 0.3,
            'similarity_threshold': 0.7,
            'preserve_structure': True,
            'use_llm_compression': True,
            'use_embedding_filter': True,
            'use_tfidf_filter': True,
            'sentence_window_size': 2,
            'min_sentence_length': 20
        }
        
        logger.info("上下文压缩服务初始化完成")
    
    def _initialize_services(self):
        """初始化服务"""
        try:
            # 嵌入模型改为VectorService
            self.vector_service = VectorService.get_instance()
            logger.info("嵌入模型(VectorService)初始化成功")
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            self.vector_service = None
        
        try:
            # LLM服务
            self.llm_service = LLMService()
            logger.info("LLM服务初始化成功")
        except Exception as e:
            logger.error(f"LLM服务初始化失败: {e}")
        
        try:
            # TF-IDF向量化器
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=None,  # 保留中文停用词处理
                ngram_range=(1, 2)
            )
            logger.info("TF-IDF向量化器初始化成功")
        except Exception as e:
            logger.error(f"TF-IDF向量化器初始化失败: {e}")
    
    def compress_context(self, 
                        documents: List[Document], 
                        query: str,
                        compression_method: str = 'hybrid') -> Tuple[List[Document], CompressionMetrics]:
        """压缩上下文"""
        try:
            logger.info(f"开始上下文压缩: {len(documents)} 个文档, 方法: {compression_method}")
            
            if not documents:
                return [], CompressionMetrics(0, 0, 0.0, 0.0, 0.0, 0.0)
            
            original_length = sum(len(doc.page_content) for doc in documents)
            
            # 选择压缩方法
            if compression_method == 'hybrid':
                compressed_docs = self._hybrid_compression(documents, query)
            elif compression_method == 'embedding_based':
                compressed_docs = self._embedding_based_compression(documents, query)
            elif compression_method == 'llm_based':
                compressed_docs = self._llm_based_compression(documents, query)
            elif compression_method == 'tfidf_based':
                compressed_docs = self._tfidf_based_compression(documents, query)
            elif compression_method == 'sentence_level':
                compressed_docs = self._sentence_level_compression(documents, query)
            else:
                compressed_docs = self._basic_compression(documents, query)
            
            # 计算压缩指标
            compressed_length = sum(len(doc.page_content) for doc in compressed_docs)
            metrics = self._calculate_compression_metrics(
                original_length, compressed_length, compressed_docs, query
            )
            
            logger.info(f"上下文压缩完成: {len(compressed_docs)} 个文档, 压缩比: {metrics.compression_ratio:.2f}")
            return compressed_docs, metrics
            
        except Exception as e:
            logger.error(f"上下文压缩失败: {e}")
            return documents, CompressionMetrics(
                original_length=sum(len(doc.page_content) for doc in documents),
                compressed_length=sum(len(doc.page_content) for doc in documents),
                compression_ratio=1.0,
                relevance_score=0.0,
                information_density=0.0,
                quality_score=0.0
            )
    
    def _hybrid_compression(self, documents: List[Document], query: str) -> List[Document]:
        """混合压缩方法"""
        try:
            # 1. 首先使用嵌入过滤
            if self.config['use_embedding_filter'] and self.vector_service:
                documents = self._embedding_based_compression(documents, query)
            
            # 2. 然后使用TF-IDF过滤
            if self.config['use_tfidf_filter'] and len(documents) > 3:
                documents = self._tfidf_based_compression(documents, query)
            
            # 3. 最后使用句子级压缩
            if len(documents) > 2:
                documents = self._sentence_level_compression(documents, query)
            
            # 4. 如果仍然太长，使用LLM压缩
            total_length = sum(len(doc.page_content) for doc in documents)
            if (total_length > self.config['max_context_length'] and 
                self.config['use_llm_compression'] and self.llm_service):
                documents = self._llm_based_compression(documents, query)
            
            return documents
            
        except Exception as e:
            logger.error(f"混合压缩失败: {e}")
            return documents
    
    def _embedding_based_compression(self, documents: List[Document], query: str) -> List[Document]:
        """基于嵌入的压缩"""
        try:
            if not self.vector_service:
                return documents
            
            # 计算查询嵌入
            query_embedding = np.array(self.vector_service.create_single_embedding(query), dtype=float)
            
            # 计算文档嵌入和相似度
            doc_texts = [doc.page_content for doc in documents]
            doc_embeddings = np.array(self.vector_service.create_embeddings(doc_texts), dtype=float)
            
            # 计算相似度分数（向量已归一化，使用点积即余弦相似度）
            similarities = (doc_embeddings @ query_embedding).tolist()
            
            # 根据相似度过滤文档
            filtered_docs = []
            for i, (doc, similarity) in enumerate(zip(documents, similarities)):
                if similarity >= self.config['relevance_threshold']:
                    # 添加相似度信息到元数据
                    doc.metadata['relevance_score'] = float(similarity)
                    doc.metadata['compression_method'] = 'embedding_based'
                    filtered_docs.append(doc)
            
            # 按相似度排序
            filtered_docs.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
            
            # 限制文档数量
            max_docs = min(5, len(filtered_docs))
            return filtered_docs[:max_docs]
            
        except Exception as e:
            logger.error(f"基于嵌入的压缩失败: {e}")
            return documents
    
    def _llm_based_compression(self, documents: List[Document], query: str) -> List[Document]:
        """基于LLM的压缩"""
        try:
            if not self.llm_service:
                return documents
            
            compressed_docs = []
            
            for doc in documents:
                # 构建压缩提示
                compression_prompt = f"""
请根据用户问题压缩以下文档内容，保留最相关的信息：

用户问题：{query}

文档内容：
{doc.page_content}

要求：
1. 保留与问题直接相关的信息
2. 删除无关的细节和重复内容
3. 保持信息的完整性和准确性
4. 压缩后的内容应该简洁明了
5. 如果整个文档都不相关，返回"无相关内容"

压缩后的内容：
"""
                
                try:
                    compressed_content = self.llm_service.generate_response(compression_prompt)
                    
                    if (compressed_content and 
                        compressed_content.strip() != "无相关内容" and
                        len(compressed_content.strip()) >= self.config['min_sentence_length']):
                        
                        compressed_doc = Document(
                            page_content=compressed_content.strip(),
                            metadata={**doc.metadata, 
                                    'compression_method': 'llm_based',
                                    'original_length': len(doc.page_content),
                                    'compressed_length': len(compressed_content)}
                        )
                        compressed_docs.append(compressed_doc)
                        
                except Exception as e:
                    logger.error(f"LLM压缩单个文档失败: {e}")
                    # 降级到原文档
                    compressed_docs.append(doc)
            
            return compressed_docs
            
        except Exception as e:
            logger.error(f"基于LLM的压缩失败: {e}")
            return documents
    
    def _tfidf_based_compression(self, documents: List[Document], query: str) -> List[Document]:
        """基于TF-IDF的压缩"""
        try:
            if not self.tfidf_vectorizer or len(documents) < 2:
                return documents
            
            # 准备文本数据
            doc_texts = [doc.page_content for doc in documents]
            all_texts = [query] + doc_texts
            
            # 计算TF-IDF矩阵
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            # 查询向量是第一个
            query_vector = tfidf_matrix[0]
            doc_vectors = tfidf_matrix[1:]
            
            # 计算相似度
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            
            # 根据相似度过滤和排序
            doc_similarities = list(zip(documents, similarities))
            doc_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 过滤低相关性文档
            filtered_docs = []
            for doc, similarity in doc_similarities:
                if similarity >= self.config['relevance_threshold']:
                    doc.metadata['tfidf_score'] = similarity
                    doc.metadata['compression_method'] = 'tfidf_based'
                    filtered_docs.append(doc)
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"基于TF-IDF的压缩失败: {e}")
            return documents
    
    def _sentence_level_compression(self, documents: List[Document], query: str) -> List[Document]:
        """句子级压缩"""
        try:
            if not self.vector_service:
                return documents
            
            # 计算查询嵌入（向量已归一化，使用点积即余弦相似度）
            query_embedding = np.array(self.vector_service.create_single_embedding(query), dtype=float)
            
            compressed_docs = []
            
            for doc in documents:
                # 分割句子
                sentences = self._split_into_sentences(doc.page_content)
                if len(sentences) <= 2:
                    compressed_docs.append(doc)
                    continue
                
                # 计算句子嵌入（归一化）
                sentence_embeddings = np.array(self.vector_service.create_embeddings(sentences), dtype=float)
                
                # 计算句子与查询的相似度（点积=余弦相似度）
                sentence_scores = []
                for sentence_embedding in sentence_embeddings:
                    similarity = float(np.dot(query_embedding, sentence_embedding))
                    sentence_scores.append(similarity)
                
                # 选择高相关性句子
                selected_sentences = []
                for i, (sentence, score) in enumerate(zip(sentences, sentence_scores)):
                    if score >= self.config['relevance_threshold']:
                        selected_sentences.append((sentence, score, i))
                
                # 如果没有高相关性句子，保留分数最高的几个
                if not selected_sentences:
                    sorted_sentences = sorted(
                        zip(sentences, sentence_scores, range(len(sentences))),
                        key=lambda x: x[1], reverse=True
                    )
                    selected_sentences = sorted_sentences[:2]
                
                # 按原始顺序重新排列
                selected_sentences.sort(key=lambda x: x[2])
                
                # 构建压缩后的内容
                compressed_content = ' '.join([s[0] for s in selected_sentences])
                
                if len(compressed_content.strip()) >= self.config['min_sentence_length']:
                    compressed_doc = Document(
                        page_content=compressed_content,
                        metadata={**doc.metadata,
                                'compression_method': 'sentence_level',
                                'selected_sentences': len(selected_sentences),
                                'total_sentences': len(sentences)}
                    )
                    compressed_docs.append(compressed_doc)
            
            return compressed_docs
            
        except Exception as e:
            logger.error(f"句子级压缩失败: {e}")
            return documents
    
    def _basic_compression(self, documents: List[Document], query: str) -> List[Document]:
        """基础压缩方法"""
        try:
            # 简单的长度限制
            total_length = 0
            compressed_docs = []
            
            for doc in documents:
                if total_length + len(doc.page_content) <= self.config['max_context_length']:
                    doc.metadata['compression_method'] = 'basic'
                    compressed_docs.append(doc)
                    total_length += len(doc.page_content)
                else:
                    # 截断文档
                    remaining_length = self.config['max_context_length'] - total_length
                    if remaining_length > self.config['min_sentence_length']:
                        truncated_content = doc.page_content[:remaining_length]
                        # 尝试在句子边界截断
                        last_sentence_end = max(
                            truncated_content.rfind('。'),
                            truncated_content.rfind('！'),
                            truncated_content.rfind('？')
                        )
                        if last_sentence_end > remaining_length * 0.7:
                            truncated_content = truncated_content[:last_sentence_end + 1]
                        
                        truncated_doc = Document(
                            page_content=truncated_content,
                            metadata={**doc.metadata,
                                    'compression_method': 'basic_truncated',
                                    'truncated': True}
                        )
                        compressed_docs.append(truncated_doc)
                    break
            
            return compressed_docs
            
        except Exception as e:
            logger.error(f"基础压缩失败: {e}")
            return documents
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """分割句子"""
        try:
            # 使用正则表达式分割句子
            sentence_pattern = r'[。！？；：]\s*'
            sentences = re.split(sentence_pattern, text)
            
            # 清理和过滤句子
            cleaned_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) >= self.config['min_sentence_length']:
                    cleaned_sentences.append(sentence)
            
            return cleaned_sentences
            
        except Exception as e:
            logger.error(f"句子分割失败: {e}")
            return [text]
    
    def _calculate_compression_metrics(self, 
                                     original_length: int, 
                                     compressed_length: int,
                                     compressed_docs: List[Document],
                                     query: str) -> CompressionMetrics:
        """计算压缩指标"""
        try:
            # 压缩比
            compression_ratio = compressed_length / original_length if original_length > 0 else 0.0
            
            # 相关性分数
            relevance_score = self._calculate_relevance_score(compressed_docs, query)
            
            # 信息密度
            information_density = self._calculate_information_density(compressed_docs)
            
            # 质量分数
            quality_score = self._calculate_quality_score(
                compression_ratio, relevance_score, information_density
            )
            
            return CompressionMetrics(
                original_length=original_length,
                compressed_length=compressed_length,
                compression_ratio=compression_ratio,
                relevance_score=relevance_score,
                information_density=information_density,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"计算压缩指标失败: {e}")
            return CompressionMetrics(original_length, compressed_length, 1.0, 0.0, 0.0, 0.0)
    
    def _calculate_relevance_score(self, documents: List[Document], query: str) -> float:
        """计算相关性分数"""
        try:
            if not documents or not self.vector_service:
                return 0.0
            
            query_embedding = np.array(self.vector_service.create_single_embedding(query), dtype=float)
            doc_texts = [doc.page_content for doc in documents]
            doc_embeddings = np.array(self.vector_service.create_embeddings(doc_texts), dtype=float)
            
            similarities = []
            for doc_embedding in doc_embeddings:
                similarity = float(np.dot(query_embedding, doc_embedding))
                similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"计算相关性分数失败: {e}")
            return 0.0
    
    def _calculate_information_density(self, documents: List[Document]) -> float:
        """计算信息密度"""
        try:
            if not documents:
                return 0.0
            
            total_chars = sum(len(doc.page_content) for doc in documents)
            total_words = sum(len(doc.page_content.split()) for doc in documents)
            
            # 简单的信息密度指标：词汇多样性
            all_words = []
            for doc in documents:
                all_words.extend(doc.page_content.split())
            
            unique_words = len(set(all_words))
            total_words = len(all_words)
            
            diversity = unique_words / total_words if total_words > 0 else 0.0
            
            # 结合长度因子
            length_factor = min(1.0, total_chars / 1000)  # 标准化到1000字符
            
            return diversity * length_factor
            
        except Exception as e:
            logger.error(f"计算信息密度失败: {e}")
            return 0.0
    
    def _calculate_quality_score(self, 
                               compression_ratio: float, 
                               relevance_score: float,
                               information_density: float) -> float:
        """计算质量分数"""
        try:
            # 压缩比分数（目标压缩比附近得分最高）
            target_ratio = self.config['target_compression_ratio']
            ratio_score = 1.0 - abs(compression_ratio - target_ratio) / target_ratio
            ratio_score = max(0.0, ratio_score)
            
            # 综合质量分数
            quality_score = (
                relevance_score * 0.4 +
                information_density * 0.3 +
                ratio_score * 0.3
            )
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.error(f"计算质量分数失败: {e}")
            return 0.0
    
    def create_compression_pipeline(self, similarity_threshold: float = 0.3) -> DocumentCompressorPipeline:
        """创建压缩管道"""
        try:
            if not self.vector_service:
                logger.warning("嵌入模型不可用，无法创建压缩管道")
                return None
            
            # 创建嵌入过滤器（使用VectorService的LangChain兼容包装器）
            embeddings_filter = EmbeddingsFilter(
                embeddings=self.vector_service._langchain_embeddings,
                similarity_threshold=similarity_threshold
            )
            
            # 创建文本分割器
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=50
            )
            
            # 创建压缩管道
            compressor = DocumentCompressorPipeline(
                transformers=[splitter, embeddings_filter]
            )
            
            return compressor
            
        except Exception as e:
            logger.error(f"创建压缩管道失败: {e}")
            return None
    
    def batch_compress_context(self, 
                             document_batches: List[List[Document]], 
                             queries: List[str],
                             compression_method: str = 'hybrid') -> List[Tuple[List[Document], CompressionMetrics]]:
        """批量压缩上下文"""
        try:
            logger.info(f"开始批量上下文压缩: {len(document_batches)} 批文档")
            
            results = []
            for i, (docs, query) in enumerate(zip(document_batches, queries)):
                compressed_docs, metrics = self.compress_context(docs, query, compression_method)
                results.append((compressed_docs, metrics))
            
            logger.info(f"批量上下文压缩完成: {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"批量上下文压缩失败: {e}")
            return []
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        try:
            self.config.update(new_config)
            logger.info(f"上下文压缩配置已更新: {new_config}")
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
    
    def get_compression_stats(self, metrics_list: List[CompressionMetrics]) -> Dict[str, Any]:
        """获取压缩统计信息"""
        try:
            if not metrics_list:
                return {'total_compressions': 0}
            
            compression_ratios = [m.compression_ratio for m in metrics_list]
            relevance_scores = [m.relevance_score for m in metrics_list]
            quality_scores = [m.quality_score for m in metrics_list]
            
            stats = {
                'total_compressions': len(metrics_list),
                'avg_compression_ratio': np.mean(compression_ratios),
                'avg_relevance_score': np.mean(relevance_scores),
                'avg_quality_score': np.mean(quality_scores),
                'min_compression_ratio': min(compression_ratios),
                'max_compression_ratio': max(compression_ratios),
                'high_quality_ratio': len([q for q in quality_scores if q >= 0.7]) / len(quality_scores)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取压缩统计失败: {e}")
            return {'error': str(e)}
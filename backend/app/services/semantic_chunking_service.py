from typing import List, Dict, Any, Optional, Tuple
import logging
import re
import numpy as np
from dataclasses import dataclass

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from app.core.config import settings
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)

@dataclass
class ChunkMetrics:
    """分块质量指标"""
    coherence_score: float  # 连贯性分数
    completeness_score: float  # 完整性分数
    size_score: float  # 大小适宜性分数
    overlap_score: float  # 重叠质量分数
    overall_score: float  # 总体分数

class SemanticChunkingService:
    """
    语义分块服务 - 基于语义相似度的智能文档分块
    """
    
    def __init__(self):
        self.vector_service = None
        self._initialize_embeddings()
        
        # 分块配置
        self.config = {
            'min_chunk_size': 200,
            'max_chunk_size': 1000,
            'target_chunk_size': 600,
            'overlap_ratio': 0.15,
            'similarity_threshold': 0.7,
            'sentence_window_size': 3,
            'use_semantic_boundaries': True,
            'preserve_structure': True
        }
        
        # 句子分割器
        self.sentence_splitter = self._create_sentence_splitter()
        
        logger.info("语义分块服务初始化完成")
    
    def _initialize_embeddings(self):
        """初始化嵌入模型"""
        try:
            self.vector_service = VectorService.get_instance()
            logger.info("嵌入模型初始化成功")
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            self.vector_service = None
    
    def _create_sentence_splitter(self):
        """创建句子分割器"""
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", ";", ":", "，", " "],
            chunk_size=100,
            chunk_overlap=0,
            length_function=len,
            keep_separator=True
        )
    
    def semantic_chunk_document(self, 
                              document: Document, 
                              chunk_method: str = 'adaptive') -> List[Document]:
        """对文档进行语义分块"""
        try:
            logger.info(f"开始语义分块: {document.metadata.get('title', 'Unknown')}")
            
            text = document.page_content
            if not text or len(text) < self.config['min_chunk_size']:
                return [document]
            
            # 选择分块方法
            if chunk_method == 'adaptive':
                chunks = self._adaptive_semantic_chunking(text, document.metadata)
            elif chunk_method == 'similarity_based':
                chunks = self._similarity_based_chunking(text, document.metadata)
            elif chunk_method == 'structure_aware':
                chunks = self._structure_aware_chunking(text, document.metadata)
            else:
                chunks = self._basic_semantic_chunking(text, document.metadata)
            
            logger.info(f"语义分块完成: {len(chunks)} 个分块")
            return chunks
            
        except Exception as e:
            logger.error(f"语义分块失败: {e}")
            # 降级到基础分块
            return self._fallback_chunking(document)
    
    def _adaptive_semantic_chunking(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """自适应语义分块"""
        try:
            # 1. 预处理文本
            sentences = self._split_into_sentences(text)
            if len(sentences) < 3:
                return [Document(page_content=text, metadata=metadata)]
            
            # 2. 计算句子嵌入
            if not self.vector_service:
                return self._fallback_chunking_text(text, metadata)
            
            sentence_embeddings = self._get_sentence_embeddings(sentences)
            if sentence_embeddings is None:
                return self._fallback_chunking_text(text, metadata)
            
            # 3. 计算语义边界
            boundaries = self._find_semantic_boundaries(sentence_embeddings)
            
            # 4. 基于边界创建分块
            chunks = self._create_chunks_from_boundaries(sentences, boundaries, metadata)
            
            # 5. 优化分块大小
            optimized_chunks = self._optimize_chunk_sizes(chunks)
            
            # 6. 添加重叠窗口
            final_chunks = self._add_overlap_windows(optimized_chunks)
            
            return final_chunks
            
        except Exception as e:
            logger.error(f"自适应语义分块失败: {e}")
            return self._fallback_chunking_text(text, metadata)
    
    def _similarity_based_chunking(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """基于相似度的分块"""
        try:
            sentences = self._split_into_sentences(text)
            if len(sentences) < 3:
                return [Document(page_content=text, metadata=metadata)]
            
            if not self.vector_service:
                return self._fallback_chunking_text(text, metadata)
            
            # 计算句子嵌入
            embeddings = self._get_sentence_embeddings(sentences)
            if embeddings is None:
                return self._fallback_chunking_text(text, metadata)
            
            # 使用滑动窗口计算相似度
            chunks = []
            current_chunk = [sentences[0]]
            
            for i in range(1, len(sentences)):
                # 计算当前句子与分块的相似度
                chunk_embedding = np.mean([embeddings[j] for j in range(len(current_chunk))], axis=0)
                sentence_embedding = embeddings[i]
                
                similarity = cosine_similarity(
                    [chunk_embedding], [sentence_embedding]
                )[0][0]
                
                # 检查分块大小
                current_text = ' '.join(current_chunk)
                
                if (similarity >= self.config['similarity_threshold'] and 
                    len(current_text) < self.config['max_chunk_size']):
                    current_chunk.append(sentences[i])
                else:
                    # 创建新分块
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        if len(chunk_text) >= self.config['min_chunk_size']:
                            chunks.append(Document(
                                page_content=chunk_text,
                                metadata={**metadata, 'chunk_method': 'similarity_based'}
                            ))
                    
                    current_chunk = [sentences[i]]
            
            # 处理最后一个分块
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.config['min_chunk_size']:
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata={**metadata, 'chunk_method': 'similarity_based'}
                    ))
                elif chunks:  # 合并到上一个分块
                    chunks[-1].page_content += ' ' + chunk_text
            
            return chunks if chunks else [Document(page_content=text, metadata=metadata)]
            
        except Exception as e:
            logger.error(f"基于相似度的分块失败: {e}")
            return self._fallback_chunking_text(text, metadata)
    
    def _structure_aware_chunking(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """结构感知分块"""
        try:
            # 识别文档结构
            structure_markers = self._identify_structure_markers(text)
            
            if not structure_markers:
                return self._adaptive_semantic_chunking(text, metadata)
            
            chunks = []
            current_pos = 0
            
            for marker in structure_markers:
                start_pos = marker['start']
                end_pos = marker['end']
                
                # 处理标记前的内容
                if start_pos > current_pos:
                    pre_content = text[current_pos:start_pos].strip()
                    if len(pre_content) >= self.config['min_chunk_size']:
                        pre_chunks = self._adaptive_semantic_chunking(pre_content, metadata)
                        chunks.extend(pre_chunks)
                
                # 处理标记内容
                marker_content = text[start_pos:end_pos].strip()
                if len(marker_content) >= self.config['min_chunk_size']:
                    if len(marker_content) <= self.config['max_chunk_size']:
                        chunks.append(Document(
                            page_content=marker_content,
                            metadata={**metadata, 
                                    'structure_type': marker['type'],
                                    'chunk_method': 'structure_aware'}
                        ))
                    else:
                        # 大段落需要进一步分块
                        sub_chunks = self._adaptive_semantic_chunking(marker_content, metadata)
                        for chunk in sub_chunks:
                            chunk.metadata['structure_type'] = marker['type']
                            chunk.metadata['chunk_method'] = 'structure_aware'
                        chunks.extend(sub_chunks)
                
                current_pos = end_pos
            
            # 处理剩余内容
            if current_pos < len(text):
                remaining_content = text[current_pos:].strip()
                if len(remaining_content) >= self.config['min_chunk_size']:
                    remaining_chunks = self._adaptive_semantic_chunking(remaining_content, metadata)
                    chunks.extend(remaining_chunks)
            
            return chunks if chunks else [Document(page_content=text, metadata=metadata)]
            
        except Exception as e:
            logger.error(f"结构感知分块失败: {e}")
            return self._adaptive_semantic_chunking(text, metadata)
    
    def _basic_semantic_chunking(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """基础语义分块"""
        try:
            # 使用改进的递归分割器
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "。", "！", "？", ";", ":", "，", " "],
                chunk_size=self.config['target_chunk_size'],
                chunk_overlap=int(self.config['target_chunk_size'] * self.config['overlap_ratio']),
                length_function=len,
                keep_separator=True
            )
            
            chunks = splitter.split_text(text)
            
            documents = []
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) >= self.config['min_chunk_size']:
                    documents.append(Document(
                        page_content=chunk.strip(),
                        metadata={**metadata, 
                                'chunk_index': i,
                                'chunk_method': 'basic_semantic'}
                    ))
            
            return documents if documents else [Document(page_content=text, metadata=metadata)]
            
        except Exception as e:
            logger.error(f"基础语义分块失败: {e}")
            return [Document(page_content=text, metadata=metadata)]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割为句子"""
        try:
            # 使用正则表达式分割句子
            sentence_pattern = r'[。！？；：]\s*'
            sentences = re.split(sentence_pattern, text)
            
            # 清理和过滤句子
            cleaned_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:  # 过滤太短的句子
                    cleaned_sentences.append(sentence)
            
            return cleaned_sentences
            
        except Exception as e:
            logger.error(f"句子分割失败: {e}")
            return [text]
    
    def _get_sentence_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        """获取句子嵌入"""
        try:
            if not self.vector_service:
                return None
            
            # 批量计算嵌入
            embeddings = self.vector_service.create_embeddings(sentences)
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"计算句子嵌入失败: {e}")
            return None
    
    def _find_semantic_boundaries(self, embeddings: np.ndarray) -> List[int]:
        """寻找语义边界"""
        try:
            boundaries = [0]  # 开始位置
            
            # 计算相邻句子的相似度
            for i in range(len(embeddings) - 1):
                similarity = cosine_similarity(
                    [embeddings[i]], [embeddings[i + 1]]
                )[0][0]
                
                # 如果相似度低于阈值，标记为边界
                if similarity < self.config['similarity_threshold']:
                    boundaries.append(i + 1)
            
            boundaries.append(len(embeddings))  # 结束位置
            
            return boundaries
            
        except Exception as e:
            logger.error(f"寻找语义边界失败: {e}")
            return [0, len(embeddings)]
    
    def _create_chunks_from_boundaries(self, 
                                     sentences: List[str], 
                                     boundaries: List[int],
                                     metadata: Dict[str, Any]) -> List[Document]:
        """基于边界创建分块"""
        try:
            chunks = []
            
            for i in range(len(boundaries) - 1):
                start_idx = boundaries[i]
                end_idx = boundaries[i + 1]
                
                chunk_sentences = sentences[start_idx:end_idx]
                chunk_text = ' '.join(chunk_sentences)
                
                if len(chunk_text.strip()) >= self.config['min_chunk_size']:
                    chunks.append(Document(
                        page_content=chunk_text.strip(),
                        metadata={**metadata, 
                                'chunk_method': 'adaptive_semantic',
                                'boundary_start': start_idx,
                                'boundary_end': end_idx}
                    ))
            
            return chunks
            
        except Exception as e:
            logger.error(f"基于边界创建分块失败: {e}")
            return []
    
    def _optimize_chunk_sizes(self, chunks: List[Document]) -> List[Document]:
        """优化分块大小"""
        try:
            optimized_chunks = []
            i = 0
            
            while i < len(chunks):
                current_chunk = chunks[i]
                current_length = len(current_chunk.page_content)
                
                # 如果分块太小，尝试合并
                if (current_length < self.config['min_chunk_size'] and 
                    i + 1 < len(chunks)):
                    
                    next_chunk = chunks[i + 1]
                    combined_length = current_length + len(next_chunk.page_content)
                    
                    if combined_length <= self.config['max_chunk_size']:
                        # 合并分块
                        combined_content = current_chunk.page_content + ' ' + next_chunk.page_content
                        combined_metadata = current_chunk.metadata.copy()
                        combined_metadata['merged'] = True
                        
                        optimized_chunks.append(Document(
                            page_content=combined_content,
                            metadata=combined_metadata
                        ))
                        i += 2  # 跳过下一个分块
                        continue
                
                # 如果分块太大，尝试分割
                elif current_length > self.config['max_chunk_size']:
                    sub_chunks = self._split_large_chunk(current_chunk)
                    optimized_chunks.extend(sub_chunks)
                else:
                    optimized_chunks.append(current_chunk)
                
                i += 1
            
            return optimized_chunks
            
        except Exception as e:
            logger.error(f"优化分块大小失败: {e}")
            return chunks
    
    def _split_large_chunk(self, chunk: Document) -> List[Document]:
        """分割大分块"""
        try:
            text = chunk.page_content
            metadata = chunk.metadata
            
            # 使用句子边界分割
            sentences = self._split_into_sentences(text)
            
            sub_chunks = []
            current_sentences = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                if (current_length + sentence_length <= self.config['target_chunk_size'] or
                    not current_sentences):
                    current_sentences.append(sentence)
                    current_length += sentence_length
                else:
                    # 创建子分块
                    if current_sentences:
                        sub_chunk_text = ' '.join(current_sentences)
                        sub_chunks.append(Document(
                            page_content=sub_chunk_text,
                            metadata={**metadata, 'split_from_large': True}
                        ))
                    
                    current_sentences = [sentence]
                    current_length = sentence_length
            
            # 处理最后一个子分块
            if current_sentences:
                sub_chunk_text = ' '.join(current_sentences)
                sub_chunks.append(Document(
                    page_content=sub_chunk_text,
                    metadata={**metadata, 'split_from_large': True}
                ))
            
            return sub_chunks if sub_chunks else [chunk]
            
        except Exception as e:
            logger.error(f"分割大分块失败: {e}")
            return [chunk]
    
    def _add_overlap_windows(self, chunks: List[Document]) -> List[Document]:
        """添加重叠窗口"""
        try:
            if len(chunks) <= 1:
                return chunks
            
            overlap_size = int(self.config['target_chunk_size'] * self.config['overlap_ratio'])
            
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i]
                next_chunk = chunks[i + 1]
                
                # 从当前分块末尾提取重叠内容
                current_text = current_chunk.page_content
                if len(current_text) > overlap_size:
                    overlap_content = current_text[-overlap_size:]
                    
                    # 添加到下一个分块的开头
                    next_chunk.page_content = overlap_content + ' ' + next_chunk.page_content
                    
                    # 更新元数据
                    next_chunk.metadata['has_overlap'] = True
                    next_chunk.metadata['overlap_size'] = len(overlap_content)
            
            return chunks
            
        except Exception as e:
            logger.error(f"添加重叠窗口失败: {e}")
            return chunks
    
    def _identify_structure_markers(self, text: str) -> List[Dict[str, Any]]:
        """识别文档结构标记"""
        try:
            markers = []
            
            # 标题模式
            title_patterns = [
                (r'^#{1,6}\s+(.+)$', 'heading'),  # Markdown标题
                (r'^\d+\.\s+(.+)$', 'numbered_section'),  # 编号章节
                (r'^[一二三四五六七八九十]+[、.]\s*(.+)$', 'chinese_numbered'),  # 中文编号
                (r'^\*\s+(.+)$', 'bullet_point'),  # 项目符号
                (r'^-\s+(.+)$', 'dash_point'),  # 破折号
            ]
            
            lines = text.split('\n')
            current_pos = 0
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    current_pos += 1
                    continue
                
                for pattern, marker_type in title_patterns:
                    if re.match(pattern, line, re.MULTILINE):
                        # 找到结构标记的结束位置
                        end_pos = self._find_section_end(lines, i)
                        
                        markers.append({
                            'type': marker_type,
                            'start': current_pos,
                            'end': current_pos + len('\n'.join(lines[i:end_pos])),
                            'title': line,
                            'line_number': i
                        })
                        break
                
                current_pos += len(line) + 1  # +1 for newline
            
            return markers
            
        except Exception as e:
            logger.error(f"识别文档结构失败: {e}")
            return []
    
    def _find_section_end(self, lines: List[str], start_idx: int) -> int:
        """找到章节结束位置"""
        try:
            for i in range(start_idx + 1, len(lines)):
                line = lines[i].strip()
                
                # 遇到新的标题标记
                if (re.match(r'^#{1,6}\s+', line) or
                    re.match(r'^\d+\.\s+', line) or
                    re.match(r'^[一二三四五六七八九十]+[、.]\s*', line)):
                    return i
            
            return len(lines)
            
        except Exception as e:
            logger.error(f"找到章节结束位置失败: {e}")
            return len(lines)
    
    def _fallback_chunking(self, document: Document) -> List[Document]:
        """降级分块方法"""
        try:
            return self._basic_semantic_chunking(document.page_content, document.metadata)
        except Exception as e:
            logger.error(f"降级分块失败: {e}")
            return [document]
    
    def _fallback_chunking_text(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """降级分块方法（文本）"""
        try:
            return self._basic_semantic_chunking(text, metadata)
        except Exception as e:
            logger.error(f"降级分块失败: {e}")
            return [Document(page_content=text, metadata=metadata)]
    
    def evaluate_chunk_quality(self, chunks: List[Document]) -> List[ChunkMetrics]:
        """评估分块质量"""
        try:
            metrics = []
            
            for i, chunk in enumerate(chunks):
                # 计算各项指标
                coherence = self._calculate_coherence(chunk)
                completeness = self._calculate_completeness(chunk, chunks)
                size_score = self._calculate_size_score(chunk)
                overlap_score = self._calculate_overlap_score(chunk, chunks, i)
                
                # 计算总体分数
                overall = (coherence * 0.3 + completeness * 0.25 + 
                          size_score * 0.25 + overlap_score * 0.2)
                
                metrics.append(ChunkMetrics(
                    coherence_score=coherence,
                    completeness_score=completeness,
                    size_score=size_score,
                    overlap_score=overlap_score,
                    overall_score=overall
                ))
            
            return metrics
            
        except Exception as e:
            logger.error(f"评估分块质量失败: {e}")
            return []
    
    def _calculate_coherence(self, chunk: Document) -> float:
        """计算连贯性分数"""
        try:
            text = chunk.page_content
            sentences = self._split_into_sentences(text)
            
            if len(sentences) < 2:
                return 1.0
            
            if not self.vector_service:
                return 0.5  # 默认分数
            
            # 计算句子间的平均相似度
            embeddings = self._get_sentence_embeddings(sentences)
            if embeddings is None:
                return 0.5
            
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                similarities.append(sim)
            
            return np.mean(similarities) if similarities else 0.5
            
        except Exception as e:
            logger.error(f"计算连贯性失败: {e}")
            return 0.5
    
    def _calculate_completeness(self, chunk: Document, all_chunks: List[Document]) -> float:
        """计算完整性分数"""
        try:
            text = chunk.page_content
            
            # 检查是否以完整句子结束
            ends_complete = text.rstrip().endswith(('。', '！', '？', '；', '：'))
            
            # 检查是否以完整句子开始
            starts_complete = not text.lstrip().startswith(('，', '、', '；', '：'))
            
            completeness = 0.0
            if ends_complete:
                completeness += 0.5
            if starts_complete:
                completeness += 0.5
            
            return completeness
            
        except Exception as e:
            logger.error(f"计算完整性失败: {e}")
            return 0.5
    
    def _calculate_size_score(self, chunk: Document) -> float:
        """计算大小适宜性分数"""
        try:
            length = len(chunk.page_content)
            target = self.config['target_chunk_size']
            min_size = self.config['min_chunk_size']
            max_size = self.config['max_chunk_size']
            
            if length < min_size:
                return length / min_size * 0.5
            elif length > max_size:
                return max(0.0, 1.0 - (length - max_size) / max_size)
            else:
                # 在合理范围内，越接近目标大小分数越高
                distance = abs(length - target) / target
                return max(0.0, 1.0 - distance)
            
        except Exception as e:
            logger.error(f"计算大小分数失败: {e}")
            return 0.5
    
    def _calculate_overlap_score(self, chunk: Document, all_chunks: List[Document], index: int) -> float:
        """计算重叠质量分数"""
        try:
            if index == 0 or index >= len(all_chunks) - 1:
                return 1.0  # 首尾分块不需要重叠
            
            has_overlap = chunk.metadata.get('has_overlap', False)
            if not has_overlap:
                return 0.5  # 没有重叠
            
            overlap_size = chunk.metadata.get('overlap_size', 0)
            target_overlap = int(self.config['target_chunk_size'] * self.config['overlap_ratio'])
            
            if overlap_size == 0:
                return 0.0
            
            # 计算重叠大小的适宜性
            ratio = overlap_size / target_overlap
            if 0.5 <= ratio <= 1.5:  # 合理范围
                return 1.0
            else:
                return max(0.0, 1.0 - abs(ratio - 1.0))
            
        except Exception as e:
            logger.error(f"计算重叠分数失败: {e}")
            return 0.5
    
    def batch_semantic_chunk(self, documents: List[Document], chunk_method: str = 'adaptive') -> List[Document]:
        """批量语义分块"""
        try:
            logger.info(f"开始批量语义分块: {len(documents)} 个文档")
            
            all_chunks = []
            for doc in documents:
                chunks = self.semantic_chunk_document(doc, chunk_method)
                all_chunks.extend(chunks)
            
            logger.info(f"批量语义分块完成: {len(all_chunks)} 个分块")
            return all_chunks
            
        except Exception as e:
            logger.error(f"批量语义分块失败: {e}")
            return documents
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        try:
            self.config.update(new_config)
            logger.info(f"语义分块配置已更新: {new_config}")
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
    
    def get_chunking_stats(self, chunks: List[Document]) -> Dict[str, Any]:
        """获取分块统计信息"""
        try:
            if not chunks:
                return {'total_chunks': 0}
            
            chunk_sizes = [len(chunk.page_content) for chunk in chunks]
            chunk_methods = [chunk.metadata.get('chunk_method', 'unknown') for chunk in chunks]
            
            stats = {
                'total_chunks': len(chunks),
                'avg_chunk_size': np.mean(chunk_sizes),
                'min_chunk_size': min(chunk_sizes),
                'max_chunk_size': max(chunk_sizes),
                'std_chunk_size': np.std(chunk_sizes),
                'chunk_methods': {}
            }
            
            # 统计分块方法使用情况
            for method in chunk_methods:
                stats['chunk_methods'][method] = stats['chunk_methods'].get(method, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"获取分块统计失败: {e}")
            return {'error': str(e)}
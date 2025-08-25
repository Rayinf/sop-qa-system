from typing import List, Optional, Dict, Any
import os
import hashlib
import logging
import uuid
from pathlib import Path
from datetime import datetime
import pandas as pd

from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.database import Document as DocumentModel, VectorIndex
from app.core.database import get_db
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """文档处理器类"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        
        # 初始化LLM用于自动分类
        try:
            # 初始化 LLM 服务
            self.llm_service = LLMService()
            logger.info("LLM服务初始化完成")
        except Exception as e:
            logger.warning(f"LLM服务初始化失败，将使用关键词分类: {e}")
            self.llm_service = None
        
        # 支持的文件类型
        self.supported_types = {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'docx',
            '.txt': 'txt',
            '.md': 'markdown',
            '.markdown': 'markdown',
            '.xlsx': 'excel',
            '.xls': 'excel'
        }
    
    def get_file_type(self, file_path: str) -> Optional[str]:
        """获取文件类型"""
        file_extension = Path(file_path).suffix.lower()
        return self.supported_types.get(file_extension)
    
    def is_supported_file(self, file_path: str) -> bool:
        """检查文件是否支持"""
        return self.get_file_type(file_path) is not None
    
    def calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希失败: {e}")
            return ""
    
    def load_document(self, file_path: str, file_type: str) -> List[Document]:
        """加载文档"""
        try:
            if file_type == 'pdf':
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif file_type == 'docx':
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
            elif file_type == 'txt':
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
            elif file_type == 'markdown':
                loader = UnstructuredMarkdownLoader(file_path)
                documents = loader.load()
            elif file_type == 'excel':
                documents = self._load_excel_document(file_path)
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")
            
            logger.info(f"成功加载文档: {file_path}, 页数: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"加载文档失败 {file_path}: {e}")
            raise
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"文档分割完成，共 {len(chunks)} 个片段")
            return chunks
        except Exception as e:
            logger.error(f"文档分割失败: {e}")
            raise
    
    def enhance_metadata(self, chunks: List[Document], 
                        document_id: str, 
                        original_metadata: Dict[str, Any]) -> List[Document]:
        """增强元数据"""
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            # 生成唯一的chunk_id
            chunk_id = f"{document_id}_chunk_{i}"
            
            # 增强元数据
            enhanced_metadata = {
                **chunk.metadata,
                **original_metadata,
                'document_id': document_id,
                'chunk_id': chunk_id,
                'chunk_index': i,
                'chunk_length': len(chunk.page_content),
                'processed_at': datetime.utcnow().isoformat()
            }
            
            # 创建新的Document对象
            enhanced_chunk = Document(
                page_content=chunk.page_content,
                metadata=enhanced_metadata
            )
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks

    def _load_excel_document(self, file_path: str) -> List[Document]:
        """加载Excel文档 - 优化大文件处理"""
        try:
            # 读取Excel文件的所有工作表
            excel_file = pd.ExcelFile(file_path)
            documents = []
            
            for sheet_name in excel_file.sheet_names:
                try:
                    # 先读取工作表信息
                    df_info = pd.read_excel(file_path, sheet_name=sheet_name, nrows=0)
                    total_rows = len(pd.read_excel(file_path, sheet_name=sheet_name))
                    
                    logger.info(f"处理工作表 {sheet_name}: {total_rows} 行, {len(df_info.columns)} 列")
                    
                    # 如果行数超过1000行，分批处理
                    if total_rows > 1000:
                        documents.extend(self._process_large_excel_sheet(file_path, sheet_name, total_rows))
                    else:
                        documents.extend(self._process_small_excel_sheet(file_path, sheet_name))
                    
                except Exception as e:
                    logger.warning(f"读取工作表 {sheet_name} 失败: {e}")
                    # 创建错误文档
                    error_content = f"工作表: {sheet_name}\n读取失败: {str(e)}"
                    error_metadata = {
                        'source': file_path,
                        'sheet_name': sheet_name,
                        'page': len(documents),
                        'file_type': 'excel',
                        'error': str(e)
                    }
                    error_document = Document(page_content=error_content, metadata=error_metadata)
                    documents.append(error_document)
            
            if not documents:
                # 如果没有成功读取任何工作表，创建一个空文档
                empty_content = f"Excel文件: {Path(file_path).name}\n无法读取任何工作表内容"
                empty_metadata = {
                    'source': file_path,
                    'page': 0,
                    'file_type': 'excel',
                    'error': 'No readable sheets'
                }
                documents.append(Document(page_content=empty_content, metadata=empty_metadata))
            
            logger.info(f"成功加载Excel文件: {file_path}, 总文档数: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"加载Excel文件失败 {file_path}: {e}")
            # 创建错误文档以确保不会完全失败
            error_content = f"Excel文件: {Path(file_path).name}\n加载失败: {str(e)}"
            error_metadata = {
                'source': file_path,
                'page': 0,
                'file_type': 'excel',
                'error': str(e)
            }
            return [Document(page_content=error_content, metadata=error_metadata)]
    
    def _process_small_excel_sheet(self, file_path: str, sheet_name: str) -> List[Document]:
        """处理小型Excel工作表（行数<=1000）"""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # 将DataFrame转换为文本
            content_parts = []
            
            # 添加工作表标题
            content_parts.append(f"工作表: {sheet_name}")
            content_parts.append("=" * 50)
            
            # 如果DataFrame不为空，转换为文本
            if not df.empty:
                # 处理列名
                if not df.columns.empty:
                    content_parts.append("列名: " + ", ".join(str(col) for col in df.columns))
                    content_parts.append("-" * 30)
                
                # 转换数据为文本格式
                for index, row in df.iterrows():
                    row_text = []
                    for col, value in row.items():
                        if pd.notna(value):  # 跳过空值
                            row_text.append(f"{col}: {value}")
                    if row_text:  # 只添加非空行
                        content_parts.append(" | ".join(row_text))
            else:
                content_parts.append("(空工作表)")
            
            # 创建Document对象
            content = "\n".join(content_parts)
            metadata = {
                'source': file_path,
                'sheet_name': sheet_name,
                'page': 0,
                'file_type': 'excel',
                'rows': len(df) if not df.empty else 0,
                'columns': len(df.columns) if not df.empty else 0
            }
            
            return [Document(page_content=content, metadata=metadata)]
            
        except Exception as e:
            logger.error(f"处理小型工作表失败 {sheet_name}: {e}")
            return []
    
    def _process_large_excel_sheet(self, file_path: str, sheet_name: str, total_rows: int) -> List[Document]:
        """处理大型Excel工作表（行数>1000），分批处理"""
        try:
            documents = []
            batch_size = 1000  # 每批处理1000行
            
            # 先读取列信息
            df_info = pd.read_excel(file_path, sheet_name=sheet_name, nrows=0)
            columns = df_info.columns.tolist()
            
            # 创建工作表概览文档
            overview_content = f"""工作表: {sheet_name}
{"=" * 50}
总行数: {total_rows}
列数: {len(columns)}
列名: {', '.join(str(col) for col in columns)}

注意: 由于数据量较大，此工作表已分批处理。"""
            
            overview_metadata = {
                'source': file_path,
                'sheet_name': sheet_name,
                'page': 0,
                'file_type': 'excel',
                'rows': total_rows,
                'columns': len(columns),
                'batch_type': 'overview'
            }
            
            documents.append(Document(page_content=overview_content, metadata=overview_metadata))
            
            # 分批处理数据
            for batch_start in range(0, total_rows, batch_size):
                batch_end = min(batch_start + batch_size, total_rows)
                
                try:
                    # 读取当前批次的数据
                    df_batch = pd.read_excel(
                        file_path, 
                        sheet_name=sheet_name, 
                        skiprows=range(1, batch_start + 1) if batch_start > 0 else None,
                        nrows=batch_size
                    )
                    
                    if df_batch.empty:
                        continue
                    
                    # 转换批次数据为文本
                    content_parts = []
                    content_parts.append(f"工作表: {sheet_name} (第 {batch_start + 1}-{batch_end} 行)")
                    content_parts.append("-" * 50)
                    
                    # 转换数据为文本格式
                    for index, row in df_batch.iterrows():
                        row_text = []
                        for col, value in row.items():
                            if pd.notna(value):  # 跳过空值
                                row_text.append(f"{col}: {value}")
                        if row_text:  # 只添加非空行
                            content_parts.append(" | ".join(row_text))
                    
                    # 创建批次Document对象
                    content = "\n".join(content_parts)
                    metadata = {
                        'source': file_path,
                        'sheet_name': sheet_name,
                        'page': len(documents),
                        'file_type': 'excel',
                        'rows': len(df_batch),
                        'columns': len(df_batch.columns),
                        'batch_type': 'data',
                        'batch_start': batch_start,
                        'batch_end': batch_end,
                        'total_rows': total_rows
                    }
                    
                    documents.append(Document(page_content=content, metadata=metadata))
                    
                    # 记录处理进度
                    if len(documents) % 10 == 0:
                        logger.info(f"已处理 {sheet_name} 工作表 {batch_end}/{total_rows} 行")
                    
                except Exception as e:
                    logger.warning(f"处理批次 {batch_start}-{batch_end} 失败: {e}")
                    continue
            
            logger.info(f"大型工作表 {sheet_name} 处理完成，生成 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            logger.error(f"处理大型工作表失败 {sheet_name}: {e}")
            return []

    @staticmethod
    def _sanitize_metadata(data: Any) -> Any:
        """递归将 UUID、datetime 等不可直接 JSON 序列化的对象转换为可序列化格式。"""
        from datetime import datetime as _dt
        import uuid as _uuid

        if isinstance(data, dict):
            return {k: DocumentProcessor._sanitize_metadata(v) for k, v in data.items()}
        if isinstance(data, list):
            return [DocumentProcessor._sanitize_metadata(v) for v in data]
        if isinstance(data, _uuid.UUID):
            return str(data)
        if isinstance(data, _dt):
            return data.isoformat()
        return data
    
    def process_document(self, 
                        file_path: str, 
                        title: str,
                        category: str = "通用文档",
                        tags: List[str] = None,
                        version: str = "1.0",
                        user_id: int = None,
                        kb_id: Optional[str] = None) -> Dict[str, Any]:
        """处理文档的主要方法"""
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 检查文件类型
            file_type = self.get_file_type(file_path)
            if not file_type:
                raise ValueError(f"不支持的文件类型: {Path(file_path).suffix}")
            
            # 计算文件哈希
            file_hash = self.calculate_file_hash(file_path)
            
            # 获取文件信息
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size
            
            # 加载文档
            documents = self.load_document(file_path, file_type)
            
            # 分割文档
            chunks = self.split_documents(documents)
            
            # 自动分类（如果category是默认值）
            if category == "通用文档":
                auto_category = self._auto_classify_document(documents, title)
                if auto_category:
                    category = auto_category
                    logger.info(f"文档自动分类结果: {title} -> {category}")
            
            # 生成文档ID
            document_id = str(uuid.uuid4())
            
            # 准备元数据
            metadata = {
                'title': title,
                'filename': os.path.basename(file_path),
                'file_path': file_path,
                'file_type': file_type,
                'file_size': file_size,
                'file_hash': file_hash,
                'category': category,
                'tags': tags or [],
                'version': version,
                'upload_by': user_id,
                'total_chunks': len(chunks),
                'kb_id': kb_id
            }
            
            # 增强chunk元数据
            enhanced_chunks = self.enhance_metadata(chunks, document_id, metadata)
            
            result = {
                'document_id': document_id,
                'chunks': enhanced_chunks,
                'metadata': metadata,
                'total_chunks': len(enhanced_chunks),
                'processing_time': datetime.utcnow().isoformat()
            }
            
            logger.info(f"文档处理完成: {title}, 文档ID: {document_id}, 片段数: {len(enhanced_chunks)}")
            return result
            
        except Exception as e:
            logger.error(f"文档处理失败 {file_path}: {e}")
            raise
    
    def save_document_to_db(self, 
                           db: Session, 
                           processing_result: Dict[str, Any]) -> DocumentModel:
        """保存文档信息到数据库"""
        try:
            metadata = processing_result['metadata']
            
            # 创建文档记录
            document = DocumentModel(
                id=processing_result['document_id'],
                title=metadata['title'],
                filename=metadata['filename'],
                file_path=metadata['file_path'],
                file_type=metadata['file_type'],
                file_size=metadata['file_size'],
                category=metadata['category'],
                tags=metadata['tags'],
                version=metadata['version'],
                uploaded_by=None,  # 暂时设为None，避免外键约束问题
                kb_id=metadata.get('kb_id'),  # 添加知识库ID
                doc_metadata={
                    'processing_time': processing_result['processing_time'],
                    'chunk_size': settings.chunk_size,
                    'chunk_overlap': settings.chunk_overlap,
                    'file_hash': metadata['file_hash'],
                    'total_chunks': metadata['total_chunks']
                },
                # 设置文档状态为已处理
                status="processed"
            )
            
            db.add(document)
            db.commit()
            db.refresh(document)
            
            logger.info(f"文档保存到数据库成功: {document.id}")
            return document
            
        except Exception as e:
            logger.error(f"保存文档到数据库失败: {e}")
            db.rollback()
            raise
    
    def save_chunks_to_db(self, 
                         db: Session, 
                         document_id: str, 
                         chunks: List[Document]) -> List[VectorIndex]:
        """保存文档片段到数据库"""
        try:
            logger.info(f"开始保存文档片段，document_id: {document_id}, chunks数量: {len(chunks)}")
            vector_indices = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"处理第 {i+1} 个chunk，metadata: {chunk.metadata}")
                # 清洗元数据，避免出现 UUID 等导致 JSON 序列化失败
                sanitized_meta = self._sanitize_metadata(chunk.metadata)

                vector_index = VectorIndex(
                    document_id=uuid.UUID(document_id),
                    chunk_id=sanitized_meta['chunk_id'],
                    chunk_text=chunk.page_content,
                    chunk_index=sanitized_meta['chunk_index'],
                    page_number=sanitized_meta.get('page', None),
                    kb_id=sanitized_meta.get('kb_id'),  # 添加知识库ID
                    vector_metadata=sanitized_meta
                )
                vector_indices.append(vector_index)
            
            logger.info(f"准备保存 {len(vector_indices)} 个VectorIndex到数据库")
            db.add_all(vector_indices)
            db.commit()
            
            logger.info(f"成功保存 {len(vector_indices)} 个文档片段到数据库")
            return vector_indices
            
        except Exception as e:
            logger.error(f"保存文档片段到数据库失败: {e}")
            logger.exception("详细错误信息:")
            db.rollback()
            raise
    
    def get_document_by_hash(self, db: Session, file_hash: str) -> Optional[DocumentModel]:
        """根据文件哈希查找文档"""
        try:
            return db.query(DocumentModel).filter(
                DocumentModel.doc_metadata['file_hash'].astext == file_hash
            ).first()
        except Exception as e:
            logger.error(f"查找文档失败: {e}")
            return None
    
    def delete_document(self, db: Session, document_id: str) -> bool:
        """删除文档及其相关数据"""
        try:
            # 删除向量索引
            db.query(VectorIndex).filter(
                VectorIndex.document_id == document_id
            ).delete()
            
            # 删除文档记录
            document = db.query(DocumentModel).filter(
                DocumentModel.id == document_id
            ).first()
            
            if document:
                # 删除文件
                if os.path.exists(document.file_path):
                    os.remove(document.file_path)
                
                db.delete(document)
                db.commit()
                
                logger.info(f"文档删除成功: {document_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            db.rollback()
            return False
    
    def update_document_status(self, 
                              db: Session, 
                              document_id: str, 
                              status: str) -> bool:
        """更新文档状态"""
        try:
            document = db.query(DocumentModel).filter(
                DocumentModel.id == document_id
            ).first()
            
            if document:
                document.status = status
                document.updated_at = datetime.utcnow()
                db.commit()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"更新文档状态失败: {e}")
            db.rollback()
            return False


    
    def _auto_classify_document(self, documents: List[Document], title: str) -> Optional[str]:
        """自动分类文档"""
        try:
            # 提取文档内容用于分类
            content_sample = ""
            for doc in documents[:3]:  # 只取前3页内容进行分析
                content_sample += doc.page_content[:500] + "\n"
            
            # 结合标题和内容进行分类
            full_text = f"标题: {title}\n内容: {content_sample[:1000]}"
            
            # 优先使用LLM分类
            if self.llm_service:
                return self._llm_classify(full_text)
            else:
                return self._keyword_classify(full_text)
                
        except Exception as e:
            logger.error(f"文档自动分类失败: {e}")
            return None
    
    def _llm_classify(self, text: str) -> Optional[str]:
        """使用LLM进行文档分类"""
        try:
            classification_prompt = f"""请根据以下文档内容，判断其最合适的类别。

可选类别：
- manual: 质量手册、管理制度、规范标准、操作手册
- procedure: 操作程序、工作流程、作业指导书、程序文件
- development: 开发程序、技术文档、开发指南、编程相关
- policy: 政策文件、规章制度、管理规定
- guideline: 指导文件、指南、建议
- other: 其他类型文档

文档内容：
{text}

请仅返回最合适的类别名称（manual/procedure/development/policy/guideline/other），不要包含其他内容。"""
            
            result = (self.llm_service.generate_response(classification_prompt) if self.llm_service else "").strip().lower()
            
            # 验证分类结果
            valid_categories = ["manual", "procedure", "development", "policy", "guideline", "other"]
            if result in valid_categories:
                return result
            else:
                logger.warning(f"LLM分类结果无效: {result}，使用关键词分类")
                return self._keyword_classify(text)
                
        except Exception as e:
            logger.error(f"LLM分类失败: {e}")
            return self._keyword_classify(text)
    
    def _keyword_classify(self, text: str) -> str:
        """基于关键词的文档分类"""
        text_lower = text.lower()
        
        # 定义关键词映射
        keyword_mapping = {
            "development": [
                "开发", "程序", "代码", "编程", "软件", "系统", "技术", "api", "接口",
                "development", "code", "programming", "software", "system", "technical"
            ],
            "manual": [
                "手册", "规范", "标准", "质量", "管理", "制度", "体系",
                "manual", "standard", "quality", "management", "specification"
            ],
            "procedure": [
                "流程", "程序", "操作", "作业", "指导书", "步骤", "工艺",
                "procedure", "process", "operation", "workflow", "instruction"
            ],
            "policy": [
                "政策", "规章", "规定", "条例", "办法", "制度",
                "policy", "regulation", "rule", "guideline"
            ],
            "guideline": [
                "指南", "指导", "建议", "推荐", "最佳实践",
                "guide", "guideline", "recommendation", "best practice"
            ]
        }
        
        # 计算每个类别的匹配分数
        category_scores = {}
        for category, keywords in keyword_mapping.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        # 返回得分最高的类别
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            logger.info(f"关键词分类结果: {best_category} (得分: {category_scores[best_category]})")
            return best_category
        
        # 如果没有匹配的关键词，返回other
        return "other"

# 文档处理工具函数
def validate_file_size(file_path: str, max_size_mb: int = 1024) -> bool:
    """验证文件大小"""
    try:
        file_size = os.path.getsize(file_path)
        max_size_bytes = max_size_mb * 1024 * 1024
        return file_size <= max_size_bytes
    except Exception:
        return False

def clean_text(text: str) -> str:
    """清理文本内容"""
    import re
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊字符（保留中英文、数字、常用标点）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.,;:!?()\[\]{}"\'-]', '', text)
    
    # 去除首尾空格
    text = text.strip()
    
    return text

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """提取关键词（简单实现）"""
    import re
    from collections import Counter
    
    # 简单的中英文分词
    chinese_words = re.findall(r'[\u4e00-\u9fa5]+', text)
    english_words = re.findall(r'[a-zA-Z]+', text.lower())
    
    # 过滤停用词（简化版）
    stop_words = {'的', '是', '在', '了', '和', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    all_words = [word for word in chinese_words + english_words 
                 if len(word) > 1 and word not in stop_words]
    
    # 统计词频
    word_counts = Counter(all_words)
    
    # 返回最常见的关键词
    return [word for word, count in word_counts.most_common(max_keywords)]
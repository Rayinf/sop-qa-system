#!/usr/bin/env python3

import os
import sys
import logging
from pathlib import Path
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.vector_service import VectorService
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_excel_as_documents(file_path):
    """加载Excel文件并转换为文档"""
    documents = []
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        logger.info(f"Excel文件包含 {len(df)} 行数据")
        
        # 将每行转换为文档
        for index, row in df.iterrows():
            # 将行数据转换为文本
            content_parts = []
            for col, value in row.items():
                if pd.notna(value):
                    content_parts.append(f"{col}: {value}")
            
            content = "\n".join(content_parts)
            
            # 创建文档
            doc = Document(
                page_content=content,
                metadata={
                    "source": os.path.basename(file_path),
                    "category": "company_data",
                    "row_index": index,
                    "title": f"公司数据_{index+1}"
                }
            )
            documents.append(doc)
            
    except Exception as e:
        logger.error(f"加载Excel文件失败: {e}")
        
    return documents

def rebuild_vector_database():
    """重建向量数据库"""
    
    # 文档目录
    docs_dir = "../data/documents"
    
    if not os.path.exists(docs_dir):
        logger.error(f"文档目录不存在: {docs_dir}")
        return False
    
    try:
        # 初始化向量服务
        vector_service = VectorService.get_instance()
        
        # 收集所有文档
        all_documents = []
        
        # 处理所有文件
        for filename in os.listdir(docs_dir):
            file_path = os.path.join(docs_dir, filename)
            
            if filename.endswith(('.xls', '.xlsx')):
                logger.info(f"处理Excel文件: {filename}")
                excel_docs = load_excel_as_documents(file_path)
                all_documents.extend(excel_docs)
                logger.info(f"从Excel文件加载了 {len(excel_docs)} 个文档")
                
            elif filename.endswith('.pdf'):
                logger.info(f"处理PDF文件: {filename}")
                try:
                    loader = PyPDFLoader(file_path)
                    pdf_docs = loader.load()
                    
                    # 分割PDF文档
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    split_docs = text_splitter.split_documents(pdf_docs)
                    
                    # 更新元数据
                    for doc in split_docs:
                        doc.metadata.update({
                            "category": "procedure",
                            "title": filename.replace('.pdf', '')
                        })
                    
                    all_documents.extend(split_docs)
                    logger.info(f"从PDF文件加载了 {len(split_docs)} 个文档块")
                    
                except Exception as e:
                    logger.error(f"处理PDF文件 {filename} 失败: {e}")
                    continue
                    
            elif filename.endswith('.txt'):
                logger.info(f"处理TXT文件: {filename}")
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    txt_docs = loader.load()
                    
                    # 分割文本文档
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    split_docs = text_splitter.split_documents(txt_docs)
                    
                    # 更新元数据
                    for doc in split_docs:
                        doc.metadata.update({
                            "category": "document",
                            "title": filename.replace('.txt', '')
                        })
                    
                    all_documents.extend(split_docs)
                    logger.info(f"从TXT文件加载了 {len(split_docs)} 个文档块")
                    
                except Exception as e:
                    logger.error(f"处理TXT文件 {filename} 失败: {e}")
                    continue
                    
            elif filename.endswith('.md'):
                logger.info(f"处理Markdown文件: {filename}")
                try:
                    loader = UnstructuredMarkdownLoader(file_path)
                    md_docs = loader.load()
                    
                    # 分割Markdown文档
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    split_docs = text_splitter.split_documents(md_docs)
                    
                    # 更新元数据
                    for doc in split_docs:
                        doc.metadata.update({
                            "category": "document",
                            "title": filename.replace('.md', '')
                        })
                    
                    all_documents.extend(split_docs)
                    logger.info(f"从Markdown文件加载了 {len(split_docs)} 个文档块")
                    
                except Exception as e:
                    logger.error(f"处理Markdown文件 {filename} 失败: {e}")
                    continue
                    
            elif filename.endswith('.docx'):
                logger.info(f"处理DOCX文件: {filename}")
                try:
                    loader = Docx2txtLoader(file_path)
                    docx_docs = loader.load()
                    
                    # 分割DOCX文档
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    split_docs = text_splitter.split_documents(docx_docs)
                    
                    # 更新元数据
                    for doc in split_docs:
                        doc.metadata.update({
                            "category": "document",
                            "title": filename.replace('.docx', '')
                        })
                    
                    all_documents.extend(split_docs)
                    logger.info(f"从DOCX文件加载了 {len(split_docs)} 个文档块")
                    
                except Exception as e:
                    logger.error(f"处理DOCX文件 {filename} 失败: {e}")
                    continue
                    
            else:
                logger.info(f"跳过不支持的文件格式: {filename}")
        
        logger.info(f"总共收集了 {len(all_documents)} 个文档")
        
        if all_documents:
            # 重建向量数据库
            logger.info("开始重建向量数据库...")
            vector_service.rebuild_vector_store_from_documents(all_documents)
            logger.info("向量数据库重建完成")
            
            # 测试搜索
            logger.info("\n=== 测试搜索 ===")
            
            # 测试Excel数据搜索
            excel_results = vector_service.search_similar_documents("苏州慧胜自动化设备有限公司", k=3)
            logger.info(f"Excel数据搜索结果: {len(excel_results)} 个")
            
            # 测试PDF数据搜索
            pdf_results = vector_service.search_similar_documents("质量手册", k=3)
            logger.info(f"PDF数据搜索结果: {len(pdf_results)} 个")
            
            return True
        else:
            logger.error("没有找到任何文档")
            return False
            
    except Exception as e:
        logger.error(f"重建向量数据库失败: {e}")
        logger.exception("详细错误信息:")
        return False

if __name__ == "__main__":
    success = rebuild_vector_database()
    if success:
        print("✅ 向量数据库重建成功")
    else:
        print("❌ 向量数据库重建失败")
        sys.exit(1)
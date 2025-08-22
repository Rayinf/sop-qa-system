#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from app.services.vector_service import VectorService
from app.core.config import settings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rebuild_vector_database():
    """重建向量数据库，包含所有PDF文档内容"""
    try:
        # 初始化向量服务
        vector_service = VectorService.get_instance()
        
        # 文档目录
        doc_dir = "/Volumes/PortableSSD/langchain/sop-qa-system/data/documents"
        
        # 获取所有PDF文件
        pdf_files = [f for f in os.listdir(doc_dir) if f.endswith('.pdf') and not f.startswith('._')]
        
        logger.info(f"找到 {len(pdf_files)} 个PDF文档")
        for pdf_file in pdf_files:
            logger.info(f"  - {pdf_file}")
        
        # 初始化文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        
        all_documents = []
        
        # 处理每个PDF文档
        for pdf_file in pdf_files:
            file_path = os.path.join(doc_dir, pdf_file)
            logger.info(f"正在处理: {pdf_file}")
            
            try:
                # 加载PDF
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
                # 分块
                chunks = text_splitter.split_documents(documents)
                
                # 增强元数据
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "title": pdf_file.replace('.pdf', ''),
                        "source": pdf_file,
                        "chunk_id": f"{pdf_file}_{i}"
                    })
                
                all_documents.extend(chunks)
                logger.info(f"✅ {pdf_file}: {len(documents)}页 -> {len(chunks)}个分块")
                
            except Exception as e:
                logger.error(f"❌ 处理文档失败 {pdf_file}: {str(e)}")
        
        logger.info(f"总共生成 {len(all_documents)} 个文档分块")
        
        # 重建向量数据库
        logger.info("开始重建向量数据库...")
        success = vector_service.rebuild_vector_store_from_documents(all_documents)
        if success:
            logger.info("✅ 向量数据库重建成功！")
        else:
            logger.error("❌ 向量数据库重建失败！")
            return False
        
        # 测试搜索
        logger.info("测试向量搜索...")
        test_results = vector_service.search_similar_documents("4.6合同变更管理输入输出", k=3)
        logger.info(f"搜索到 {len(test_results)} 个相关文档")
        
        for i, (doc, score) in enumerate(test_results):
            logger.info(f"结果{i+1}: {doc.metadata.get('title', '未知')} (相似度: {score:.3f}) - {doc.page_content[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"重建向量数据库失败: {str(e)}")
        return False

if __name__ == "__main__":
    success = rebuild_vector_database()
    if success:
        print("✅ 向量数据库重建成功")
    else:
        print("❌ 向量数据库重建失败")
        sys.exit(1)
#!/usr/bin/env python3
"""
初始化向量数据库脚本
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import SessionLocal
from app.services.document_service import DocumentService
from app.services.vector_service import VectorService
from app.models.database import Document as DocumentModel
from sqlalchemy.orm import Session
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_vector_database():
    """初始化向量数据库"""
    try:
        # 创建数据库会话
        db = SessionLocal()
        
        # 创建服务实例
        document_service = DocumentService()
        vector_service = VectorService.get_instance()
        
        # 检查是否有测试文档
        test_file_path = Path("data/documents/test_sop.txt")
        if not test_file_path.exists():
            logger.error(f"测试文档不存在: {test_file_path}")
            return False
            
        logger.info("开始初始化向量数据库...")
        
        # 读取测试文档内容
        with open(test_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 创建文档记录
        document = DocumentModel(
            title="SOP操作手册",
            filename="test_sop.txt",
            file_path=str(test_file_path),
            file_size=len(content.encode('utf-8')),
            file_type="text/plain",
            category="manual",
            status="active",
            processing_status="pending",
            uploaded_by=None
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)
        
        logger.info(f"文档记录创建成功: {document.id}")
        
        # 处理文档
        processed_doc = document_service.process_document(
            db=db,
            file_path=str(test_file_path),
            title="SOP操作手册",
            category="manual",
            tags=[],
            version="1.0",
            user_id=None
        )
        
        # 向量化文档
        success = document_service.vectorize_document(
            db=db,
            document_id=str(document.id)
        )
        
        if success:
            logger.info("向量数据库初始化成功！")
            
            # 测试向量搜索
            results = vector_service.search_similar_documents(
                query="什么是SOP？",
                k=3
            )
            
            if results:
                logger.info(f"向量搜索测试成功，找到 {len(results)} 个相关文档")
                for i, (doc, score) in enumerate(results):
                    logger.info(f"结果 {i+1}: 相似度={score:.3f}, 内容={doc.page_content[:100]}...")
            else:
                logger.warning("向量搜索测试失败")
                
            return True
        else:
            logger.error("文档向量化失败")
            return False
            
    except Exception as e:
        logger.error(f"初始化向量数据库失败: {e}")
        return False
    finally:
        db.close()

if __name__ == "__main__":
    success = init_vector_database()
    if success:
        print("✅ 向量数据库初始化成功")
        sys.exit(0)
    else:
        print("❌ 向量数据库初始化失败")
        sys.exit(1)
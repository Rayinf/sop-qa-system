#!/usr/bin/env python3

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.services.document_service import DocumentService
from app.models.database import Document, VectorIndex

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_vector_database():
    """初始化向量数据库"""
    
    # 测试文档目录
    docs_dir = "data/documents"
    
    if not os.path.exists(docs_dir):
        logger.error(f"文档目录不存在: {docs_dir}")
        return False
    
    # 创建数据库会话
    db = SessionLocal()
    
    try:
        # 清理现有数据
        logger.info("清理现有数据...")
        db.query(VectorIndex).delete()
        db.query(Document).delete()
        db.commit()
        logger.info("清理现有数据完成")
        
        # 创建文档服务
        document_service = DocumentService()
        
        # 处理所有文档
        processed_count = 0
        for filename in os.listdir(docs_dir):
            if filename.endswith(('.txt', '.md', '.pdf', '.docx', '.xls', '.xlsx')):
                file_path = os.path.join(docs_dir, filename)
                logger.info(f"处理文档: {filename}")
                
                try:
                    document = document_service.process_document(
                        db=db,
                        file_path=file_path,
                        title=filename,
                        category="SOP",
                        tags=["测试"],
                        version="1.0",
                        user_id=None  # 设为None
                    )
                    
                    logger.info(f"文档处理完成: {document.id}")
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"处理文档 {filename} 失败: {e}")
                    continue
        
        # 检查最终结果
        doc_count = db.query(Document).count()
        vi_count = db.query(VectorIndex).count()
        
        logger.info(f"初始化完成 - 处理文档: {processed_count}, 数据库记录 - 文档: {doc_count}, 向量索引: {vi_count}")
        
        if vi_count > 0:
            logger.info("✅ 向量数据库初始化成功")
            return True
        else:
            logger.error("❌ 向量索引创建失败")
            return False
            
    except Exception as e:
        logger.error(f"向量数据库初始化失败: {e}")
        logger.exception("详细错误信息:")
        db.rollback()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    success = init_vector_database()
    if success:
        print("✅ 向量数据库初始化成功")
    else:
        print("❌ 向量数据库初始化失败")
        sys.exit(1)
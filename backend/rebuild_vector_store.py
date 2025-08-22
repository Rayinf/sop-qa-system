#!/usr/bin/env python3

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import SessionLocal
from app.services.vector_service import VectorService
from app.models.database import VectorIndex, Document
from langchain.schema import Document as LangChainDocument

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def rebuild_vector_store():
    """从数据库重建向量数据库"""
    
    # 创建数据库会话
    db = SessionLocal()
    
    try:
        # 检查数据库中的向量索引记录
        vector_indices = db.query(VectorIndex).all()
        
        if not vector_indices:
            logger.error("数据库中没有向量索引记录")
            return False
        
        logger.info(f"找到 {len(vector_indices)} 个向量索引记录")
        
        # 创建向量服务
        vector_service = VectorService.get_instance()
        
        # 准备文档列表
        documents = []
        for vi in vector_indices:
            # 创建LangChain文档对象
            doc = LangChainDocument(
                page_content=vi.chunk_text,
                metadata={
                    'document_id': str(vi.document_id),
                    'chunk_id': vi.chunk_id,
                    'chunk_index': vi.chunk_index,
                    'page_number': vi.page_number,
                    'start_char': vi.start_char,
                    'end_char': vi.end_char,
                    **vi.vector_metadata
                }
            )
            documents.append(doc)
        
        logger.info(f"准备向量化 {len(documents)} 个文档块")
        
        # 重建向量数据库
        success = vector_service.rebuild_vector_store_from_documents(documents)
        
        if success:
            logger.info("✅ 向量数据库重建成功")
            
            # 更新所有相关文档的状态为已向量化
            document_ids = set(vi.document_id for vi in vector_indices)
            updated_count = 0
            
            for doc_id in document_ids:
                document = db.query(Document).filter(Document.id == doc_id).first()
                if document:
                    document.status = 'vectorized'
                    updated_count += 1
            
            db.commit()
            logger.info(f"✅ 已更新 {updated_count} 个文档状态为已向量化")
            
            # 测试向量数据库
            test_query = "测试查询"
            results = vector_service.search_similar_documents(test_query, k=3)
            logger.info(f"测试查询返回 {len(results)} 个结果")
            
            return True
        else:
            logger.error("❌ 向量数据库重建失败")
            return False
            
    except Exception as e:
        logger.error(f"重建向量数据库失败: {e}")
        logger.exception("详细错误信息:")
        return False
    finally:
        db.close()

if __name__ == "__main__":
    success = rebuild_vector_store()
    if success:
        print("✅ 向量数据库重建成功")
    else:
        print("❌ 向量数据库重建失败")
        sys.exit(1)
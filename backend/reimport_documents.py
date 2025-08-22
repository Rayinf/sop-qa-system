#!/usr/bin/env python3
"""
重新导入文档脚本
用于将文档目录中的PDF文件重新导入到数据库中
"""

import os
import sys
from pathlib import Path
from typing import BinaryIO
import logging
from fastapi import UploadFile
from io import BytesIO

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from app.core.database import SessionLocal
from app.services.document_service import DocumentService
from app.models.database import User, Document, VectorIndex
from app.core.config import settings

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockUploadFile:
    """模拟UploadFile对象"""
    
    def __init__(self, file_path: Path):
        self.filename = file_path.name
        self.content_type = "application/pdf"
        self.file_path = file_path
        self._file = None
    
    @property
    def file(self) -> BinaryIO:
        if self._file is None:
            self._file = open(self.file_path, 'rb')
        return self._file
    
    def read(self, size: int = -1) -> bytes:
        return self.file.read(size)
    
    def close(self):
        if self._file:
            self._file.close()
            self._file = None

def ensure_admin_user(db):
    """确保存在管理员用户"""
    admin_user = db.query(User).filter(User.username == 'admin').first()
    if not admin_user:
        logger.info("创建管理员用户...")
        from app.core.auth import AuthService
        auth_service = AuthService()
        
        admin_user = User(
            username="admin",
            email="admin@example.com",
            password_hash=auth_service.get_password_hash("admin123456"),
            full_name="系统管理员",
            role="admin",
            is_active=True,
            is_superuser=True
        )
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        logger.info(f"管理员用户创建成功: {admin_user.id}")
    
    return admin_user

def reimport_documents():
    """重新导入文档目录中的所有文档"""
    
    # 文档目录
    docs_dir = Path(settings.upload_path)
    logger.info(f"文档目录: {docs_dir.absolute()}")
    
    if not docs_dir.exists():
        logger.error(f"文档目录不存在: {docs_dir}")
        return False
    
    # 获取所有PDF文件
    pdf_files = list(docs_dir.glob("*.pdf"))
    logger.info(f"找到 {len(pdf_files)} 个PDF文件")
    
    if not pdf_files:
        logger.warning("没有找到PDF文件")
        return True
    
    # 创建数据库会话
    db = SessionLocal()
    
    try:
        # 确保有管理员用户
        admin_user = ensure_admin_user(db)
        logger.info(f"使用管理员用户: {admin_user.username} (ID: {admin_user.id})")
        
        # 创建文档服务
        doc_service = DocumentService()
        
        success_count = 0
        error_count = 0
        
        # 处理每个PDF文件
        for pdf_file in pdf_files:
            try:
                logger.info(f"开始处理文档: {pdf_file.name}")
                
                # 确定文档类别
                category = "procedure"
                if "QMS-01" in pdf_file.name:
                    category = "manual"
                elif "QMS-OP" in pdf_file.name:
                    category = "procedure"
                
                logger.info(f"文档类别: {category}")
                
                # 创建模拟的UploadFile对象
                mock_file = MockUploadFile(pdf_file)
                
                try:
                    # 处理文档
                    document = doc_service.process_document(
                        db=db,
                        file=mock_file,
                        category=category,
                        tags=[],
                        description=f"重新导入的文档: {pdf_file.name}"
                    )
                    
                    logger.info(f"✅ 文档处理成功: {document.title} (ID: {document.id})")
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ 处理文档失败 {pdf_file.name}: {e}")
                    logger.exception("详细错误信息:")
                    error_count += 1
                
                finally:
                    mock_file.close()
                
            except Exception as e:
                logger.error(f"❌ 处理文档时发生意外错误 {pdf_file.name}: {e}")
                logger.exception("详细错误信息:")
                error_count += 1
        
        # 检查结果
        doc_count = db.query(Document).count()
        vector_count = db.query(VectorIndex).count()
        
        logger.info(f"重新导入完成:")
        logger.info(f"  - 成功处理: {success_count} 个文档")
        logger.info(f"  - 处理失败: {error_count} 个文档")
        logger.info(f"  - 数据库中文档总数: {doc_count}")
        logger.info(f"  - 数据库中向量索引总数: {vector_count}")
        
        return error_count == 0
        
    except Exception as e:
        logger.error(f"重新导入过程中发生错误: {e}")
        logger.exception("详细错误信息:")
        return False
    finally:
        db.close()

if __name__ == "__main__":
    logger.info("开始重新导入文档...")
    success = reimport_documents()
    if success:
        print("✅ 文档重新导入成功")
    else:
        print("❌ 文档重新导入失败")
        sys.exit(1)
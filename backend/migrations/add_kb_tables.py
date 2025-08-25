#!/usr/bin/env python3
"""
数据库迁移脚本：添加知识库相关表结构

添加以下表和字段：
1. 新增 kb 表（知识库表）
2. 新增 user_kb_prefs 表（用户知识库偏好表）
3. 在 documents 表中添加 kb_id 和 collection 字段
4. 在 vector_indices 表中添加 kb_id 字段
5. 添加相关索引
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.core.database import SessionLocal, engine
import logging
import uuid
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_kb_tables():
    """
    执行知识库表结构添加
    """
    db = SessionLocal()
    try:
        logger.info("开始执行知识库表结构迁移...")
        
        # 检查 kb 表是否已存在
        result = db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name = 'kb'
        """))
        
        if result.fetchone():
            logger.info("kb 表已存在，迁移已完成或不需要执行")
            return True
        
        # 1. 创建 kb 表
        logger.info("创建 kb 表...")
        db.execute(text("""
            CREATE TABLE kb (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(200) NOT NULL,
                code VARCHAR(50) NOT NULL UNIQUE,
                description TEXT,
                is_active BOOLEAN NOT NULL DEFAULT true,
                visibility VARCHAR(20) NOT NULL DEFAULT 'private',
                kb_metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        # 2. 创建 user_kb_prefs 表
        logger.info("创建 user_kb_prefs 表...")
        db.execute(text("""
            CREATE TABLE user_kb_prefs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                kb_id UUID NOT NULL REFERENCES kb(id) ON DELETE CASCADE,
                pinned BOOLEAN NOT NULL DEFAULT false,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, kb_id)
            )
        """))
        
        # 3. 在 documents 表中添加 kb_id 和 collection 字段
        logger.info("在 documents 表中添加 kb_id 和 collection 字段...")
        
        # 检查字段是否已存在
        result = db.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'documents' AND column_name = 'kb_id'
        """))
        
        if not result.fetchone():
            db.execute(text("""
                ALTER TABLE documents 
                ADD COLUMN kb_id UUID REFERENCES kb(id) ON DELETE SET NULL
            """))
            
        result = db.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'documents' AND column_name = 'collection'
        """))
        
        if not result.fetchone():
            db.execute(text("""
                ALTER TABLE documents 
                ADD COLUMN collection VARCHAR(100)
            """))
        
        # 4. 在 vector_indices 表中添加 kb_id 字段
        logger.info("在 vector_indices 表中添加 kb_id 字段...")
        
        result = db.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'vector_indices' AND column_name = 'kb_id'
        """))
        
        if not result.fetchone():
            db.execute(text("""
                ALTER TABLE vector_indices 
                ADD COLUMN kb_id UUID REFERENCES kb(id) ON DELETE SET NULL
            """))
        
        # 5. 添加索引
        logger.info("添加相关索引...")
        
        # KB 表索引
        db.execute(text("CREATE INDEX IF NOT EXISTS idx_kb_code ON kb(code)"))
        db.execute(text("CREATE INDEX IF NOT EXISTS idx_kb_active ON kb(is_active)"))
        
        # Documents 表索引
        db.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_kb_status ON documents(kb_id, status)"))
        
        # Vector indices 表索引
        db.execute(text("CREATE INDEX IF NOT EXISTS idx_vector_indices_kb ON vector_indices(kb_id)"))
        
        # User KB prefs 表索引
        db.execute(text("CREATE INDEX IF NOT EXISTS idx_user_kb_prefs_user ON user_kb_prefs(user_id)"))
        db.execute(text("CREATE INDEX IF NOT EXISTS idx_user_kb_prefs_kb ON user_kb_prefs(kb_id)"))
        db.execute(text("CREATE INDEX IF NOT EXISTS idx_user_kb_prefs_pinned ON user_kb_prefs(user_id, pinned)"))
        
        # 6. 创建默认知识库
        logger.info("创建默认知识库...")
        default_kb_id = str(uuid.uuid4())
        db.execute(text("""
            INSERT INTO kb (id, name, code, description, is_active, visibility)
            VALUES (:id, '默认知识库', 'DEFAULT', '系统默认知识库，包含所有未分类的文档', true, 'private')
        """), {"id": default_kb_id})
        
        # 7. 将现有文档关联到默认知识库
        logger.info("将现有文档关联到默认知识库...")
        result = db.execute(text("""
            UPDATE documents 
            SET kb_id = :kb_id 
            WHERE kb_id IS NULL
        """), {"kb_id": default_kb_id})
        
        affected_docs = result.rowcount
        logger.info(f"已将 {affected_docs} 个文档关联到默认知识库")
        
        # 8. 更新向量索引的 kb_id
        logger.info("更新向量索引的 kb_id...")
        result = db.execute(text("""
            UPDATE vector_indices 
            SET kb_id = d.kb_id
            FROM documents d
            WHERE vector_indices.document_id = d.id
            AND vector_indices.kb_id IS NULL
        """))
        
        affected_vectors = result.rowcount
        logger.info(f"已更新 {affected_vectors} 个向量索引的 kb_id")
        
        # 提交事务
        db.commit()
        
        logger.info("知识库表结构迁移完成！")
        logger.info(f"- 创建了 kb 和 user_kb_prefs 表")
        logger.info(f"- 在 documents 和 vector_indices 表中添加了 kb_id 字段")
        logger.info(f"- 创建了默认知识库并关联了 {affected_docs} 个文档")
        logger.info(f"- 更新了 {affected_vectors} 个向量索引")
        
        return True
        
    except Exception as e:
        logger.error(f"迁移过程中发生错误: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def rollback_kb_tables():
    """
    回滚知识库表结构（仅用于开发测试）
    """
    db = SessionLocal()
    try:
        logger.info("开始回滚知识库表结构...")
        
        # 删除索引
        logger.info("删除索引...")
        indexes_to_drop = [
            "idx_kb_code", "idx_kb_active", "idx_documents_kb_status",
            "idx_vector_indices_kb", "idx_user_kb_prefs_user",
            "idx_user_kb_prefs_kb", "idx_user_kb_prefs_pinned"
        ]
        
        for index in indexes_to_drop:
            try:
                db.execute(text(f"DROP INDEX IF EXISTS {index}"))
            except Exception as e:
                logger.warning(f"删除索引 {index} 失败: {e}")
        
        # 删除字段
        logger.info("删除添加的字段...")
        try:
            db.execute(text("ALTER TABLE vector_indices DROP COLUMN IF EXISTS kb_id"))
        except Exception as e:
            logger.warning(f"删除 vector_indices.kb_id 字段失败: {e}")
            
        try:
            db.execute(text("ALTER TABLE documents DROP COLUMN IF EXISTS collection"))
        except Exception as e:
            logger.warning(f"删除 documents.collection 字段失败: {e}")
            
        try:
            db.execute(text("ALTER TABLE documents DROP COLUMN IF EXISTS kb_id"))
        except Exception as e:
            logger.warning(f"删除 documents.kb_id 字段失败: {e}")
        
        # 删除表
        logger.info("删除表...")
        try:
            db.execute(text("DROP TABLE IF EXISTS user_kb_prefs"))
        except Exception as e:
            logger.warning(f"删除 user_kb_prefs 表失败: {e}")
            
        try:
            db.execute(text("DROP TABLE IF EXISTS kb"))
        except Exception as e:
            logger.warning(f"删除 kb 表失败: {e}")
        
        db.commit()
        logger.info("知识库表结构回滚完成！")
        return True
        
    except Exception as e:
        logger.error(f"回滚过程中发生错误: {e}")
        db.rollback()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="知识库表结构迁移脚本")
    parser.add_argument("--rollback", action="store_true", help="回滚迁移（仅用于开发测试）")
    
    args = parser.parse_args()
    
    if args.rollback:
        success = rollback_kb_tables()
    else:
        success = add_kb_tables()
    
    if success:
        logger.info("操作成功完成")
        sys.exit(0)
    else:
        logger.error("操作失败")
        sys.exit(1)
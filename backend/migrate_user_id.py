#!/usr/bin/env python3
"""
数据库迁移脚本：将qa_logs表的user_id字段改为可空
"""

from sqlalchemy import text
from app.core.database import engine

def migrate_user_id_nullable():
    """将qa_logs表的user_id字段改为可空"""
    try:
        with engine.connect() as conn:
            # 开始事务
            trans = conn.begin()
            try:
                # 修改user_id字段为可空
                conn.execute(text('ALTER TABLE qa_logs ALTER COLUMN user_id DROP NOT NULL'))
                trans.commit()
                print("✅ 数据库迁移完成：qa_logs.user_id字段现在允许为空")
            except Exception as e:
                trans.rollback()
                print(f"❌ 迁移失败，已回滚: {e}")
                raise
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        raise

if __name__ == "__main__":
    migrate_user_id_nullable()
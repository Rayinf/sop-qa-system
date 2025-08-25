#!/usr/bin/env python3
"""
数据库迁移脚本：统一文档状态字段

将 processing_status 字段的值合并到 status 字段，并移除 processing_status 字段。

状态映射规则：
- processing_status="pending" + status="active" -> status="uploaded"
- processing_status="processing" -> status="processing"
- processing_status="completed" + status="active" -> status="processed"
- processing_status="failed" -> status="failed"
- processing_status="vectorizing" -> status="vectorizing"
- processing_status="vectorized" -> status="vectorized"
- status="archived" -> 保持不变
- status="deleted" -> 保持不变
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.core.database import SessionLocal, engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_status_fields():
    """
    执行状态字段迁移
    """
    db = SessionLocal()
    try:
        logger.info("开始执行状态字段迁移...")
        
        # 检查 processing_status 字段是否存在
        result = db.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'documents' AND column_name = 'processing_status'
        """))
        
        if not result.fetchone():
            logger.info("processing_status 字段不存在，迁移已完成或不需要执行")
            return True
        
        # 获取当前数据统计
        result = db.execute(text("""
            SELECT status, processing_status, COUNT(*) as count
            FROM documents 
            GROUP BY status, processing_status
            ORDER BY status, processing_status
        """))
        
        current_data = result.fetchall()
        logger.info("当前数据状态分布：")
        for row in current_data:
            logger.info(f"  status='{row[0]}', processing_status='{row[1]}', count={row[2]}")
        
        # 执行状态合并迁移
        migrations = [
            # processing_status="pending" + status="active" -> status="uploaded"
            {
                "condition": "processing_status = 'pending' AND status = 'active'",
                "new_status": "uploaded",
                "description": "pending + active -> uploaded"
            },
            # processing_status="processing" -> status="processing"
            {
                "condition": "processing_status = 'processing'",
                "new_status": "processing",
                "description": "processing -> processing"
            },
            # processing_status="completed" + status="active" -> status="processed"
            {
                "condition": "processing_status = 'completed' AND status = 'active'",
                "new_status": "processed",
                "description": "completed + active -> processed"
            },
            # processing_status="failed" -> status="failed"
            {
                "condition": "processing_status = 'failed'",
                "new_status": "failed",
                "description": "failed -> failed"
            },
            # processing_status="vectorizing" -> status="vectorizing"
            {
                "condition": "processing_status = 'vectorizing'",
                "new_status": "vectorizing",
                "description": "vectorizing -> vectorizing"
            },
            # processing_status="vectorized" -> status="vectorized"
            {
                "condition": "processing_status = 'vectorized'",
                "new_status": "vectorized",
                "description": "vectorized -> vectorized"
            },
            # status="vectorized" + processing_status="completed" -> 保持 status="vectorized"
            {
                "condition": "status = 'vectorized' AND processing_status = 'completed'",
                "new_status": "vectorized",
                "description": "vectorized + completed -> vectorized (保持不变)"
            }
        ]
        
        total_updated = 0
        for migration in migrations:
            # 检查符合条件的记录数
            count_result = db.execute(text(f"""
                SELECT COUNT(*) FROM documents WHERE {migration['condition']}
            """))
            count = count_result.scalar()
            
            if count > 0:
                # 执行更新
                update_result = db.execute(text(f"""
                    UPDATE documents 
                    SET status = :new_status 
                    WHERE {migration['condition']}
                """), {"new_status": migration["new_status"]})
                
                updated_count = update_result.rowcount
                total_updated += updated_count
                logger.info(f"✅ {migration['description']}: 更新了 {updated_count} 条记录")
            else:
                logger.info(f"⏭️  {migration['description']}: 没有符合条件的记录")
        
        # 处理其他未映射的状态组合
        unmapped_result = db.execute(text("""
            SELECT status, processing_status, COUNT(*) as count
            FROM documents 
            WHERE NOT (
                (processing_status = 'pending' AND status = 'active') OR
                (processing_status = 'processing') OR
                (processing_status = 'completed' AND status = 'active') OR
                (processing_status = 'failed') OR
                (processing_status = 'vectorizing') OR
                (processing_status = 'vectorized') OR
                (status = 'vectorized' AND processing_status = 'completed') OR
                (status = 'archived') OR
                (status = 'deleted')
            )
            GROUP BY status, processing_status
        """))
        
        unmapped_data = unmapped_result.fetchall()
        if unmapped_data:
            logger.warning("发现未映射的状态组合：")
            for row in unmapped_data:
                logger.warning(f"  status='{row[0]}', processing_status='{row[1]}', count={row[2]}")
                # 对于未映射的组合，保持原有 status，只是记录警告
        
        # 提交事务
        db.commit()
        logger.info(f"状态迁移完成，总共更新了 {total_updated} 条记录")
        
        # 显示迁移后的状态分布
        result = db.execute(text("""
            SELECT status, COUNT(*) as count
            FROM documents 
            GROUP BY status
            ORDER BY status
        """))
        
        final_data = result.fetchall()
        logger.info("迁移后状态分布：")
        for row in final_data:
            logger.info(f"  status='{row[0]}', count={row[1]}")
        
        return True
        
    except Exception as e:
        logger.error(f"迁移过程中发生错误: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def drop_processing_status_column():
    """
    删除 processing_status 字段（可选，需要谨慎执行）
    """
    db = SessionLocal()
    try:
        logger.info("准备删除 processing_status 字段...")
        
        # 检查字段是否存在
        result = db.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'documents' AND column_name = 'processing_status'
        """))
        
        if not result.fetchone():
            logger.info("processing_status 字段不存在，无需删除")
            return True
        
        # 删除字段
        db.execute(text("ALTER TABLE documents DROP COLUMN processing_status"))
        db.commit()
        
        logger.info("✅ processing_status 字段已成功删除")
        return True
        
    except Exception as e:
        logger.error(f"删除字段时发生错误: {e}")
        db.rollback()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="文档状态字段迁移脚本")
    parser.add_argument("--drop-column", action="store_true", 
                       help="迁移完成后删除 processing_status 字段（谨慎使用）")
    parser.add_argument("--dry-run", action="store_true", 
                       help="仅显示当前状态分布，不执行迁移")
    
    args = parser.parse_args()
    
    if args.dry_run:
        # 仅显示当前状态
        db = SessionLocal()
        try:
            result = db.execute(text("""
                SELECT status, processing_status, COUNT(*) as count
                FROM documents 
                GROUP BY status, processing_status
                ORDER BY status, processing_status
            """))
            
            current_data = result.fetchall()
            print("当前数据状态分布：")
            for row in current_data:
                print(f"  status='{row[0]}', processing_status='{row[1]}', count={row[2]}")
        finally:
            db.close()
    else:
        # 执行迁移
        success = migrate_status_fields()
        
        if success and args.drop_column:
            print("\n⚠️  准备删除 processing_status 字段...")
            confirm = input("这是不可逆操作，确认删除？(yes/no): ")
            if confirm.lower() == 'yes':
                drop_success = drop_processing_status_column()
                if drop_success:
                    print("✅ 迁移和字段删除完成")
                else:
                    print("❌ 字段删除失败")
            else:
                print("已取消字段删除操作")
        
        if success:
            print("✅ 状态字段迁移成功完成")
            sys.exit(0)
        else:
            print("❌ 状态字段迁移失败")
            sys.exit(1)
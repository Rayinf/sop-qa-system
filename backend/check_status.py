#!/usr/bin/env python3
import os
import sys
sys.path.append('.')

from app.database import get_db
from app.models.document import Document
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# 创建数据库连接
engine = create_engine(os.getenv('DATABASE_URL', 'sqlite:///./data/sop_qa.db'))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

try:
    # 查询所有文档
    docs = db.query(Document).all()
    print('当前文档状态分布:')
    print('-' * 80)
    
    for doc in docs:
        print(f'ID: {doc.id[:8]}... | 标题: {doc.title[:20]}... | 状态: {doc.status} | 创建时间: {doc.created_at}')
    
    print('-' * 80)
    print(f'总计文档数量: {len(docs)}')
    
    # 统计各状态的文档数量
    status_count = {}
    for doc in docs:
        status = doc.status
        status_count[status] = status_count.get(status, 0) + 1
    
    print('\n状态统计:')
    for status, count in status_count.items():
        print(f'  {status}: {count} 个文档')
        
finally:
    db.close()
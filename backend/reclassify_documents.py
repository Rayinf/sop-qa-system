#!/usr/bin/env python3
"""
重新分类向量数据库中的文档
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from app.services.vector_service import VectorService
from app.services.document_processor import DocumentProcessor
from collections import Counter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def reclassify_documents():
    """重新分类文档"""
    logger.info("=== 开始重新分类文档 ===")
    
    try:
        vector_service = VectorService.get_instance()
        document_processor = DocumentProcessor()
        
        if not vector_service.vector_store:
            logger.error("❌ 向量数据库未加载")
            return
            
        # 获取所有文档
        all_docs = vector_service.vector_store.similarity_search("", k=1000)
        logger.info(f"📄 总共找到 {len(all_docs)} 个文档")
        
        if not all_docs:
            logger.warning("❌ 向量数据库中没有文档")
            return
            
        # 统计当前分类
        current_categories = Counter(doc.metadata.get('category', 'unknown') for doc in all_docs)
        logger.info("📊 当前分类统计:")
        for category, count in current_categories.most_common():
            logger.info(f"  - {category}: {count} 个文档")
            
        # 重新分类文档
        reclassified_count = 0
        category_changes = Counter()
        
        for i, doc in enumerate(all_docs):
            current_category = doc.metadata.get('category', 'unknown')
            title = doc.metadata.get('title', '')
            content = doc.page_content
            
            # 只重新分类unknown或通用文档的文档
            if current_category in ['unknown', '通用文档', 'other']:
                # 使用关键词分类
                full_text = f"标题: {title}\n内容: {content[:1000]}"
                new_category = document_processor._keyword_classify(full_text)
                
                if new_category != current_category:
                    logger.info(f"📝 文档 {i+1}: '{title[:50]}...' {current_category} -> {new_category}")
                    
                    # 更新文档元数据
                    doc.metadata['category'] = new_category
                    reclassified_count += 1
                    category_changes[f"{current_category}->{new_category}"] += 1
                    
        logger.info(f"\n✅ 重新分类完成: {reclassified_count} 个文档")
        
        if reclassified_count > 0:
            logger.info("📈 分类变更统计:")
            for change, count in category_changes.most_common():
                logger.info(f"  - {change}: {count} 个文档")
                
            # 保存更新后的向量数据库
            logger.info("💾 保存更新后的向量数据库...")
            vector_service.save_vector_store(vector_service.vector_store)
            logger.info("✅ 向量数据库保存成功")
            
            # 统计新的分类
            new_categories = Counter(doc.metadata.get('category', 'unknown') for doc in all_docs)
            logger.info("\n📊 新的分类统计:")
            for category, count in new_categories.most_common():
                logger.info(f"  - {category}: {count} 个文档")
        else:
            logger.info("ℹ️  没有文档需要重新分类")
            
    except Exception as e:
        logger.error(f"❌ 重新分类失败: {e}")
        import traceback
        traceback.print_exc()

def test_keyword_classification():
    """测试关键词分类功能"""
    logger.info("\n=== 测试关键词分类功能 ===")
    
    try:
        document_processor = DocumentProcessor()
        
        test_cases = [
            ("质量手册 QMS-01 V1.0", "质量手册内容，管理制度，规范标准"),
            ("BGA返修作业指导书", "操作程序，工作流程，作业指导"),
            ("软件开发规范", "开发程序，技术文档，编程相关"),
            ("检查记录表", "记录表单，检查清单")
        ]
        
        for title, content in test_cases:
            full_text = f"标题: {title}\n内容: {content}"
            category = document_processor._keyword_classify(full_text)
            logger.info(f"🧪 测试: '{title}' -> {category}")
            
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_keyword_classification()
    print("\n" + "="*50 + "\n")
    reclassify_documents()
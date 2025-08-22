"""
简单的embedding实现
"""
import hashlib
import numpy as np
from typing import List
from langchain.embeddings.base import Embeddings

class SimpleEmbeddings(Embeddings):
    """简单的embedding实现，用于测试"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        return [self.embed_query(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        # 使用文本的hash值生成伪随机向量
        hash_obj = hashlib.md5(text.encode('utf-8'))
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # 设置随机种子以确保相同文本产生相同向量
        np.random.seed(hash_int % (2**32))
        
        # 生成随机向量并归一化
        vector = np.random.normal(0, 1, self.dimension)
        vector = vector / np.linalg.norm(vector)
        
        return vector.tolist()
from typing import Optional, Dict, Any
import logging
import os
from langchain_openai import ChatOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """LLM服务类，提供统一的语言模型接口"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.current_model = model_name or settings.deepseek_model
        self.llm = self._create_llm(self.current_model)
        
    def _create_llm(self, model_name: str, model_config: Optional[Dict[str, Any]] = None) -> ChatOpenAI:
        """创建LLM实例"""
        try:
            # 根据模型名称选择配置
            if model_name.startswith('kimi'):
                # Kimi模型配置
                config = {
                    "openai_api_key": os.getenv('KIMI_API_KEY', settings.kimi_api_key),
                    "openai_api_base": settings.kimi_base_url,
                    "model_name": model_name,
                    "temperature": settings.kimi_temperature,
                    "max_tokens": settings.kimi_max_tokens
                }
            else:
                # DeepSeek模型配置（默认）
                config = {
                    "openai_api_key": os.getenv('DEEPSEEK_API_KEY', settings.deepseek_api_key),
                    "openai_api_base": settings.deepseek_base_url,
                    "model_name": model_name,
                    "temperature": settings.deepseek_temperature,
                    "max_tokens": settings.deepseek_max_tokens
                }
            
            # 如果提供了模型配置，则覆盖默认配置
            if model_config:
                config.update(model_config)
            
            # 创建并返回ChatOpenAI实例
            llm = ChatOpenAI(**config)
            logger.info(f"LLM实例创建成功: {model_name}")
            return llm
            
        except Exception as e:
            logger.error(f"创建LLM实例失败: {e}")
            raise
        

    
    def get_llm(self) -> ChatOpenAI:
        """获取LLM实例"""
        return self.llm
    
    def switch_model(self, model_name: str, model_config: Optional[Dict[str, Any]] = None) -> bool:
        """切换模型"""
        try:
            available_models = settings.available_models.split(',')
            if model_name not in available_models:
                logger.error(f"模型 {model_name} 不在可用模型列表中: {available_models}")
                return False
            
            self.current_model = model_name
            self.llm = self._create_llm(model_name, model_config)
            logger.info(f"已切换到模型: {model_name}")
            return True
        except Exception as e:
            logger.error(f"切换模型失败: {e}")
            return False
    
    def get_current_model(self) -> str:
        """获取当前使用的模型"""
        return self.current_model
    
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        try:
            response = self.llm.predict(prompt, **kwargs)
            return response
        except Exception as e:
            logger.error(f"生成文本失败: {e}")
            return ""
    
    def generate_with_messages(self, messages: list, **kwargs) -> str:
        """使用消息列表生成文本"""
        try:
            response = self.llm.predict_messages(messages, **kwargs)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"生成文本失败: {e}")
            return ""
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """生成响应（别名方法）"""
        return self.generate(prompt, **kwargs)
    
    def generate_answer(self, context: str, question: str, **kwargs) -> str:
        """基于上下文生成答案"""
        prompt = f"基于以下上下文回答问题：\n\n上下文：{context}\n\n问题：{question}\n\n答案："
        return self.generate(prompt, **kwargs)
from typing import Optional, List, Dict, Any
import logging
import os
from pathlib import Path
from openai import OpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

class KimiFileService:
    """Kimi文件服务类，处理文件上传和内容提取"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv('KIMI_API_KEY', settings.kimi_api_key),
            base_url=settings.kimi_base_url
        )
        
    def upload_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """上传文件到Kimi API"""
        try:
            file_object = self.client.files.create(
                file=Path(file_path), 
                purpose="file-extract"
            )
            
            logger.info(f"文件上传成功: {file_path}, file_id: {file_object.id}")
            return {
                "id": file_object.id,  # 统一使用id字段
                "name": file_object.filename,  # 统一使用name字段
                "file_id": file_object.id,  # 保留兼容性
                "filename": file_object.filename,  # 保留兼容性
                "bytes": file_object.bytes,
                "created_at": file_object.created_at,
                "status": file_object.status
            }
            
        except Exception as e:
            logger.error(f"文件上传失败: {e}")
            return None
    
    def get_file_content(self, file_id: str) -> Optional[str]:
        """获取文件内容"""
        try:
            file_content = self.client.files.content(file_id=file_id).text
            logger.info(f"文件内容获取成功: {file_id}")
            return file_content
            
        except Exception as e:
            logger.error(f"获取文件内容失败: {e}")
            return None
    
    def upload_and_extract(self, file_path: str) -> Optional[Dict[str, Any]]:
        """上传文件并提取内容"""
        try:
            # 上传文件
            file_info = self.upload_file(file_path)
            if not file_info:
                return None
            
            # 获取文件内容
            file_content = self.get_file_content(file_info["file_id"])
            if not file_content:
                return None
            
            return {
                "file_info": file_info,
                "content": file_content
            }
            
        except Exception as e:
            logger.error(f"上传并提取文件失败: {e}")
            return None
    
    def list_files(self) -> List[Dict[str, Any]]:
        """列出已上传的文件"""
        try:
            file_list = self.client.files.list()
            files = []
            
            for file in file_list.data:
                files.append({
                    "id": file.id,  # 统一使用id字段
                    "name": file.filename,  # 统一使用name字段
                    "file_id": file.id,  # 保留兼容性
                    "filename": file.filename,  # 保留兼容性
                    "bytes": file.bytes,
                    "created_at": file.created_at,
                    "status": file.status,
                    "status_details": file.status_details
                })
            
            logger.info(f"获取文件列表成功，共 {len(files)} 个文件")
            return files
            
        except Exception as e:
            logger.error(f"获取文件列表失败: {e}")
            return []
    
    def delete_file(self, file_id: str) -> bool:
        """删除文件"""
        try:
            self.client.files.delete(file_id=file_id)
            logger.info(f"文件删除成功: {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除文件失败: {e}")
            return False
    
    def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """获取文件信息"""
        try:
            file_info = self.client.files.retrieve(file_id=file_id)
            
            return {
                "file_id": file_info.id,
                "filename": file_info.filename,
                "bytes": file_info.bytes,
                "created_at": file_info.created_at,
                "status": file_info.status,
                "status_details": file_info.status_details
            }
            
        except Exception as e:
            logger.error(f"获取文件信息失败: {e}")
            return None
    
    def upload_multiple_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """批量上传文件"""
        results = []
        
        for file_path in file_paths:
            result = self.upload_and_extract(file_path)
            if result:
                results.append(result)
            else:
                logger.warning(f"文件上传失败: {file_path}")
        
        logger.info(f"批量上传完成，成功 {len(results)}/{len(file_paths)} 个文件")
        return results
    
    def create_file_messages(self, file_ids: List[str]) -> List[Dict[str, str]]:
        """为文件ID列表创建消息格式"""
        messages = []
        
        for file_id in file_ids:
            content = self.get_file_content(file_id)
            if content:
                messages.append({
                    "role": "system",
                    "content": content
                })
        
        return messages
#!/usr/bin/env python3
"""
应用程序启动脚本
"""

import os
import sys
import logging
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.core.init_db import init_db

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    主函数
    """
    try:
        # 加载环境变量
        load_dotenv()
        
        # 检查环境变量
        logger.info("检查环境配置...")
        
        required_env_vars = [
            "DATABASE_URL",
            "SECRET_KEY",
            "DEEPSEEK_API_KEY",
            "DASHSCOPE_API_KEY"
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"缺少必需的环境变量: {', '.join(missing_vars)}")
            logger.error("请检查 .env 文件或环境变量配置")
            sys.exit(1)
        
        # 初始化数据库
        logger.info("初始化数据库...")
        init_db()
        
        # 启动应用
        logger.info(f"启动应用服务器...")
        logger.info(f"服务器地址: http://0.0.0.0:8000")
        logger.info(f"API文档: http://0.0.0.0:8000/docs")
        logger.info(f"调试模式: {settings.debug}")
        
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=settings.debug,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("应用程序被用户中断")
    except Exception as e:
        logger.error(f"应用程序启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
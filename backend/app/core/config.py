from pydantic_settings import BaseSettings
from typing import List, Optional, Dict
import os
from pathlib import Path


class Settings(BaseSettings):
    # Application
    app_name: str = "langchain知识库问答系统"
    app_version: str = "1.0.0"
    debug: bool = True
    environment: str = "development"
    
    # Additional global configs for backward compatibility
    PROJECT_NAME: str = "langchain知识库问答系统"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database Configuration
    database_url: str = "postgresql://postgres:password@localhost:5432/sop_qa_db"
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "sop_qa_db"
    database_user: str = "postgres"
    database_password: str = "password"
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Security Configuration
    secret_key: str = "your-super-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # Superuser
    first_superuser_email: str = "admin@example.com"
    first_superuser_password: str = "admin123456"
    
    # LLM Provider (DeepSeek/OpenAI/Kimi)
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    deepseek_temperature: float = 0.3
    deepseek_max_tokens: int = 2000
    
    # Kimi API Configuration
    kimi_api_key: Optional[str] = "sk-oGiE4vyB6H21XOdVpsPiW1FdM3TnR0ERvCFqbKq3pK2Gc6sR"
    kimi_base_url: str = "https://api.moonshot.cn/v1"
    kimi_model: str = "kimi-k2-0711-preview"
    kimi_temperature: float = 0.6
    kimi_max_tokens: int = 2000
    
    # Embedding Configuration
    embedding_mode: str = "api"  # "api" or "local"
    
    # Embedding API Configuration (Qwen3)
    embedding_api_key: Optional[str] = None  # 从环境变量 DASHSCOPE_API_KEY 读取
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model_name: str = "text-embedding-v4"
    embedding_dimensions: int = 1024
    embedding_encoding_format: str = "float"
    
    # Local Embedding Configuration
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_embedding_device: str = "cpu"  # "cpu" or "cuda"
    
    # Available Models
    available_models: str = "deepseek-chat,deepseek-reasoner,kimi-k2-0711-preview"
    default_model: str = "deepseek-chat"
    
    # LLM Configuration (for backward compatibility)
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.3
    max_tokens: int = 2000
    
    # Vector Store Configuration
    vector_store_path: str = "../data/vectors"
    embedding_model: str = "text-embedding-v4"
    vector_dimension: int = 1024
    
    # Document Processing
    upload_dir: str = "../data/documents"
    max_file_size: str = "1GB"
    max_file_size_mb: int = 1024  # Maximum upload file size in MB (1GB)
    allowed_extensions: List[str] = ["pdf", "docx", "txt", "md", "xlsx", "xls"]
    
    # Auto Vectorization Configuration
    auto_vectorize: bool = True
    
    # Text Splitting Configuration
    chunk_size: int = 4000  # 进一步增加分块大小以保持完整信息
    chunk_overlap: int = 600  # 进一步增加重叠以确保上下文连续性
    
    # Retrieval Configuration
    similarity_threshold: float = 0.0  # 相似度阈值，完全放开过滤
    category_search_multiplier: int = 10  # 类别搜索时的k值乘数
    category_check_multiplier: int = 15  # 类别检查时的k值乘数
    filter_search_multiplier: int = 10  # 过滤搜索时的k值乘数
    
    # 统一检索配置
    retrieval_mode: str = "auto"  # vector, hybrid, multi_query, ensemble, auto
    retrieval_k: int = 10  # 检索文档数量
    retrieval_similarity_threshold: float = 0.0  # 完全放开相似度阈值
    retrieval_enable_cache: bool = True
    
    # 向量检索配置
    vector_use_mmr: bool = True
    vector_mmr_fetch_k: int = 20
    vector_mmr_lambda_mult: float = 0.5
    # 新增：向量检索的类别加权与手动降权默认配置
    vector_category_weight_mode: str = "weight"  # 可选: "filter" | "weight"
    vector_category_primary_boost: float = 1.25
    vector_category_mapped_boost: float = 1.1
    vector_category_mismatch_penalty: float = 0.9
    vector_manual_downweight_keywords: Optional[Dict[str, float]] = None
    
    # 混合检索配置
    hybrid_dense_weight: float = 0.7
    hybrid_sparse_weight: float = 0.3
    hybrid_use_reranking: bool = True
    hybrid_fusion_method: str = "rrf"
    hybrid_rrf_constant: int = 60
    
    # 多查询检索配置
    multi_query_num_queries: int = 3
    multi_query_docs_per_query: int = 8
    multi_query_enable_deduplication: bool = True
    multi_query_max_workers: int = 3
    
    # 集成检索配置
    ensemble_weights: List[float] = [0.4, 0.3, 0.3]
    ensemble_fusion_method: str = "weighted_sum"
    ensemble_enable_score_normalization: bool = True
    ensemble_min_score_threshold: float = 0.0
    ensemble_max_workers: int = 3
    
    # 自动选择配置
    auto_query_length_threshold: int = 10
    auto_prefer_hybrid_for_short: bool = True
    auto_prefer_multi_query_for_complex: bool = False
    auto_prefer_ensemble_for_important: bool = False  # 默认不使用ensemble，只有手动勾选才使用
    
    # CORS Configuration
    allowed_origins: List[str] = ["*"]
    allowed_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allowed_headers: List[str] = ["*"]
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    
    @property
    def database_url_sync(self) -> str:
        """同步数据库连接URL"""
        return self.database_url
    
    @property
    def database_url_async(self) -> str:
        """异步数据库连接URL"""
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://")
    
    @property
    def upload_path(self) -> Path:
        """上传文件路径"""
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def vector_path(self) -> Path:
        """向量存储路径"""
        path = Path(self.vector_store_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    class Config:
        env_file = "../.env"  # .env文件在项目根目录
        env_file_encoding = "utf-8"
        case_sensitive = False

# 创建全局设置实例
settings = Settings()

# 验证必要的配置
def validate_settings():
    """验证必要的配置项"""
    if not settings.openai_api_key and not settings.deepseek_api_key:
        raise ValueError("必须配置 OpenAI API Key 或 DeepSeek API Key")
    
    if not settings.secret_key or settings.secret_key == "your-super-secret-key-change-this-in-production":
        if settings.environment == "production":
            raise ValueError("生产环境必须设置安全的 SECRET_KEY")
    
    return True
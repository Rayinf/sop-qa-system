# LangChain知识库问答系统

基于LangChain和FastAPI的智能知识库问答系统，支持文档上传、向量化存储和智能检索问答。

## 🚀 功能特性

### 核心功能
- **智能问答**: 基于LangChain的检索增强生成(RAG)技术
- **文档管理**: 支持PDF、Word、TXT、Markdown、Excel等多种格式文档上传和处理
- **向量搜索**: 使用FAISS进行高效的向量相似度搜索
- **用户管理**: 完整的用户认证、授权和角色管理系统
- **多种检索模式**: 支持向量检索、混合检索、多查询检索等多种检索策略

### 技术特性
- **前后端分离**: React + FastAPI架构
- **容器化部署**: 完整的Docker和docker-compose配置
- **缓存优化**: Redis缓存提升系统性能
- **多模型支持**: 支持DeepSeek Chat和Qwen3 Embedding
- **灵活配置**: 丰富的配置选项，支持本地和API模式

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │     Nginx       │    │    Backend      │
│   (React)       │◄──►│  (Load Balancer)│◄──►│   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐             │
                       │     FAISS       │◄────────────┤
                       │  (Vector Store) │             │
                       └─────────────────┘             │
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│   PostgreSQL    │◄───│     Redis       │◄────────────┤
│   (Database)    │    │    (Cache)      │             │
└─────────────────┘    └─────────────────┘             │
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│   DeepSeek API  │◄───│   Qwen3 API     │◄────────────┘
│   (Chat Model)  │    │  (Embeddings)   │
└─────────────────┘    └─────────────────┘
```

## 📋 系统要求

### 最低配置
- **CPU**: 2核心
- **内存**: 4GB RAM
- **存储**: 20GB 可用空间
- **操作系统**: Linux/macOS/Windows

### 推荐配置
- **CPU**: 4核心或更多
- **内存**: 8GB RAM或更多
- **存储**: 50GB SSD
- **网络**: 稳定的互联网连接(用于API调用)

### 软件依赖
- Python 3.11+
- Node.js 18+
- PostgreSQL 13+
- Redis 6+
- Docker 20.10+ (可选)
- Docker Compose 2.0+ (可选)

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/your-username/sop-qa-system.git
cd sop-qa-system
```

### 2. 环境配置
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量文件
vim .env
```

**重要**: 请确保设置以下关键配置：
- `DEEPSEEK_API_KEY`: DeepSeek API密钥
- `EMBEDDING_API_KEY`: Qwen3 Embedding API密钥
- `SECRET_KEY`: JWT密钥(生产环境请使用强密码)
- 数据库密码等敏感信息

### 3. 本地开发环境

#### 后端启动
```bash
cd backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 启动后端服务
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 前端启动
```bash
cd frontend

# 安装依赖
npm install

# 启动前端服务
npm run dev
```

### 4. 使用Docker部署
```bash
# 启动开发环境
docker-compose -f docker-dev.yml up -d

# 或启动生产环境
docker-compose up -d
```

### 5. 访问系统
- **前端应用**: http://localhost:3000 (开发) 或 http://localhost (生产)
- **后端API**: http://localhost:8000
- **API文档**: http://localhost:8000/docs

## 🔧 开发指南

### 项目结构
```
sop-qa-system/
├── backend/                 # 后端代码
│   ├── app/
│   │   ├── api/            # API路由
│   │   ├── core/           # 核心配置
│   │   ├── models/         # 数据模型
│   │   ├── services/       # 业务逻辑
│   │   └── utils/          # 工具函数
│   ├── data/               # 数据存储目录
│   ├── docs/               # 文档
│   ├── scripts/            # 脚本文件
│   └── requirements.txt    # Python依赖
├── frontend/               # 前端代码
│   ├── src/
│   │   ├── components/     # React组件
│   │   ├── pages/          # 页面组件
│   │   ├── services/       # API服务
│   │   └── types/          # TypeScript类型
│   └── package.json        # Node.js依赖
├── data/                   # 数据目录
├── nginx/                  # Nginx配置
├── monitoring/             # 监控配置
├── docker-compose.yml      # 生产环境配置
├── docker-dev.yml          # 开发环境配置
└── Makefile               # 便捷命令
```

### 环境变量说明

#### 数据库配置
```bash
DATABASE_URL=postgresql://postgres:postgres123@localhost:5432/sop_qa_db
REDIS_URL=redis://:redis123@localhost:6379/0
```

#### API配置
```bash
# DeepSeek Chat API
DEEPSEEK_API_KEY=your-deepseek-api-key

# Qwen3 Embedding API
EMBEDDING_API_KEY=your-qwen3-api-key
```

#### 应用配置
```bash
SECRET_KEY=your-secret-key-change-in-production
ENVIRONMENT=development
DEBUG=true
```

### API文档

系统启动后，可以通过以下地址访问API文档：
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 检索模式说明

系统支持多种检索模式：

1. **向量检索 (Vector)**: 基于语义相似度的检索
2. **混合检索 (Hybrid)**: 结合密集向量和稀疏向量的检索
3. **多查询检索 (Multi-Query)**: 生成多个查询进行检索
4. **集成检索 (Ensemble)**: 多种检索方法的集成
5. **自动模式 (Auto)**: 根据查询自动选择最佳检索策略

## 🚀 生产部署

### 1. 环境准备
```bash
# 克隆代码到生产服务器
git clone https://github.com/your-username/sop-qa-system.git
cd sop-qa-system

# 配置生产环境变量
cp .env.example .env
vim .env  # 设置生产环境配置
```

### 2. SSL证书配置
```bash
# 将SSL证书放置到nginx/ssl目录
mkdir -p nginx/ssl
cp your-cert.pem nginx/ssl/cert.pem
cp your-key.pem nginx/ssl/key.pem

# 启用HTTPS配置
vim nginx/conf.d/default.conf  # 取消HTTPS配置的注释
```

### 3. 部署应用
```bash
# 使用Docker Compose部署
docker-compose up -d

# 或使用Make命令
make deploy
```

### 4. 验证部署
```bash
# 检查服务状态
docker-compose ps

# 查看日志
docker-compose logs

# 检查健康状态
curl http://localhost/api/v1/health
```

## 📊 监控和维护

### 日志管理
```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f backend
docker-compose logs -f frontend
```

### 数据备份
```bash
# 备份数据库
docker-compose exec postgres pg_dump -U postgres sop_qa_db > backup.sql

# 备份向量数据
tar -czf vectors_backup.tar.gz data/vectors/
```

## 🔧 常见问题

### Q: 服务启动失败
A: 检查端口占用和环境变量配置
```bash
# 检查端口占用
lsof -i :3000
lsof -i :8000

# 检查环境变量
cat .env
```

### Q: 数据库连接失败
A: 确保数据库服务正常运行
```bash
# 检查数据库状态
docker-compose ps postgres

# 查看数据库日志
docker-compose logs postgres
```

### Q: API调用失败
A: 检查API密钥和网络连接
```bash
# 检查DeepSeek API
curl -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
     https://api.deepseek.com/v1/models

# 检查Qwen3 API
curl -H "Authorization: Bearer $EMBEDDING_API_KEY" \
     https://dashscope.aliyuncs.com/compatible-mode/v1/models
```

### Q: 向量数据库加载失败
A: 检查向量数据目录和权限
```bash
# 检查向量数据目录
ls -la data/vectors/

# 重建向量数据库
cd backend
python rebuild_vector_db.py
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

### 代码规范
- 后端: 遵循PEP 8规范，使用black和flake8
- 前端: 遵循ESLint和Prettier配置
- 提交信息: 使用conventional commits格式

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 支持

如果您遇到问题或需要帮助：

1. 查看 [常见问题](#常见问题) 部分
2. 搜索现有的 [Issues](../../issues)
3. 创建新的 [Issue](../../issues/new)

## 🙏 致谢

感谢以下开源项目：
- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://langchain.com/)
- [React](https://reactjs.org/)
- [Ant Design](https://ant.design/)
- [PostgreSQL](https://www.postgresql.org/)
- [Redis](https://redis.io/)
- [FAISS](https://faiss.ai/)
- [DeepSeek](https://www.deepseek.com/)
- [Qwen](https://qwen.aliyun.com/)
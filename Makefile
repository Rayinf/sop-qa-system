.PHONY: help build up down logs clean dev-up dev-down dev-logs restart

# 默认目标
help:
	@echo "SOP Q&A System - Docker Management"
	@echo ""
	@echo "Available commands:"
	@echo "  make build      - 构建所有Docker镜像"
	@echo "  make up         - 启动完整生产环境"
	@echo "  make down       - 停止所有服务"
	@echo "  make logs       - 查看所有服务日志"
	@echo "  make clean      - 清理所有容器和卷"
	@echo "  make dev-up     - 启动开发环境"
	@echo "  make dev-down   - 停止开发环境"
	@echo "  make dev-logs   - 查看开发环境日志"
	@echo "  make restart    - 重启所有服务"
	@echo "  make backend    - 只启动后端服务"
	@echo "  make frontend   - 只启动前端服务"
	@echo "  make db         - 只启动数据库服务"
	@echo "  make migrate    - 运行数据库迁移"
	@echo "  make test       - 运行测试"
	@echo "  make lint       - 运行代码检查"

# 生产环境命令
build:
	@echo "构建Docker镜像..."
	docker-compose build --no-cache

up:
	@echo "启动完整生产环境..."
	docker-compose up -d
	@echo "服务启动完成！"
	@echo "前端访问地址: http://localhost"
	@echo "后端API地址: http://localhost/api"
	@echo "Grafana监控: http://localhost:3001 (admin/admin123)"
	@echo "Prometheus: http://localhost:9090"

down:
	@echo "停止所有服务..."
	docker-compose down

logs:
	@echo "查看所有服务日志..."
	docker-compose logs -f

clean:
	@echo "清理所有容器、网络和卷..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	restart: down up

# 开发环境命令
dev-up:
	@echo "启动开发环境..."
	docker-compose -f docker-dev.yml up -d
	@echo "开发环境启动完成！"
	@echo "前端访问地址: http://localhost:3000"
	@echo "后端API地址: http://localhost:8000"

dev-down:
	@echo "停止开发环境..."
	docker-compose -f docker-dev.yml down

dev-logs:
	@echo "查看开发环境日志..."
	docker-compose -f docker-dev.yml logs -f

dev-clean:
	@echo "清理开发环境..."
	docker-compose -f docker-dev.yml down -v --remove-orphans

# 单独服务命令
backend:
	@echo "启动后端相关服务..."
	docker-compose up -d postgres redis elasticsearch backend

frontend:
	@echo "启动前端服务..."
	docker-compose up -d frontend

db:
	@echo "启动数据库服务..."
	docker-compose up -d postgres redis elasticsearch

# 数据库操作
migrate:
	@echo "运行数据库迁移..."
	docker-compose exec backend alembic upgrade head

migrate-dev:
	@echo "运行开发环境数据库迁移..."
	docker-compose -f docker-dev.yml exec backend alembic upgrade head

# 测试和检查
test:
	@echo "运行后端测试..."
	docker-compose exec backend python -m pytest tests/ -v

test-frontend:
	@echo "运行前端测试..."
	docker-compose exec frontend npm test

lint:
	@echo "运行后端代码检查..."
	docker-compose exec backend flake8 app/
	docker-compose exec backend black --check app/
	docker-compose exec backend isort --check-only app/

lint-frontend:
	@echo "运行前端代码检查..."
	docker-compose exec frontend npm run lint

# 实用工具
shell-backend:
	@echo "进入后端容器shell..."
	docker-compose exec backend bash

shell-frontend:
	@echo "进入前端容器shell..."
	docker-compose exec frontend sh

shell-db:
	@echo "进入数据库shell..."
	docker-compose exec postgres psql -U postgres -d sop_qa_db

# 监控和调试
status:
	@echo "查看服务状态..."
	docker-compose ps

stats:
	@echo "查看资源使用情况..."
	docker stats

# 备份和恢复
backup-db:
	@echo "备份数据库..."
	mkdir -p backups
	docker-compose exec postgres pg_dump -U postgres sop_qa_db > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql

restore-db:
	@echo "恢复数据库 (需要指定备份文件: make restore-db BACKUP=backup_file.sql)"
	@if [ -z "$(BACKUP)" ]; then echo "请指定备份文件: make restore-db BACKUP=backup_file.sql"; exit 1; fi
	docker-compose exec -T postgres psql -U postgres -d sop_qa_db < backups/$(BACKUP)

# 环境设置
setup:
	@echo "初始化项目环境..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "已创建.env文件，请根据需要修改配置"; fi
	@echo "环境设置完成！"

# 完整部署
deploy: setup build up migrate
	@echo "完整部署完成！"

deploy-dev: setup dev-up migrate-dev
	@echo "开发环境部署完成！"
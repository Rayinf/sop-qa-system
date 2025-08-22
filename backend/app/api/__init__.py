from fastapi import APIRouter
from app.api import auth, documents, qa, settings

api_router = APIRouter()

# 包含所有API路由
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(qa.router, prefix="/qa", tags=["qa"])
api_router.include_router(settings.router, prefix="/settings", tags=["settings"])
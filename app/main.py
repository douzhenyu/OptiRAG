"""FastAPI 应用入口 — 光学科研 RAG 助手"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

from app.config import config
from loguru import logger
from app.api import health, upload, query, documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    logger.info("=" * 60)
    logger.info(f"🔬 {config.app_name} v{config.app_version} 启动中...")
    logger.info(f"🌐 监听: http://{config.host}:{config.port}")
    logger.info(f"📚 API 文档: http://{config.host}:{config.port}/docs")
    logger.info(f"💾 Milvus: {config.milvus_host}:{config.milvus_port}")
    logger.info(f"🧠 LLM: {config.llm_model}")
    logger.info(f"👁️  VLM: {config.vision_model}")
    logger.info("=" * 60)
    yield
    logger.info(f"👋 {config.app_name} 关闭")


app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description="光学科研 RAG 助手 — 多模态文档问答与实验方案设计",
    lifespan=lifespan,
)

# CORS — 生产环境需限制 origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.debug and ["*"] or ["http://localhost:9900"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, tags=["健康检查"])
app.include_router(upload.router, prefix="/api", tags=["文档上传"])
app.include_router(documents.router, prefix="/api", tags=["知识库管理"])
app.include_router(query.router, prefix="/api", tags=["智能问答"])

# 静态文件
static_dir = "static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """返回首页"""
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": f"Welcome to {config.app_name}",
        "version": config.app_version,
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info",
    )

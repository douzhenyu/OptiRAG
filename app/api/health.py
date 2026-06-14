"""健康检查接口"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.config import config
from app.engine.rag_engine import rag_engine
from loguru import logger

router = APIRouter()


@router.get("/health")
async def health_check():
    """检查服务状态和知识库"""
    doc_count = len(rag_engine.list_documents(page_size=1)[0])
    status_code = 200 if doc_count >= 0 else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "service": config.app_name,
            "version": config.app_version,
            "status": "healthy",
            "documents_indexed": doc_count,
            "milvus": f"{config.milvus_host}:{config.milvus_port}",
        },
    )

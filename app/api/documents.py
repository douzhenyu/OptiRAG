"""知识库管理接口"""

from fastapi import APIRouter, HTTPException
from app.engine.rag_engine import rag_engine
from app.models.request import DocumentQuery
from app.models.document import DocumentListResponse
from loguru import logger

router = APIRouter()


@router.get("/documents")
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    keyword: str | None = None,
    file_type: str | None = None,
):
    """获取知识库文档列表"""
    docs, total = rag_engine.list_documents(
        page=page, page_size=page_size,
        keyword=keyword, file_type=file_type,
    )
    return DocumentListResponse(
        total=total,
        page=page,
        page_size=page_size,
        documents=docs,
    )


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """获取文档详情"""
    info = rag_engine.get_document(doc_id)
    if info is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return info


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档及其索引"""
    ok = rag_engine.remove_document(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="文档不存在")
    logger.info(f"文档已删除: {doc_id}")
    return {"status": "deleted", "document_id": doc_id}

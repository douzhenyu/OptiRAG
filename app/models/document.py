"""文档元数据模型"""

from datetime import datetime
from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    """文档信息"""
    document_id: str
    filename: str
    file_type: str
    file_size: int
    chunks: int = 0
    tables: int = 0
    images: int = 0
    formulas: int = 0
    status: str = "indexed"  # "indexing" | "indexed" | "failed"
    created_at: datetime = Field(default_factory=datetime.now)


class IngestResult(BaseModel):
    """文档摄入结果"""
    document_id: str
    filename: str
    status: str
    chunks: int
    tables: int
    images: int
    formulas: int
    error: str | None = None


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    total: int
    page: int
    page_size: int
    documents: list[DocumentInfo]

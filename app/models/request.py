"""请求数据模型"""

import uuid
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求"""
    session_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="会话 ID"
    )
    question: str = Field(..., min_length=1, max_length=5000, description="用户问题")
    reference_docs: list[str] | None = Field(
        default=None,
        description="可选：指定参考的文档 ID 列表"
    )


class DocumentQuery(BaseModel):
    """文档列表查询"""
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    keyword: str | None = None
    file_type: str | None = None

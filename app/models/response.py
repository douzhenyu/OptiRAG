"""响应数据模型"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any


class Source(BaseModel):
    """引用来源"""
    document_name: str
    content_snippet: str
    page: int | None = None
    relevance: float


class ToolCall(BaseModel):
    """工具调用记录"""
    tool: str
    status: str  # "start" | "end" | "error"
    input: dict[str, Any] | None = None
    output: str | None = None


class ChatResponse(BaseModel):
    """对话响应"""
    session_id: str
    answer: str
    sources: list[Source] = []
    tool_calls: list[ToolCall] | None = None


class SessionInfo(BaseModel):
    """会话信息"""
    session_id: str
    message_count: int
    created_at: datetime
    updated_at: datetime

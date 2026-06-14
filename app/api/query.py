"""问答接口 — 同步 + 流式实验方案设计"""

import json
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse
from loguru import logger

from app.models.request import ChatRequest
from app.agent.optical_agent import optical_agent

router = APIRouter()


@router.post("/chat")
async def chat(request: ChatRequest):
    """同步问答 — 适合简单的事实查询"""
    try:
        logger.info(f"[{request.session_id}] 同步问答: {request.question}")
        answer = await optical_agent.query(request.question, request.session_id)
        return {
            "session_id": request.session_id,
            "answer": answer,
        }
    except Exception as e:
        logger.error(f"同步问答失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式实验方案设计 — SSE，Plan-Execute-Replan"""
    logger.info(f"[{request.session_id}] 流式方案设计: {request.question}")

    async def event_generator():
        try:
            async for event in optical_agent.design_experiment(
                request.question, request.session_id
            ):
                yield {
                    "event": "message",
                    "data": json.dumps(event, ensure_ascii=False),
                }
        except Exception as e:
            logger.error(f"流式方案设计失败: {e}")
            yield {
                "event": "message",
                "data": json.dumps(
                    {"type": "error", "message": str(e)},
                    ensure_ascii=False,
                ),
            }

    return EventSourceResponse(event_generator())


@router.get("/chat/session/{session_id}")
async def get_session(session_id: str):
    """获取会话详情"""
    return {
        "session_id": session_id,
        "message": "会话历史由 checkpointer 管理，暂不暴露完整历史",
    }


@router.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    """清空会话"""
    return {"status": "ok", "session_id": session_id}

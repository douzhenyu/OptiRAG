"""API 层集成测试"""

import pytest
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app():
    from app.main import app
    return app


@pytest.mark.asyncio
async def test_health_check(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code in (200, 503)
        data = resp.json()
        assert "service" in data
        assert "documents_indexed" in data


@pytest.mark.asyncio
async def test_list_documents(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert "documents" in data


@pytest.mark.asyncio
@patch("app.agent.optical_agent.optical_agent.query", new_callable=AsyncMock)
@patch("app.engine.rag_engine.rag_engine.query", return_value=[])
async def test_chat_sync(mock_rag, mock_query, app):
    mock_query.return_value = "Z-scan 是一种测量非线性光学特性的实验技术。"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/chat", json={"question": "Z-scan 是什么"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data


@pytest.mark.asyncio
async def test_chat_stream(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/api/chat/stream", json={
            "question": "设计Z-scan实验方案"
        }) as resp:
            assert resp.status_code == 200
            count = 0
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    count += 1
                    if count >= 2:
                        break
            assert count > 0


@pytest.mark.asyncio
async def test_get_document_not_found(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/documents/nonexistent_id")
        assert resp.status_code == 404

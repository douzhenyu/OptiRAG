"""测试配置"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_rag_engine():
    """Mock RAGEngine"""
    with patch("app.engine.rag_engine.rag_engine") as mock:
        mock.query = MagicMock(return_value=[
            {"content": "测试文档内容", "metadata": {"source": "test.pdf"}, "score": 0.95},
        ])
        mock.list_documents = MagicMock(return_value=([], 0))
        mock.get_document = MagicMock(return_value=None)
        mock.remove_document = MagicMock(return_value=True)
        yield mock


@pytest.fixture
def mock_openai():
    """Mock OpenAI client"""
    with patch("openai.OpenAI") as mock:
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"steps": ["步骤1", "步骤2"]}'
        client.chat.completions.create.return_value = mock_response
        mock.return_value = client
        yield mock


@pytest.fixture
def sample_state():
    """测试用状态"""
    return {
        "input": "设计Z-scan实验方案",
        "plan": ["检索Z-scan原理", "查询光源参数", "匹配探测器"],
        "past_steps": [],
        "response": "",
    }

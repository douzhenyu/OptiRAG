"""Agent 层单元测试"""

import pytest
from unittest.mock import patch, MagicMock


class TestPlanner:
    """Planner 节点测试"""

    @pytest.mark.asyncio
    async def test_planner_returns_plan(self, mock_openai, sample_state):
        from app.agent.planner import planner

        result = await planner(sample_state)

        assert "plan" in result
        assert len(result["plan"]) > 0
        assert isinstance(result["plan"][0], str)


    @pytest.mark.asyncio
    async def test_planner_fallback_on_error(self, sample_state):
        from app.agent.planner import planner

        with patch("app.agent.planner.OpenAI") as mock:
            mock.return_value.chat.completions.create.side_effect = Exception("API error")
            result = await planner(sample_state)

        assert "plan" in result
        assert len(result["plan"]) > 0


class TestExecutor:
    """Executor 节点测试"""

    @pytest.mark.asyncio
    async def test_executor_removes_first_plan_step(self, sample_state, mock_openai):
        from app.agent.executor import executor

        original_len = len(sample_state["plan"])
        result = await executor(sample_state)

        assert len(result.get("plan", [])) == original_len - 1
        assert len(result.get("past_steps", [])) == 1


    @pytest.mark.asyncio
    async def test_executor_empty_plan(self, sample_state):
        from app.agent.executor import executor

        sample_state["plan"] = []
        result = await executor(sample_state)
        assert result == {}


class TestReplanner:
    """Replanner 节点测试"""

    @pytest.mark.asyncio
    async def test_replanner_responds_when_no_plan(self, mock_openai, sample_state):
        from app.agent.replanner import replanner

        sample_state["plan"] = []
        result = await replanner(sample_state)

        assert "response" in result
        assert len(result["response"]) > 0


    @pytest.mark.asyncio
    async def test_replanner_forces_respond_at_max_steps(self, mock_openai, sample_state):
        from app.agent.replanner import replanner

        sample_state["past_steps"] = [("step1", "result1")] * 8
        result = await replanner(sample_state)

        assert "response" in result


class TestTools:
    """工具集测试"""

    def test_retrieve_knowledge(self, mock_rag_engine):
        from app.agent.tools import retrieve_knowledge

        result = retrieve_knowledge.invoke({"query": "Z-scan"})
        assert "测试文档内容" in result

    def test_search_specs(self, mock_rag_engine):
        from app.agent.tools import search_specs

        result = search_specs.invoke({"device_name": "TL-WD 650"})
        assert result is not None

    def test_compare_devices(self, mock_rag_engine):
        from app.agent.tools import compare_devices

        result = compare_devices.invoke({"device_a": "A", "device_b": "B"})
        assert result is not None

"""Executor — 执行单个检索/研究步骤"""

from typing import Any
from openai import OpenAI
from loguru import logger

from app.config import config
from app.agent.state import PlanExecuteState
from app.agent.tools import DEFAULT_TOOLS, retrieve_knowledge, search_specs, compare_devices

TOOL_MAP = {
    "retrieve_knowledge": retrieve_knowledge,
    "search_specs": search_specs,
    "compare_devices": compare_devices,
}


EXECUTOR_SYSTEM_PROMPT = """\
你是一位光学实验助手，负责执行单个研究步骤。

可用工具：
- retrieve_knowledge: 语义检索知识库文档
- search_specs: 精确查询设备规格参数
- compare_devices: 对比两台设备

执行规则：
1. 理解当前步骤的目标
2. 选择最合适的工具，传入准确的参数
3. 基于工具返回的真实数据回答
4. 不要编造数据，如果信息不足就诚实说明
5. 返回简明的执行结果
"""


async def executor(state: PlanExecuteState) -> dict[str, Any]:
    """执行节点：执行计划中的下一个步骤"""
    logger.info("=== Executor: 执行步骤 ===")

    plan = state.get("plan", [])
    if not plan:
        logger.info("计划为空，跳过执行")
        return {}

    task = plan[0]
    logger.info(f"当前任务: {task}")

    try:
        client = OpenAI(
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
        )

        tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_knowledge",
                    "description": "从知识库中语义检索相关文档",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "自然语言查询"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_specs",
                    "description": "精确查询设备规格参数",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "device_name": {"type": "string", "description": "设备名称/型号"},
                            "param": {"type": "string", "description": "可选，具体参数名"},
                        },
                        "required": ["device_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_devices",
                    "description": "对比两台设备的参数",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "device_a": {"type": "string"},
                            "device_b": {"type": "string"},
                            "aspect": {"type": "string"},
                        },
                        "required": ["device_a", "device_b"],
                    },
                },
            },
        ]

        response = client.chat.completions.create(
            model=config.llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
                {"role": "user", "content": f"请执行以下任务: {task}"},
            ],
            tools=tools_schema,
            tool_choice="auto",
        )

        choice = response.choices[0].message

        if choice.tool_calls:
            result_parts = []
            for tc in choice.tool_calls:
                tool_name = tc.function.name
                import json
                tool_args = json.loads(tc.function.arguments)

                tool_func = TOOL_MAP.get(tool_name)
                if tool_func:
                    logger.info(f"调用工具: {tool_name}({tool_args})")
                    tool_result = tool_func.invoke(tool_args)
                    result_parts.append(str(tool_result))

            result = "\n".join(result_parts) if result_parts else choice.content or "无结果"
        else:
            result = choice.content or "无结果"

        logger.info(f"步骤执行完成，结果长度: {len(result)}")

        return {
            "plan": plan[1:],
            "past_steps": [(task, result)],
        }

    except Exception as e:
        logger.error(f"执行步骤失败: {e}", exc_info=True)
        return {
            "plan": plan[1:],
            "past_steps": [(task, f"执行失败: {str(e)}")],
        }

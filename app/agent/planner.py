"""Planner — 制定实验方案执行计划"""

from textwrap import dedent
from typing import Any
from openai import OpenAI
from pydantic import BaseModel, Field
from loguru import logger

from app.config import config
from app.agent.state import PlanExecuteState
from app.agent.tools import DEFAULT_TOOLS
from app.agent.utils import format_tools_description


class Plan(BaseModel):
    """计划输出格式"""
    steps: list[str] = Field(
        description="任务步骤列表，步骤应当按顺序执行且逻辑独立"
    )


PLANNER_SYSTEM_PROMPT = dedent("""\
你是一位资深光学实验设计专家，负责将复杂的实验问题分解为可执行的检索和研究步骤。

可用工具（供制定计划时参考）：
{tools_description}

你的职责是制定计划，实际工具调用由 Executor 负责。

计划制定原则：
- 将任务分解为 3-7 个逻辑独立的步骤
- 每个步骤描述要具体，包含要检索的关键词或要对比的参数
- 步骤顺序合理：先查原理→再查设备→匹配选型→安全保障→最终方案
- 如有相关经验文档，优先参考其中的方法

示例输入："设计测量非线性晶体三阶磁化率的 Z-scan 实验方案"
示例输出（步骤）：
1. 检索 Z-scan 实验原理和标准方法
2. 查询可用光源规格参数(波长范围、功率)
3. 根据光源参数匹配合适的探测器和聚焦透镜
4. 检索实验室光路安全规范
5. 综合以上信息生成完整实验方案
""")


async def planner(state: PlanExecuteState) -> dict[str, Any]:
    """规划节点：根据用户需求生成检索和研究计划"""
    logger.info("=== Planner: 制定实验方案计划 ===")

    input_text = state.get("input", "")
    logger.info(f"用户输入: {input_text}")

    tools_description = format_tools_description(DEFAULT_TOOLS)

    prompt = PLANNER_SYSTEM_PROMPT.format(tools_description=tools_description)

    try:
        client = OpenAI(
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
        )
        response = client.chat.completions.create(
            model=config.llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text},
            ],
            response_format={"type": "json_object"},
        )
        import json
        content = response.choices[0].message.content
        plan_data = json.loads(content)
        steps = plan_data.get("steps", [])

        logger.info(f"计划已生成: {len(steps)} 个步骤")
        for i, step in enumerate(steps, 1):
            logger.info(f"  步骤{i}: {step}")

        return {"plan": steps}

    except Exception as e:
        logger.error(f"规划失败: {e}", exc_info=True)
        return {
            "plan": [
                "检索相关实验原理和方法",
                "查询所需设备规格参数",
                "匹配合适的器件型号",
                "生成实验方案",
            ]
        }

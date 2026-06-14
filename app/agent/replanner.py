"""Replanner — 评估进展，决定继续/调整/生成方案"""

from textwrap import dedent
from typing import Any
from openai import OpenAI
from pydantic import BaseModel, Field
from loguru import logger

from app.config import config
from app.agent.state import PlanExecuteState
from app.agent.tools import DEFAULT_TOOLS
from app.agent.utils import format_tools_description


class Act(BaseModel):
    """重新规划决策"""
    action: str = Field(description="'continue' | 'replan' | 'respond'")
    new_steps: list[str] = Field(
        default_factory=list,
        description="如果 action='replan'，提供新步骤列表（替换剩余计划）"
    )
    reason: str = Field(default="", description="决策理由")


REPLANNER_PROMPT = dedent("""\
你是一位光学实验设计审查专家，负责评估当前研究进展并决定下一步。

可用工具：{tools_description}

决策规则（按优先级）：
1. **respond** — 信息充足，立即生成最终实验方案
   - 当已获取了实验原理+设备参数+安全规范等关键信息时
   - 已执行步骤 >= 3 且关键信息齐全
   - 已执行步骤 >= 5（无论结果如何）
2. **continue** — 当前计划合理，继续执行下一步
3. **replan** — 当前计划有严重缺陷（谨慎使用）
   - 新步骤数必须 <= 剩余步骤数
   - 已执行步骤 >= 5 时禁止 replan，只能 respond
   - 优先简化计划，不添加不必要的步骤

口诀："信息足够就出方案，计划合理就继续，不到万不得已不改计划"
""")

RESPONSE_PROMPT = dedent("""\
基于已执行的检索和研究结果，生成一份完整的实验方案。

方案格式要求（Markdown）：
# [实验名称]

## 1. 实验目的
[简述实验目标]

## 2. 实验原理
[基于检索到的原理和方法]

## 3. 所需仪器与材料
| 设备 | 型号 | 关键参数 |
|------|------|----------|
| ... | ... | ... |

## 4. 实验步骤
1. ...
2. ...

## 5. 数据采集与处理
[说明采集参数和数据处理方法]

## 6. 安全注意事项
[基于检索到的安全规范]

## 7. 参考文献
[列出引用来源]

重要：所有内容必须基于实际检索到的数据，严禁编造。如果某部分信息不足，明确标注"待补充"。
""")


async def replanner(state: PlanExecuteState) -> dict[str, Any]:
    """重新规划节点"""
    logger.info("=== Replanner: 评估进展 ===")

    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])
    input_text = state.get("input", "")

    logger.info(f"已执行: {len(past_steps)} 步, 剩余: {len(plan)} 步")

    MAX_STEPS = 8
    if len(past_steps) >= MAX_STEPS:
        logger.warning(f"超过最大步数限制({MAX_STEPS})，强制生成方案")
        return await _generate_response(state)

    try:
        client = OpenAI(
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
        )

        tools_description = format_tools_description(DEFAULT_TOOLS)

        steps_summary = "\n".join([
            f"步骤{i+1}: {step}\n结果摘要: {result[:500]}..."
            for i, (step, result) in enumerate(past_steps)
        ])

        if not plan:
            logger.info("计划已全部执行，生成最终方案")
            return await _generate_response(state)

        response = client.chat.completions.create(
            model=config.llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": REPLANNER_PROMPT.format(tools_description=tools_description)},
                {"role": "user", "content": dedent(f"""\
原始任务: {input_text}
已执行步骤:
{steps_summary}
剩余计划: {', '.join(plan)}
已执行 {len(past_steps)} 步，请优先考虑是否信息已足够生成方案（respond）。""")},
            ],
            response_format={"type": "json_object"},
        )

        import json
        content = response.choices[0].message.content
        act_data = json.loads(content)
        action = act_data.get("action", "continue")
        new_steps = act_data.get("new_steps", [])
        reason = act_data.get("reason", "")

        logger.info(f"决策: {action} — {reason}")

        if action == "respond":
            return await _generate_response(state)

        elif action == "replan":
            if len(past_steps) >= 5:
                logger.warning("已执行>=5步，禁止replan，强制respond")
                return await _generate_response(state)
            if len(new_steps) > len(plan):
                new_steps = new_steps[:len(plan)]
            logger.info(f"调整计划: {len(new_steps)} 个新步骤")
            return {"plan": new_steps} if new_steps else {}

        else:
            return {}

    except Exception as e:
        logger.error(f"Replanner 失败: {e}，继续执行")
        return {}


async def _generate_response(state: PlanExecuteState) -> dict[str, Any]:
    """生成最终实验方案"""
    logger.info("生成最终实验方案...")

    past_steps = state.get("past_steps", [])
    input_text = state.get("input", "")

    execution_history = "\n\n".join([
        f"### 步骤: {step}\n**结果:**\n{result}"
        for step, result in past_steps
    ])

    try:
        client = OpenAI(
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
        )

        response = client.chat.completions.create(
            model=config.llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": RESPONSE_PROMPT},
                {"role": "user", "content": dedent(f"""\
原始任务: {input_text}
检索和研究结果:
{execution_history}
请基于以上信息生成完整的实验方案。""")},
            ],
        )

        final_response = response.choices[0].message.content or ""
        logger.info(f"方案生成完成，长度: {len(final_response)}")
        return {"response": final_response}

    except Exception as e:
        logger.error(f"方案生成失败: {e}")
        fallback = f"""# 实验方案（部分）
## 原始任务
{input_text}
## 已收集的信息
{execution_history}
## 说明
系统在生成完整方案时遇到错误，以上是已收集的信息片段。请基于此手动整理实验方案。
"""
        return {"response": fallback}

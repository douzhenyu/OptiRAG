"""Plan-Execute-Replan 状态定义"""

from typing import Annotated, TypedDict
import operator


class PlanExecuteState(TypedDict):
    """Agent 工作流状态"""
    input: str                              # 用户输入（任务描述）
    plan: list[str]                         # 执行计划（步骤列表）
    past_steps: Annotated[list[tuple[str, str]], operator.add]  # 已执行步骤 [(步骤, 结果)]
    response: str                           # 最终响应/报告

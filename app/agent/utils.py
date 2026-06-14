"""Agent 工具函数"""

from typing import Any


def format_tools_description(tools: list[Any]) -> str:
    """格式化工具列表为提示词描述"""
    lines: list[str] = []
    for tool in tools:
        name = getattr(tool, "name", str(tool))
        desc = getattr(tool, "description", "")
        if name and desc:
            lines.append(f"- **{name}**: {desc}")
    return "\n".join(lines)

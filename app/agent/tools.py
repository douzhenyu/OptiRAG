"""Agent 工具集 — 检索 + 规格查询 + 设备对比"""

import asyncio
from langchain_core.tools import tool
from loguru import logger
from app.engine.rag_engine import rag_engine


@tool
def retrieve_knowledge(query: str) -> str:
    """从知识库中语义检索相关文档（含图表描述、公式、表格）。
    适用于：查找实验原理、设备说明、光路设计参考等需要从文档中获取信息的场景。

    Args:
        query: 自然语言查询，建议包含关键词和上下文
    Returns:
        格式化的检索结果文本
    """
    try:
        results = asyncio.run(rag_engine.query(query))
        if not results:
            return "未在知识库中找到相关信息。"

        parts = []
        for i, r in enumerate(results, 1):
            content = r.get("content", "")
            source = r.get("metadata", {}).get("source", "未知来源")
            score = r.get("score", 0)
            modality = r.get("modality", "text")
            label = {"text": "文本", "image": "图表", "table": "表格", "formula": "公式"}.get(modality, modality)
            parts.append(
                f"【来源 {i} — {label}】来源: {source} (相关度: {score:.2f})\n{content}"
            )

        logger.info(f"检索完成: {len(results)} 条结果")
        return "\n\n---\n\n".join(parts)

    except Exception as e:
        logger.error(f"检索失败: {e}")
        return f"检索时发生错误: {str(e)}"


@tool
def search_specs(device_name: str, param: str | None = None) -> str:
    """精确查询设备规格参数。
    适用于：查找特定型号仪器的技术参数（波长范围、分辨率、透过率等）。

    Args:
        device_name: 设备名称或型号，如 "TL-WD 650"
        param: 可选，具体参数名如 "wavelength_range"，不指定则返回所有参数
    Returns:
        设备规格信息
    """
    query_text = f"{device_name} 规格参数"
    if param:
        query_text += f" {param}"

    try:
        results = asyncio.run(rag_engine.query(query_text, top_k=3))
        if not results:
            return f"未找到 '{device_name}' 的规格信息。"

        parts = [f"## {device_name} 规格参数\n"]
        for r in results:
            parts.append(r.get("content", ""))
        return "\n".join(parts)

    except Exception as e:
        return f"规格查询失败: {str(e)}"


@tool
def compare_devices(device_a: str, device_b: str, aspect: str | None = None) -> str:
    """对比两台设备的参数。
    适用于：选型对比、替代方案评估。

    Args:
        device_a: 第一台设备名称/型号
        device_b: 第二台设备名称/型号
        aspect: 可选，关注的对比维度（如 "波长范围"、"分辨率"）
    Returns:
        对比结果
    """
    query_text = f"对比 {device_a} 和 {device_b}"
    if aspect:
        query_text += f" 在 {aspect} 方面"

    try:
        a_results = asyncio.run(rag_engine.query(f"{device_a} 规格参数", top_k=2))
        b_results = asyncio.run(rag_engine.query(f"{device_b} 规格参数", top_k=2))

        parts = [f"## {device_a} vs {device_b}"]
        if aspect:
            parts.append(f"对比维度: {aspect}")

        parts.append(f"\n### {device_a}")
        parts.extend([r.get("content", "") for r in a_results] if a_results else [f"未找到 {device_a} 的信息"])

        parts.append(f"\n### {device_b}")
        parts.extend([r.get("content", "") for r in b_results] if b_results else [f"未找到 {device_b} 的信息"])

        return "\n".join(parts)

    except Exception as e:
        return f"对比失败: {str(e)}"


# 默认工具集
DEFAULT_TOOLS = [
    retrieve_knowledge,
    search_specs,
    compare_devices,
]

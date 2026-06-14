"""MCP Server 模板 — 对接外部 API

复制此文件作为模板，实现你的自定义工具。
运行: python mcp_servers/example_server.py
"""

from fastmcp import FastMCP

mcp = FastMCP("OpticalExternalTool")


@mcp.tool()
def example_external_query(query: str) -> dict:
    """示例：查询外部光学数据库。

    Args:
        query: 查询关键词
    Returns:
        dict: 查询结果
    """
    return {"result": f"查询 '{query}' 的结果（示例占位）", "status": "ok"}


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8005, path="/mcp")

# MCP Server 扩展

此目录用于放置可选的 MCP (Model Context Protocol) 服务器，供 OpticalAgent 动态加载。

## 如何添加

1. 创建 `xxx_server.py`
2. 实现 FastMCP 工具
3. 在 `.env` 中配置连接信息

## 示例

```bash
# .env
MCP_SERVERS_CONFIG={"my_tool": {"transport": "stdio", "command": "python", "args": ["mcp_servers/my_server.py"]}}
```

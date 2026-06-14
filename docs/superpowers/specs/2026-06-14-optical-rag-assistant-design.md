# 光学科研 RAG 助手 — 设计规格

> 从 SuperBizAgent (AIOps) 改造为光学科研多模态 RAG 助手

## 1. 项目概述

### 1.1 目标

构建一个面向光学科研团队的多模态 RAG 助手，支持：
- **文档问答**：上传设备说明书、规格书、论文、设计文档，自然语言查询
- **实验方案设计**：基于知识库文档自动设计实验方案（Plan-Execute-Replan）
- **设备对比**：精确查询和对比不同器件的规格参数
- **小团队共享**：Web 界面，3-5 人共用，知识库统一管理

### 1.2 关键决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| RAG 引擎 | RAG-Anything (LightRAG) | 原生多模态（图/表/公式），知识图谱混合检索 |
| Agent 模式 | Plan-Execute-Replan | 实验方案设计需要多步推理和动态调整 |
| LLM | DashScope Qwen-Max + Qwen-VL | 无 GPU，云端 API |
| 向量库 | Milvus（维持现有） | 已有基础设施，RAG-Anything 官方支持 |
| 部署 | 无 GPU，本地服务器 | 小团队 Web 共享 |
| MCP | 保留为可选扩展点 | 未来对接外部光学 API/专利检索 |

### 1.3 不要什么

- ❌ 不保留 AIOps 诊断功能
- ❌ 不保留 LangChain/LangGraph 依赖
- ❌ 不需要 GPU 本地推理
- ❌ 不需要多租户/权限管理（当前阶段）
- ❌ 不需要 MCP Server 独立进程（当前阶段，保留扩展口）

---

## 2. 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                    光学科研 RAG 助手                              │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                     Web 前端                               │ │
│  │  文档上传(拖拽+批量) | 智能问答(流式) | 知识库管理           │ │
│  └───────────────────────┬───────────────────────────────────┘ │
│                          │ HTTP + SSE                          │
│  ┌───────────────────────▼───────────────────────────────────┐ │
│  │                    FastAPI 服务层                           │ │
│  │  POST /upload | POST /chat/stream | GET /documents | ...   │ │
│  └───────────────────────┬───────────────────────────────────┘ │
│                          │                                     │
│  ┌───────────────────────▼───────────────────────────────────┐ │
│  │              OpticalAgent (Plan-Execute-Replan)            │ │
│  │  Planner → Executor → Replanner → loop / respond          │ │
│  │  工具: retrieve_knowledge | search_specs | compare_devices │ │
│  │  扩展: [MCP 可选] → 外部 API / 专利检索                     │ │
│  └───────────────────────┬───────────────────────────────────┘ │
│                          │                                     │
│  ┌───────────────────────▼───────────────────────────────────┐ │
│  │                RAG-Anything 引擎                            │ │
│  │  MinerU 解析 → 模态处理器(图/表/公式) → 知识图谱 → 混合检索  │ │
│  └───────────────────────┬───────────────────────────────────┘ │
│                          │                                     │
│  ┌───────────────────────▼───────────────────────────────────┐ │
│  │              存储层                                        │ │
│  │  Milvus(向量+图谱) | uploads/(原始文件) | SQLite(会话)     │ │
│  └───────────────────────────────────────────────────────────┘ │
│                          │                                     │
│  ┌───────────────────────▼───────────────────────────────────┐ │
│  │              云端 LLM API                                  │ │
│  │  Qwen-Max(文本生成) | Qwen-VL(视觉理解) | Embedding v4    │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 文件结构

```
optical_rag_assistant/
├── app/
│   ├── __init__.py
│   ├── main.py                  ✏️ 改造：路由精简
│   ├── config.py                ✏️ 改造：配置项更新
│   ├── api/
│   │   ├── __init__.py
│   │   ├── upload.py            ➕ 多格式文档上传
│   │   ├── query.py             ➕ 问答接口(同步+流式)
│   │   ├── documents.py         ➕ 知识库管理 CRUD
│   │   └── health.py            ✏️ 健康检查项更新
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── optical_agent.py     ➕ Plan-Execute-Replan 编排
│   │   ├── planner.py           ➕ 实验方案规划
│   │   ├── executor.py          ➕ 步骤执行
│   │   ├── replanner.py         ➕ 方案评估与调整
│   │   ├── state.py             ➕ 状态定义
│   │   ├── tools.py             ➕ 检索工具集
│   │   └── utils.py             ➕ 工具函数
│   ├── engine/
│   │   ├── __init__.py
│   │   └── rag_engine.py        ➕ RAG-Anything 封装
│   ├── models/
│   │   ├── __init__.py
│   │   ├── request.py           ✏️ 简化请求模型
│   │   ├── response.py          ✏️ 简化响应模型
│   │   └── document.py          ✏️ 文档元数据模型
│   ├── core/
│   │   ├── __init__.py
│   │   └── milvus_client.py     ✏️ 简化(RAG-Anything 托管)
│   └── utils/
│       ├── __init__.py
│       └── logger.py            ✅ 保留
├── mcp_servers/                 ✏️ 精简为扩展框架
│   ├── README.md
│   └── example_server.py
├── static/                      ✏️ 前端重写
│   ├── index.html
│   ├── css/
│   │   └── styles.css
│   └── js/
│       ├── app.js               → 入口+初始化
│       ├── chat.js              → 对话管理+SSE
│       ├── upload.js            → 文件上传
│       └── markdown.js          → 渲染(含KaTeX/Mermaid)
├── uploads/                     ➕ 文档存储
├── .env                         ✏️ 改造
├── .env.example                 ➕ 模板文件
├── pyproject.toml               ✏️ 依赖更新
├── Makefile                     ✏️ 改造
└── tests/                       ➕ 必须新增
    ├── test_agent.py
    ├── test_engine.py
    └── test_api.py
```

---

## 4. 配置与环境

### 4.1 `.env`

```bash
APP_NAME=OpticalRAG
DEBUG=True
HOST=0.0.0.0
PORT=9900

# DashScope LLM
DASHSCOPE_API_KEY=your-api-key-here
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-max
VISION_MODEL=qwen-vl-max
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=1024

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_DB_NAME=optical_rag

# RAG
RAG_TOP_K=5
CHUNK_MAX_SIZE=800
CHUNK_OVERLAP=100

# 文档处理
ALLOWED_EXTENSIONS=pdf,docx,xlsx,pptx,txt,md,png,jpg,jpeg
MAX_FILE_SIZE_MB=50
UPLOAD_DIR=./uploads
```

### 4.2 `pyproject.toml` 依赖

```toml
[project]
name = "optical-rag-assistant"
version = "2.0.0"
description = "光学科研RAG助手 - 多模态文档问答系统"
requires-python = ">=3.11,<3.14"
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "sse-starlette>=2.1.0",
    "raganything>=1.2.0",
    "openai>=1.10.0",
    "dashscope>=1.14.0",
    "pymilvus>=2.3.5",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "httpx>=0.26.0",
    "aiofiles>=23.2.0",
    "python-multipart>=0.0.6",
    "loguru>=0.7.2",
    "python-dotenv>=1.0.0",
]
```

移除的依赖：`langchain*`、`langgraph*`、`langchain-mcp-adapters`、`fastmcp`、`langchain-milvus`、`langchain-text-splitters`、`langchain-qwq`、`aiohttp`。

---

## 5. API 设计

### 5.1 端点总览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/upload` | 文档上传+自动解析索引 |
| GET | `/api/documents` | 知识库文档列表(分页+筛选) |
| GET | `/api/documents/{id}` | 文档详情 |
| DELETE | `/api/documents/{id}` | 删除文档+索引 |
| POST | `/api/chat` | 同步问答 |
| POST | `/api/chat/stream` | 流式问答+实验设计(SSE) |
| GET | `/api/chat/session/{id}` | 会话历史 |
| DELETE | `/api/chat/session/{id}` | 清空会话 |
| GET | `/api/health` | 健康检查 |

### 5.2 SSE 事件类型

| type | 触发节点 | 前端行为 |
|------|---------|---------|
| `plan` | Planner 完成 | 显示计划卡片 |
| `step_start` | Executor 开始 | 步骤状态→"执行中" |
| `step_complete` | Executor 完成 | 追加步骤结果 |
| `replan` | Replanner | 显示决策(继续/调整/完成) |
| `content` | LLM 流式生成 | 打字机追加文本 |
| `report` | Replanner 产出 | 渲染最终 Markdown |
| `done` | 流程结束 | 关闭 SSE 连接 |
| `error` | 任意异常 | 显示错误信息 |

### 5.3 请求/响应模型

```python
class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    question: str = Field(..., min_length=1, max_length=5000)
    reference_docs: list[str] | None = None  # 可选：参考文档ID

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size: int
    chunks: int
    tables: int
    images: int
    formulas: int
    created_at: datetime

class Source(BaseModel):
    document_name: str
    content_snippet: str
    page: int | None
    relevance: float
```

### 5.4 文档摄入数据流

```
用户上传 → 格式验证 → SHA256去重 → MinerU 2.0 解析
  ├── 文本提取
  ├── 图片分离 → Qwen-VL 生成描述
  ├── 表格识别 → 结构化保留
  └── 公式识别 → LaTeX 保留
→ 知识图谱构建(跨模态关系)
→ Milvus 向量索引 + 原始文件存储
→ 返回 IngestResult
```

---

## 6. Agent 层设计

### 6.1 Plan-Execute-Replan 工作流

复用旧项目 AIOps 的编排骨架，替换工具集和提示词：

```
Planner
  输入: 用户问题 + 知识库上下文
  输出: 执行计划（3-7 步骤）
  工具: retrieve_knowledge 预检索，获取经验文档

Executor
  输入: 当前步骤 + 可用工具
  输出: 步骤执行结果
  工具: retrieve_knowledge | search_specs | compare_devices | [MCP]

Replanner
  输入: 已执行步骤+结果 + 剩余计划
  输出: continue / replan / respond
  规则:
    - 信息充足(>=3步+关键信息) → respond
    - 已执行>=8步 → 强制 respond
    - 剩余步骤不合理 → replan(新步骤数 <= 剩余数)
```

### 6.2 工具集

```python
@tool
def retrieve_knowledge(query: str) -> str:
    """语义检索知识库，返回相关文档片段(含图表描述、公式)"""

@tool
def search_specs(device_name: str, param: str = None) -> str:
    """精确查询设备规格参数，支持按参数名筛选"""

@tool
def compare_devices(device_a: str, device_b: str, aspect: str = None) -> str:
    """对比两台设备的参数"""
```

### 6.3 MCP 扩展点

Agent 初始化时扫描 `mcp_servers/` 配置，异步加载可用 MCP Server。加载失败不阻塞启动，仅记录 warning。

保留 `mcp_servers/example_server.py` 作为模板，供未来对接外部 API（Thorlabs、专利检索等）。

---

## 7. RAG 引擎层

### 7.1 RAGEngine 封装

```python
class RAGEngine:
    """RAG-Anything 封装，管理文档生命周期"""

    async def ingest(file_path: Path) -> IngestResult
    async def ingest_batch(paths: list[Path], on_progress: Callable) -> BatchResult
    async def query(question: str, top_k: int = 5) -> QueryResult
    def list_documents() -> list[DocumentMeta]
    def remove_document(doc_id: str) -> bool
    def get_document_info(doc_id: str) -> DocumentInfo | None
```

### 7.2 处理管线

- 文档解析：MinerU 2.0（PDF/Word/Excel/PPT/图片/MD）
- 图像处理器：ImageModalProcessor → Qwen-VL 生成描述
- 表格处理器：TableModalProcessor → 结构化提取
- 公式处理器：EquationModalProcessor → LaTeX 语义理解
- 知识图谱：双图构建（跨模态关系图 + 文本语义图）
- 混合检索：向量匹配 + 知识图谱导航 + 模态感知排序

---

## 8. 前端设计

### 8.1 布局

```
┌──────────┬───────────────────────────────┐
│ 📁 知识库 │  🤖 对话区                     │
│           │                               │
│ [+ 上传]  │  消息流(用户/助手/计划卡片)      │
│           │                               │
│ 📄 doc1   │  引用来源(点击展开原文)          │
│ 📄 doc2   │                               │
│ 📄 doc3   │  Markdown 渲染(含KaTeX公式     │
│           │   和Mermaid图表)               │
│           │                               │
│           │  ┌─────────────────────────┐  │
│           │  │ 输入问题...          [→]│  │
│           │  └─────────────────────────┘  │
└──────────┴───────────────────────────────┘
```

### 8.2 技术选型

| 层 | 方案 | 说明 |
|----|------|------|
| 框架 | 原生 JS (ES Modules) | 不引入构建工具链 |
| Markdown | marked.js | 基础渲染 |
| 数学公式 | KaTeX | LaTeX 公式渲染 |
| 图表 | Mermaid.js | 光路图/流程图 |
| 代码高亮 | highlight.js | 代码块 |
| 上传 | Fetch + FormData | 拖拽 + 批量 + 进度条 |
| SSE | EventSource API | 流式响应 |

### 8.3 文件拆分

```
static/js/
  app.js       → 初始化、全局状态、导航
  chat.js      → 对话管理、SSE 流解析、消息渲染
  upload.js    → 文件上传、拖拽、进度条、批量
  markdown.js  → Markdown 渲染、KaTeX、Mermaid、代码高亮
```

目标：从旧项目单文件 1692 行降到 ~1200 行（4 个模块）。

---

## 9. 文件变动总览

### 移除 (20+ 文件)

```
app/agent/aiops/          (5 files)  → AIOps 诊断逻辑
app/agent/mcp_client.py              → MCP 客户端管理
app/services/aiops_service.py        → AIOps 服务
app/services/rag_agent_service.py    → 旧 RAG Agent
app/services/vector_*_service.py     (5 files) → 旧向量管线
app/tools/                           (3 files) → 旧工具集
app/api/aiops.py                     → AIOps API
app/api/file.py                      → 旧文件上传
app/api/chat.py                      → 旧对话 API
app/core/llm_factory.py              → 旧 LLM 工厂
mcp_servers/cls_server.py            → MCP CLS
mcp_servers/monitor_server.py        → MCP Monitor
aiops-docs/                          → AIOps 知识库
```

### 改造 (6 文件)

```
app/main.py              → 路由精简、移除 AIOps 相关
app/config.py            → 配置项更新
app/api/health.py        → 健康检查项更新
app/models/request.py    → 简化请求模型
app/models/response.py   → 简化响应模型
app/core/milvus_client.py → Monkey-patch 简化(RAG-Anything 托管)
```

### 新增 (15+ 文件)

```
app/agent/optical_agent.py   → Agent 编排
app/agent/planner.py         → 实验方案规划
app/agent/executor.py        → 步骤执行
app/agent/replanner.py       → 方案评估调整
app/agent/state.py           → 状态定义
app/agent/tools.py           → 检索工具集
app/agent/utils.py           → 工具函数
app/engine/rag_engine.py     → RAG-Anything 封装
app/api/upload.py            → 多格式文档上传
app/api/query.py             → 问答接口
app/api/documents.py         → 知识库管理
app/models/document.py       → 文档元数据模型
mcp_servers/example_server.py → MCP 扩展模板
tests/test_agent.py          → Agent 测试
tests/test_engine.py         → 引擎测试
tests/test_api.py            → API 测试
```

### 保留不变 (2 文件)

```
app/utils/logger.py     → 日志配置
app/utils/__init__.py
```

---

## 10. 风险与缓解

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| RAG-Anything 成熟度不足 | 🟡 中 | 先做功能验证 POC；保留 Milvus 直接访问作为降级路径 |
| MinerU 2.0 解析质量 | 🟡 中 | 对关键文档类型(PDF)做解析质量抽样验证 |
| Qwen-VL 图片描述准确性 | 🟢 低 | 图片原文也保留在 chunk 中，检索时可同时匹配 |
| Milvus Monkey-patch 兼容性 | 🟢 低 | RAG-Anything 自带 Milvus 集成，monkey-patch 可移除 |

---

## 11. 测试策略

| 层级 | 内容 | 工具 |
|------|------|------|
| 单元测试 | Agent 节点(planner/executor/replanner) | pytest + mock |
| 单元测试 | RAGEngine 封装方法 | pytest + mock |
| 集成测试 | API 端点(上传/问答/文档管理) | pytest + httpx |
| 集成测试 | Plan-Execute-Replan 完整流程 | pytest-asyncio |
| 覆盖率目标 | >= 70% | pytest-cov |
```

> 注：旧项目测试目录不存在，新项目必须从零建立测试体系。

---

## 12. 实施阶段建议

| 阶段 | 内容 | 预估 |
|------|------|------|
| Phase 1 | 清理旧代码、搭建新骨架、配置依赖 | 1-2 天 |
| Phase 2 | RAGEngine 封装 + 文档摄入管线 | 3-4 天 |
| Phase 3 | Agent 层(Planner/Executor/Replanner) + 工具集 | 3-4 天 |
| Phase 4 | API 层(upload/query/documents/health) | 2-3 天 |
| Phase 5 | 前端重写(文档上传+对话+渲染) | 3-4 天 |
| Phase 6 | 测试 + MCP 扩展框架 + 文档 | 2-3 天 |
| **总计** | | **14-20 天** |

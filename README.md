# 🔬 光学科研 RAG 助手

> 多模态文档问答 + 实验方案设计，面向光学科研团队

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![RAG-Anything](https://img.shields.io/badge/RAG--Anything-1.2+-orange.svg)](https://github.com/HKUDS/RAG-Anything)

## ✨ 核心特性

- 📚 **多模态知识库** — 支持 PDF / Word / Excel / PPT / 图片 / Markdown 上传
- 🔍 **混合检索** — 向量匹配 + 知识图谱，跨模态语义关联
- 🧪 **实验方案设计** — Plan-Execute-Replan Agent 自动生成实验方案
- 🖼️ **图文表公式** — 光路图、光谱图、规格表、LaTeX 公式全保留
- 💬 **流式对话** — SSE 实时输出，支持计划可视化和步骤追踪
- 🔧 **规格查询** — 精确匹配设备参数，多型号对比

## 🛠️ 技术栈

- **框架**: FastAPI
- **RAG 引擎**: RAG-Anything (LightRAG + MinerU 2.0)
- **Agent**: LangGraph Plan-Execute-Replan
- **LLM**: 阿里云 DashScope (Qwen-Max + Qwen-VL)
- **向量库**: Milvus
- **前端**: 原生 JS + KaTeX + Mermaid

## 🚀 快速开始

### 环境要求

- Python 3.11+
- 阿里云 DashScope API Key ([获取地址](https://dashscope.aliyun.com/))
- Milvus 向量数据库（Docker 或本地安装）

### 安装和启动

```bash
# 1. 克隆项目
git clone <repo_url>
cd optical_rag_assistant

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -e ".[dev]"

# 4. 配置环境变量
cp .env.example .env
vim .env  # 填入 DASHSCOPE_API_KEY

# 5. 启动 Milvus（需要 Docker）
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest

# 6. 启动服务
make dev
```

### 访问服务

- **Web 界面**: http://localhost:9900
- **API 文档**: http://localhost:9900/docs

## 📡 API 接口

| 功能 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 文档上传 | POST | `/api/upload` | 多格式文档上传+自动索引 |
| 知识库 | GET | `/api/documents` | 文档列表（分页+筛选） |
| 文档详情 | GET | `/api/documents/{id}` | 单个文档信息 |
| 删除文档 | DELETE | `/api/documents/{id}` | 删除文档及索引 |
| 同步问答 | POST | `/api/chat` | 一次性返回答案 |
| 流式方案设计 | POST | `/api/chat/stream` | SSE 流式 + Plan-Execute-Replan |
| 会话历史 | GET | `/api/chat/session/{id}` | 查询会话状态 |
| 清空会话 | DELETE | `/api/chat/session/{id}` | 清除会话记录 |
| 健康检查 | GET | `/api/health` | 服务状态 + 知识库统计 |

### 使用示例

```bash
# 上传文档
curl -X POST "http://localhost:9900/api/upload" \
  -F "file=@TL-WD_650_SpecSheet.pdf"

# 同步问答
curl -X POST "http://localhost:9900/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"TL-WD 650 的波长范围和透过率是多少？"}'

# 流式实验方案设计
curl -X POST "http://localhost:9900/api/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"question":"设计测量非线性晶体的 Z-scan 实验方案"}' \
  --no-buffer
```

## 📁 项目结构

```
optical_rag_assistant/
├── app/
│   ├── main.py                     # FastAPI 应用入口
│   ├── config.py                   # 配置管理（环境变量）
│   ├── api/                        # API 路由层
│   │   ├── upload.py               # 文档上传接口
│   │   ├── query.py                # 问答接口（同步+流式）
│   │   ├── documents.py            # 知识库管理 CRUD
│   │   └── health.py               # 健康检查
│   ├── agent/                      # Agent 模块（Plan-Execute-Replan）
│   │   ├── optical_agent.py        # 工作流编排
│   │   ├── planner.py              # 实验方案规划
│   │   ├── executor.py             # 步骤执行
│   │   ├── replanner.py            # 方案评估与调整
│   │   ├── state.py                # 状态定义
│   │   ├── tools.py                # 检索工具集
│   │   └── utils.py                # 工具函数
│   ├── engine/
│   │   └── rag_engine.py           # RAG-Anything 封装
│   ├── models/                     # 数据模型层
│   │   ├── request.py              # 请求模型
│   │   ├── response.py             # 响应模型
│   │   └── document.py             # 文档元数据模型
│   └── utils/
│       └── logger.py               # 日志配置（Loguru）
├── static/                         # Web 前端
│   ├── index.html                  # 主页面
│   ├── css/styles.css              # 样式表
│   └── js/
│       ├── app.js                  # 入口 & 知识库管理
│       ├── chat.js                 # 对话管理 & SSE 流
│       ├── upload.js               # 文件上传
│       └── markdown.js             # Markdown/KaTeX/Mermaid 渲染
├── mcp_servers/                    # MCP 扩展（可选）
│   ├── README.md                   # 扩展说明
│   └── example_server.py           # 示例模板
├── tests/                          # 测试
│   ├── conftest.py                 # 测试配置 & fixtures
│   ├── test_agent.py               # Agent 单元测试
│   └── test_api.py                 # API 集成测试
├── .env.example                    # 环境变量模板
├── pyproject.toml                  # 项目配置 & 依赖
├── Makefile                        # 开发命令
└── README.md
```

## ⚙️ 配置说明

通过 `.env` 文件配置，完整选项见 `.env.example`。

```bash
# ── 必填 ──
DASHSCOPE_API_KEY=your-api-key-here

# ── LLM 模型 ──
LLM_MODEL=qwen-max              # 文本生成
VISION_MODEL=qwen-vl-max        # 图片理解
EMBEDDING_MODEL=text-embedding-v4

# ── RAG-Anything 引擎 ──
RA_PARSER=mineru                # 解析引擎: mineru | docling | paddleocr
RA_DEVICE=cpu                   # Mac 换 mps 更快
RA_QUERY_MODE=hybrid            # hybrid | local | global | naive
RA_ENABLE_IMAGES=True           # 光路图/光谱图
RA_ENABLE_TABLES=True           # 规格参数表
RA_ENABLE_FORMULAS=True         # 光学公式

# ── 分块与检索 ──
CHUNK_SIZE=1200                 # LightRAG 分块 token 数
CHUNK_OVERLAP_SIZE=100          # 分块重叠
EMBEDDING_BATCH_NUM=20          # Embedding 批处理大小
RAG_TOP_K=5                     # 检索返回数
```

## 🧪 实验方案设计

基于 **Plan-Execute-Replan** 模式自动生成实验方案。

### 流程

```
1. Planner   制定计划 → 分解为 3-7 个检索/研究步骤
2. Executor  执行步骤 → 调用 RAG 工具检索文档
3. Replanner 评估结果 → 决定继续/调整/生成方案
4. 输出实验方案 → Markdown 格式（含仪器清单、步骤、安全规范）
```

### SSE 事件类型

| 事件 | 说明 |
|------|------|
| `plan` | 执行计划卡片 |
| `step_complete` | 步骤执行结果 |
| `replan` | 计划调整决策 |
| `report` | 最终实验方案（Markdown） |
| `done` | 流程结束 |
| `error` | 错误信息 |

## 📝 开发指南

```bash
make dev           # 开发模式（热重载）
make start         # 生产模式
make test          # 运行测试 + 覆盖率
make format        # 代码格式化
make lint          # 代码检查
make clean         # 清理临时文件
```

## 🐛 常见问题

### API Key 错误

```bash
cat .env | grep DASHSCOPE_API_KEY
```

### Milvus 连接失败

```bash
docker ps | grep milvus
# 如果没有运行，启动：
docker run -d --name milvus-standalone -p 19530:19530 milvusdb/milvus:latest
```

### RAG-Anything 未安装

如果 `raganything` 包未安装或不可用，系统会自动降级为纯文本模式。多模态能力（图片/表格/公式）需要安装：
```bash
pip install raganything
```

## 📚 参考资源

- [RAG-Anything](https://github.com/HKUDS/RAG-Anything) — 多模态 RAG 框架
- [LightRAG](https://github.com/HKUDS/LightRAG) — 知识图谱 RAG
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Agent 工作流
- [阿里云 DashScope](https://dashscope.aliyun.com/)

## 📄 许可证

MIT License

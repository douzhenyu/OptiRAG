# OptiRAG 面试材料包

> 目标岗位：AI Agent 实习生（Agent 系统开发 + RAG + Python）
> 项目地址：https://github.com/douzhenyu/OptiRAG

---

## 📄 STAR 简历项目

> **OptiRAG — 基于 Plan-Execute-Replan 的多模态 RAG Agent 系统**
>
> - 设计并实现了一个面向光学科研场景的**多模态 RAG Agent 系统**，基于 **LangGraph Plan-Execute-Replan** 架构，集成 **RAG-Anything（LightRAG + MinerU 2.0）** 实现 PDF/Word/图片/表格/公式等 10+ 格式文档的自动解析与跨模态知识图谱构建
> - 使用 **FastAPI + SSE** 构建流式对话服务，支持同步问答与异步实验方案设计双模式；基于 **SqliteSaver** 实现会话持久化，服务重启后对话历史不丢失
> - 基于 **DashScope Qwen-Max + Qwen-VL** 实现文本生成与图表视觉理解双通道，封装 LLM/VLM/Embedding 三层回调函数适配 OpenAI 兼容协议，支持 **Milvus** 向量库混合检索（向量匹配 + 知识图谱导航）
> - 预留 **MCP 协议扩展点**，支持动态加载外部工具服务；前端采用原生 JS ES Modules 拆分为 markdown/chat/upload 四个独立模块，集成 KaTeX 公式渲染 + Mermaid 图表可视化
> - 项目全程使用 **Claude Code** 开发，遵循 TDD 流程，通过 Subagent-Driven Development 分派 22 个独立子代理完成 1674 行 Python + 400 行前端代码，30+ commits 可追溯

---

## 🎤 面试 Q&A

### Q1: 为什么选择 Plan-Execute-Replan 而不是普通的 ReAct？

实验方案设计是开放式推理任务——"设计 Z-scan 实验方案"需要先查原理→查光源参数→匹配探测器→查安全规范→生成方案，每一步结果影响后续方向。ReAct 适合"查规格参数"这类 2 步收敛的封闭问题，而 Plan-Execute-Replan 的 Planner→Executor→Replanner 循环更适合多步探索，Replanner 可以根据中间结果动态调整计划（如"光源波长 350-2500nm，需要增加探测器匹配步骤"）。我们做了 MAX_STEPS=8 的强制上限和已执行>=5 步禁止 replan 的收敛机制，防止无限循环。

### Q2: RAG-Anything 的内部原理是什么？你们怎么接入的？

RAG-Anything 底层是 LightRAG 双图索引（跨模态关系图 + 文本语义图）+ MinerU 2.0 文档解析。我们没有直接传 model name，而是封装了三层回调函数：`_create_llm_func()` 封装 DashScope Qwen-Max 文本生成、`_create_vision_func()` 封装 Qwen-VL 图表理解、`_create_embedding_func()` 封装 text-embedding-v4 向量化，全部适配 OpenAI 兼容协议。这样做的好处是如果将来换模型（比如换成 OpenAI），只需要改回调函数，不碰 RAG-Anything 内部逻辑。

### Q3: 为什么把 MemorySaver 换成 SqliteSaver？

MemorySaver 数据在内存里，服务重启就丢。对于小团队共享场景，用户正在设计实验方案时服务挂了，之前的 Planner→Executor→Replanner 中间状态全没。换成 SqliteSaver 后所有 checkpoint 持久化到 `optical_rag.db`，`GET /api/chat/session/{id}` 可以实时查看执行历史和中间结果。

### Q4: 怎么处理文档上传的重复文件和超大文件？

SHA256 去重 + 大小校验双层防护。上传前先算文件哈希，跟已索引文档的哈希比对，相同就返回已有文档 ID 不重复解析。大小限制通过 `MAX_FILE_SIZE_MB` 环境变量配置，超限直接 400 拒绝。解析失败也不会丢数据——元数据状态标记为 `failed`，用户可以重试。

### Q5: 如果 RAG-Anything 没装上，系统会怎样？

不会崩。`_ensure_raganything()` 里 `ImportError` 被 catch 了，设 `self._raganything = None` 并打 warning，后续 ingest 和 query 都会走降级模式——文档摄入只记录元数据不建向量索引，查询直接返回空列表。这样开发环境可以先跑通 API 和前端，等 RAG-Anything 装好后再开启完整能力。

### Q6: Agent Tools 里 async/sync 调用是怎么处理的？

这是实际开发中踩过的坑。`rag_engine.query()` 是 `async def`，但 LangChain 的 `@tool` 装饰器默认生成同步工具。最初代码直接 `results = rag_engine.query(query)` 返回的是 coroutine 对象而不是列表，导致 `TypeError: 'coroutine' object is not iterable`。修复方式是用 `asyncio.run()` 在同步工具内部包装异步调用，保持 LangChain 工具接口兼容。

### Q7: 说一下整个项目的架构分层？

```
FastAPI 路由层 (api/)
  ├── POST /upload        → 文档上传 + 自动索引
  ├── POST /chat          → 同步问答
  ├── POST /chat/stream   → SSE 流式方案设计
  ├── GET/DELETE /documents → 知识库管理
  └── GET /health         → 健康检查
         │
Agent 编排层 (agent/)
  ├── OpticalAgent.query()        → 简单问答：检索 + LLM 合成
  └── OpticalAgent.design_experiment() → 复杂方案：Plan-Execute-Replan
         │
RAG 引擎层 (engine/)
  └── RAGEngine
       ├── ingest()  → SHA256 去重 → process_document_complete
       ├── query()   → aquery(mode=hybrid/naive)
       └── list/get/remove → 元数据 CRUD
         │
存储层
  ├── Milvus     → 向量索引 + 知识图谱
  ├── SQLite     → 会话持久化 (SqliteSaver)
  └── JSON 文件   → 文档元数据 (.rag_metadata.json)
```

### Q8: 项目还有哪些可以改进的地方？

- SQLite → PostgreSQL：多用户并发场景下 SqliteSaver 有写入锁瓶颈
- MemorySaver 降级 QUERY 路径暂无会话历史
- MCP 集成现在是预留扩展点，可以接入真实的外部光学数据库 API
- 前端可以用 React/Vue 重构，当前是原生 JS 适合快速原型

---

## 💻 核心代码讲解

### 1. RAG-Anything 回调函数封装

文件：`app/engine/rag_engine.py:44-140`

```python
@staticmethod
def _create_llm_func():
    client = OpenAI(api_key=config.dashscope_api_key, base_url=config.dashscope_api_base)

    async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=config.llm_model, messages=messages,
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_tokens", 4096),
        )
        return response.choices[0].message.content or ""
    return llm_func
```

**讲解要点**：用闭包捕获 OpenAI client 和配置 → 返回 async callable → 符合 RAG-Anything 的函数签名要求 `(prompt, system_prompt, history_messages, **kwargs) -> str` → 模型切换只需改 `.env` 里的 `LLM_MODEL`。VLM 和 Embedding 同理，都是返回符合接口的 callable。

### 2. Plan-Execute-Replan 工作流

文件：`app/agent/optical_agent.py:26-51`

```python
workflow = StateGraph(PlanExecuteState)
workflow.add_node("planner", planner)       # 制定计划
workflow.add_node("executor", executor)     # 执行步骤
workflow.add_node("replanner", replanner)   # 评估决策
workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "replanner")

def should_continue(state):
    if state.get("response"):           # 已有最终方案 → 结束
        return END
    if state.get("plan"):               # 还有步骤 → 循环
        return "executor"
    return END

workflow.add_conditional_edges("replanner", should_continue, {
    "executor": "executor", END: END
})
```

**讲解要点**：LangGraph StateGraph 建模三个节点 → `past_steps` 用 `operator.add` 做累加而非覆盖 → replanner 的收敛机制：MAX_STEPS=8 强制 respond、已执行>=5 步禁止 replan → SSE 流式输出 typed events（plan/step_complete/report/done/error）。

### 3. Agent Tools 工具集

文件：`app/agent/tools.py`

三个 LangChain `@tool`：
- `retrieve_knowledge(query)` — 语义检索，返回多模态结果（文本/图表/表格/公式）并标注来源和模态标签
- `search_specs(device_name, param)` — 精确规格查询
- `compare_devices(device_a, device_b, aspect)` — 双设备对比

关键修复：所有工具体内用 `asyncio.run(rag_engine.query(...))` 包装异步调用，解决 LangChain 同步 `@tool` 和 RAG-Anything 异步 `aquery()` 之间的调用不匹配。

---

## 📊 PPT 提示词

### 第 1 页：项目概览

> 标题：OptiRAG — 多模态 RAG Agent 系统
> 内容：解决光学科研场景下多格式文档（PDF/图片/表格/公式）的智能问答和实验方案自动设计。一张架构图展示四层：FastAPI 路由 → Agent 编排（Plan-Execute-Replan） → RAG-Anything 引擎（LightRAG + MinerU 2.0） → Milvus + SQLite 存储。

### 第 2 页：Plan-Execute-Replan Agent 设计

> 展示 Planner/Executor/Replanner 三个节点的职责、数据流和收敛策略。关键数字：MAX_STEPS=8，>=5 步禁止 replan，SSE 流式输出 6 种 typed events。

### 第 3 页：RAG-Anything 集成方案

> 三层回调封装：LLM Func（Qwen-Max 文本生成）、Vision Func（Qwen-VL 图表理解）、Embedding Func（text-embedding-v4 向量化）。全部适配 OpenAI 兼容协议。降级策略：ImportError → 纯文本模式。

### 第 4 页：工程亮点

> SqliteSaver 会话持久化、SHA256 文档去重、async/sync 适配方案、MCP 协议扩展点、前端 ES Modules 拆分 + KaTeX + Mermaid 渲染。

### 第 5 页：开发过程与成果

> Claude Code + Subagent-Driven Development 工作流。22 个子代理并行开发。最终产出：24 个 Python 文件 / 1674 行 / 4 个前端模块 / 30+ commits / 0 语法错误 / 已推送到 GitHub。

---

## ✅ 投递检查表

| 检查项 | 状态 |
|--------|------|
| 简历 STAR 4-5 行命中 JD（Agent 系统 + RAG + Python + Claude Code） | ✅ |
| 项目开源可访问 → github.com/douzhenyu/OptiRAG | ✅ |
| JD 关键词覆盖：Agent Skill/系统开发、RAG、Python、Claude Code、Ubuntu | ✅ |
| 面试 Q&A 覆盖 Agent 架构选择、工程踩坑、降级策略、系统分层 | ✅ |
| 核心代码能口头讲清（LLM 回调封装 + LangGraph 工作流 + async/sync 修复） | ✅ |
| PPT 素材准备（5 页结构化提示词） | ✅ |
| 简历中指标描述（1674 行代码、24 文件、30+ commits、0 错误） | ✅ |

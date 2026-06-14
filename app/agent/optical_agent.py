"""OpticalAgent — Plan-Execute-Replan 工作流编排"""

from typing import AsyncGenerator, Any
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from loguru import logger

from app.config import config
from app.agent.state import PlanExecuteState
from app.agent.planner import planner
from app.agent.executor import executor
from app.agent.replanner import replanner

NODE_PLANNER = "planner"
NODE_EXECUTOR = "executor"
NODE_REPLANNER = "replanner"


class OpticalAgent:
    """光学科研助手 Agent — 知识问答 + 实验方案设计"""

    def __init__(self):
        db_dir = Path(config.sqlite_db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        self.checkpointer = SqliteSaver.from_conn_string(config.sqlite_db_path)
        self.graph = self._build_graph()
        logger.info(f"OpticalAgent (Plan-Execute-Replan) 初始化完成, db={config.sqlite_db_path}")

    def _build_graph(self):
        """构建工作流图"""
        workflow = StateGraph(PlanExecuteState)

        workflow.add_node(NODE_PLANNER, planner)
        workflow.add_node(NODE_EXECUTOR, executor)
        workflow.add_node(NODE_REPLANNER, replanner)

        workflow.set_entry_point(NODE_PLANNER)
        workflow.add_edge(NODE_PLANNER, NODE_EXECUTOR)
        workflow.add_edge(NODE_EXECUTOR, NODE_REPLANNER)

        def should_continue(state: PlanExecuteState) -> str:
            if state.get("response"):
                return END
            if state.get("plan"):
                return NODE_EXECUTOR
            return END

        workflow.add_conditional_edges(
            NODE_REPLANNER,
            should_continue,
            {NODE_EXECUTOR: NODE_EXECUTOR, END: END},
        )

        return workflow.compile(checkpointer=self.checkpointer)

    async def query(self, question: str, session_id: str = "default") -> str:
        """同步问答（简单场景，不走 Plan-Execute-Replan）"""
        from app.agent.tools import retrieve_knowledge
        from openai import OpenAI
        from app.config import config

        context = retrieve_knowledge.invoke({"query": question})

        client = OpenAI(
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
        )
        response = client.chat.completions.create(
            model=config.llm_model,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": "你是一位光学科研助手。请基于提供的参考资料回答问题。如果资料不足以回答，诚实说明。引用来源时标注文档名称。用 Markdown 格式输出。",
                },
                {
                    "role": "user",
                    "content": f"参考资料:\n{context}\n\n问题: {question}",
                },
            ],
        )
        return response.choices[0].message.content or ""

    async def design_experiment(
        self, question: str, session_id: str = "default"
    ) -> AsyncGenerator[dict[str, Any], None]:
        """流式实验方案设计（Plan-Execute-Replan，SSE）"""
        logger.info(f"[{session_id}] 开始实验方案设计: {question}")

        initial_state: PlanExecuteState = {
            "input": question,
            "plan": [],
            "past_steps": [],
            "response": "",
        }

        config_dict = {"configurable": {"thread_id": session_id}}

        try:
            async for event in self.graph.astream(
                input=initial_state,
                config=config_dict,
                stream_mode="updates",
            ):
                for node_name, node_output in event.items():
                    if node_name == NODE_PLANNER:
                        plan = node_output.get("plan", [])
                        yield {
                            "type": "plan",
                            "message": f"实验方案计划已制定，共 {len(plan)} 步",
                            "plan": plan,
                        }

                    elif node_name == NODE_EXECUTOR:
                        past_steps = node_output.get("past_steps", [])
                        plan = node_output.get("plan", [])
                        if past_steps:
                            last_step, last_result = past_steps[-1]
                            yield {
                                "type": "step_complete",
                                "step": last_step,
                                "result": last_result[:1000],
                                "remaining": len(plan),
                            }

                    elif node_name == NODE_REPLANNER:
                        response = node_output.get("response", "")
                        if response:
                            yield {
                                "type": "report",
                                "report": response,
                            }
                        else:
                            plan = node_output.get("plan", [])
                            if plan:
                                yield {
                                    "type": "replan",
                                    "action": "replan",
                                    "new_steps": plan,
                                }

            final_state = self.graph.get_state(config_dict)
            final_response = ""
            if final_state and final_state.values:
                final_response = final_state.values.get("response", "")

            yield {"type": "done", "response": final_response}
            logger.info(f"[{session_id}] 实验方案设计完成")

        except Exception as e:
            logger.error(f"[{session_id}] Agent 执行失败: {e}", exc_info=True)
            yield {"type": "error", "message": str(e)}


# 全局单例
optical_agent = OpticalAgent()

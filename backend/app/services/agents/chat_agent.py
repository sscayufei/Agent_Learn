# 对话 Agent 主循环：由模型提出工具调用，服务端执行后把观察结果送回模型，直到回答或触发停止条件。
# 三个层次不要混淆：LangGraph 控制循环；StructuredTool 声明工具参数；实际工具闭包由 chat 路由注入。
# state 保留完整工具轨迹，发给模型的只是受预算约束的投影；压缩提示上下文不等于删除历史记录。
# 学习顺序：run_stream -> _build_agent_graph -> _graph_model_decision -> _complete_tool_call -> 最终结果。
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.services.agents.base import BaseAgent
from app.services.context_compression import (
    calculate_context_pressure,
    clip_to_token_budget,
    default_context_budget,
    estimate_tokens,
    is_context_overflow_error,
    prompt_cache_model_kwargs,
)
from app.services.skills import SkillRegistry, SkillRegistryError


ConversationTool = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]
ToolErrorKind = Literal[
    "unknown_tool",
    "timeout",
    "rate_limited",
    "transient_network",
    "permission_denied",
    "data_not_found",
    "model_output_invalid",
    "tool_runtime_error",
]


# 模型调用工具时提交的参数模型，StructuredTool 用它生成 JSON Schema。
# message 是原始提问，query 可以是模型为检索改写后的查询；二者保留后可观察查询改写。
# repo_id/conversation_id 标明数据范围，path/start_line/line_count 标明文件读取窗口。
# extra="allow" 允许具体工具使用通用字段之外的参数，因此这里不是严格的业务参数白名单。
# 字段有类型和说明只解决输入结构问题，路径边界、对象归属等检查仍在实际工具函数中完成。
class GenericToolArgs(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: str | None = Field(default=None, description="原始用户消息")
    query: str | None = Field(default=None, description="搜索查询")
    keyword: str | None = Field(default=None, description="关键词过滤")
    limit: int | None = Field(default=None, description="结果数量上限")
    repo_id: str | None = Field(default=None, description="仓库 UUID")
    repo_full_name: str | None = Field(default=None, description="仓库完整名称")
    conversation_id: str | None = Field(default=None, description="会话 UUID")
    session_id: str | None = Field(default=None, description="会话片段 UUID")
    message_id: str | None = Field(default=None, description="需要精确读取的消息 UUID")
    before_message_id: str | None = Field(default=None, description="只读取该消息之前的记录")
    after_message_id: str | None = Field(default=None, description="只读取该消息之后的记录")
    offset: int | None = Field(default=None, description="从消息正文的字符偏移量开始读取")
    max_chars: int | None = Field(default=None, description="本次最多返回的正文字符数")
    include_transcript: bool | None = Field(default=None, description="读取封存会话时是否包含原始 transcript")
    route: str | None = Field(default=None, description="仓库路由或意图")
    path: str | None = Field(default=None, description="相对工作区的文件或目录路径")
    pattern: str | None = Field(default=None, description="文本或 glob 模式")
    start_line: int | None = Field(default=None, description="读取文件时从第几行开始，行号从 1 开始")
    line_count: int | None = Field(default=None, description="最多读取多少行")


# LangGraph 节点间传递的状态字典，而不是数据库模型。
# message/context 是本轮任务输入；tools 是可执行包装对象，不能作为普通 JSON 直接持久化。
# pending_tool_calls 保存模型已提出但尚未执行的调用，observations 保存已执行工具的观察摘要。
# agent_events 给界面和诊断使用，tool_calls 记录执行结果，citations 记录可供回答引用的来源。
# step 统计模型决策轮数，并不等于工具调用次数；同一模型响应可能包含多个工具调用。
# TypedDict 主要帮助静态类型检查；运行时取字段仍大量使用 get/default 处理缺失值。
class ConversationState(TypedDict, total=False):
    message: str  # 本轮原始用户问题
    context: dict[str, Any]  # 服务端注入的历史、记忆、证据与运行选项
    tools: dict[str, StructuredTool]  # 可执行工具包装，留在服务端，不是普通 JSON
    tool_name_map: dict[str, str]  # 供应商兼容名到项目原始工具名的映射
    observations: list[dict[str, Any]]  # 已执行工具的观察摘要，供下一轮决策
    agent_events: list[dict[str, Any]]  # 整轮公开事件轨迹，供界面、封存和诊断
    tool_calls: list[dict[str, Any]]  # 已发生调用及其成功/错误记录
    citations: list[dict[str, Any]]  # 当前累计可追溯来源，不是模型自编链接
    pending_tool_calls: list[dict[str, Any]]  # 模型已提出但尚未执行的调用队列
    stream_events: list[dict[str, Any]]  # 当前图节点新增的待发送事件
    route: str  # 本轮主要业务路径，用于展示与追踪
    step: int  # 模型决策轮数，不是工具总次数
    final_answer: str  # 本轮最终正文，工具执行前可能为空
    stop_reason: str  # 循环为何结束，如最终回答、重复调用或预算耗尽
    completion_status: str  # 运行/完成/阻塞等结果状态
    tool_call_counts: dict[str, int]  # 按工具与参数指纹累计的调用次数
    evaluation_trace: dict[str, Any]  # 评测模式收集的真实证据与调用现场
    model_usage: list[dict[str, Any]]  # 各模型请求返回的用量信息
    prompt_diagnostics: list[dict[str, Any]]  # 每轮提示的预算与估算诊断
    runtime_context_stats: dict[str, Any]  # 当前提示投影/压缩统计
    context_overflow_recovery_attempted: bool  # 本轮是否已用过一次溢出恢复，防止无限重试


class ChatAgent(BaseAgent):
    name = "chat_agent"
    description = "使用原生工具调用的对话 Agent。LLM 选择 LangChain 工具，记忆工具通过 MCP 执行。"
    available_tools = [
        "devflow_search_evidence",
        "devflow_get_thread_context",
        "devflow_read_session_events",
        "workspace.list_files",
        "workspace.read_file",
        "workspace.search_code",
        "repo_health.generate_summary",
        "rag.search_similar_documents",
        "issue_agent.analyze_issue",
        "pr_review_agent.analyze_pr",
        "ci_debug_agent.analyze_workflow_run",
        "report_agent.generate_weekly_report",
        "safety_agent.classify_action_risk",
        "workflow.run_engineering_review",
    ]

    max_steps = 12
    max_repeated_tool_calls = 2
    tool_retry_attempts = 1

    # 构造时允许注入 model，单元测试因此可以使用可控模型，不必请求真实供应商。
    # 没有注入模型但配置了 llm_api_key 时，才创建兼容 OpenAI 协议的 ChatOpenAI 客户端。
    # temperature=0 用于降低随机性，timeout 限制单次请求等待时间，缓存参数由配置函数统一提供。
    # 没有密钥时 self.model 保持 None；run_stream 会选择明确的无模型路径，而不是伪造模型响应。
    def __init__(self, model: ChatOpenAI | None = None) -> None:
        self.model = model
        if self.model is None and settings.llm_api_key:
            self.model = ChatOpenAI(
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                temperature=0,
                timeout=60,
                model_kwargs=prompt_cache_model_kwargs(),
            )

    # 非流式入口复用 run_stream：持续消费事件，只取 event="final" 对应的完整结果。
    # 中间的工具、公开说明和文本片段不作为此接口的返回值，避免调用方重复拼接答案。
    # 如果整个流没有交付 final，则返回明确的空运行/无回答结构，让接口保持基本返回形状。
    # 这样流式与非流式共享同一套激活、工具执行和停止逻辑，减少两种接口行为漂移。
    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        final: dict[str, Any] | None = None
        async for event in self.run_stream(input_data, context):
            if event["event"] == "final":
                final = event["data"]
        return final or {
            "run_id": "",
            "route": "unknown",
            "tool_calls": [],
            "answer": "没有生成最终回答。",
            "citations": [],
        }

    # 一次对话运行的总入口，输入为 message，服务端 context 提供工具、历史、证据和运行选项。
    # 阶段一：复制 context、激活本次 Skill，再把业务函数包装成模型能够调用的工具。
    # 阶段二：建立新的 state；轨迹、引用、调用计数均从本轮重新开始，不能沿用上一次运行的容器。
    # 阶段三：按记忆策略处理精确回忆、最近历史、项目概览等路径，必要时可以在进入自由工具循环前收尾。
    # 阶段四：普通开放问题进入 LangGraph；无模型时仅使用已有证据或明确提示配置不足。
    # 阶段五：统一校验 Skill 执行情形，输出文本 delta，最后给 final 完整答案、引用和诊断。
    # 这里的 run_id 暂留空，由路由在持久化 AgentRun 时赋真实 ID；不要在 UI 把空 ID 当唯一标识。
    async def run_stream(self, input_data: dict[str, Any], context: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        message = str(input_data.get("message") or "")  # 本轮原始提问；后续工具 query 可以改写，但不能覆盖这个任务锚点
        # 复制顶层字典，避免给调用方 context 直接新增本轮字段；嵌套对象仍是共享引用，不是深拷贝。
        runtime_context = dict(context)
        try:
            active_skills = self._activate_skills(message, runtime_context)
        except SkillRegistryError as exc:
            answer = f"Skill 激活失败：{exc}"
            async for chunk in self._stream_answer(answer):
                yield chunk
            yield self._event(
                "final",
                {
                    "run_id": "",
                    "route": "skill_error",
                    "tool_calls": [],
                    "agent_events": [{"type": "skill_activation_error", "error": str(exc)}],
                    "answer": answer,
                    "citations": [],
                    "completion": {"status": "blocked", "reason": "skill_activation_error", "steps": 0, "tool_call_count": 0, "tool_error_count": 0},
                },
            )
            return
        # 完整激活结果只属于这次运行；可用工具声明与实际激活 Skill 是两组不同数据。
        runtime_context["active_skills"] = active_skills
        # 工具执行函数由服务端注入，不从用户文本动态导入 Python 代码；随后才包装给模型看 schema。
        raw_tools: dict[str, ConversationTool] = runtime_context["tools"]
        tools = self._build_langchain_tools(raw_tools, runtime_context)
        # 每次提问新建独立状态：空列表表示本轮尚无观察/引用/待执行调用，不是把持久历史删空。
        # 历史与记忆通过 runtime_context 输入，运行产生的新轨迹再写入这些容器，不能混为同一份数据。
        state: ConversationState = {
            "message": message,
            "context": runtime_context,
            "tools": tools,
            "tool_name_map": self._tool_name_map(raw_tools),
            "observations": [],
            "agent_events": [],
            "tool_calls": [],
            "citations": [],
            "pending_tool_calls": [],
            "stream_events": [],
            "route": "direct_answer",
            "step": 0,  # 本轮尚未向模型做任何决策请求，所以决策计数从 0 开始，不沿用上轮次数
            "final_answer": "",
            "stop_reason": "",  # 运行刚开始还没有停止原因，结束分支会明确填写
            "completion_status": "running",  # 正在执行并不等于成功，最终由回答、工具失败或循环保护路径修改状态
            "tool_call_counts": {},
            "model_usage": [],
            "prompt_diagnostics": [],
            "runtime_context_stats": {},
            "context_overflow_recovery_attempted": False,  # 本轮尚未使用超窗恢复机会，首次溢出后会改为 True 阻止无限恢复
        }
        # 只有评测模式才额外积累评测轨迹，普通对话不必承担同样的采样与序列化开销。
        if context.get("evaluation_mode"):
            state["evaluation_trace"] = self._initial_evaluation_trace(message, context)
        self._append_agent_event(
            state,
            "agent_loop_start",
            {
                "mode": "langgraph_tool_calling_loop",
                "max_steps": self.max_steps,
                "available_tools": list(self._tool_name_map(raw_tools).values()),
            },
        )
        # 一个任务可激活多个 Skill，逐个记录实际版本和激活原因；记录资源摘要但不把完整资源正文重复发给界面。
        for skill in active_skills:
            activation_event = {
                key: value
                for key, value in skill.items()
                if key
                in {
                    "skill_name",
                    "skill_title",
                    "skill_version",
                    "entrypoint",
                    "tools",
                    "workflow_steps",
                    "output_contract",
                    "safety_level",
                    "activation_mode",
                    "activation_reason",
                    "instruction_digest",
                }
            }
            activation_event["resources"] = [
                {key: value for key, value in resource.items() if key != "content"}
                for resource in skill.get("resources") or []
                if isinstance(resource, dict)
            ]
            self._append_agent_event(
                state,
                "skill_activated",
                activation_event,
            )
        # 预检索由上下文装配器完成，模型还没开始自由选择工具；后续 RAG 工具补查可以得到另一批证据。
        prefetched_evidence = context.get("evidence") if isinstance(context.get("evidence"), list) else []
        # “没有启用预检索”与“启用了但没命中”不同：这里看键是否存在，不看列表是否非空。
        pre_retrieval_enabled = "evidence" in context
        reasoning_retrieval_enabled = "rag.search_similar_documents" in raw_tools
        pre_retrieval_sources = [
            {
                "source_id": str(item.get("source_id") or item.get("id") or ""),
                "source_type": str(item.get("source_type") or "evidence"),
                "title": str(item.get("title") or "未命名证据"),
                "score": item.get("score") if isinstance(item.get("score"), (int, float)) else None,
            }
            for item in prefetched_evidence[:8]
            if isinstance(item, dict)
        ]
        self._append_agent_event(
            state,
            "rag_strategy",
            {
                "mode": "hybrid_rag" if pre_retrieval_enabled and reasoning_retrieval_enabled else "tool_calling",
                "pre_retrieval": {
                    "enabled": pre_retrieval_enabled,
                    "evidence_count": len(prefetched_evidence),
                    "query": message,
                    "sources": pre_retrieval_sources,
                },
                "reasoning_retrieval": {
                    "enabled": reasoning_retrieval_enabled,
                    "supports_query_rewrite": reasoning_retrieval_enabled,
                    "supports_multiple_searches": reasoning_retrieval_enabled,
                },
            },
        )

        memory_policy = self._memory_context_policy(message, context)
        if memory_policy:
            self._append_agent_event(state, "context_policy", memory_policy)
        # 刚才的精确内容已在最近历史中时，先使用现有内容，不必为一个简单追问读取整段封存会话。
        if memory_policy and memory_policy.get("mode") == "direct_recent_history":
            state["final_answer"] = await self._direct_answer_from_recent_history(state)
            answer = str(state.get("final_answer") or "").strip()
            if not answer:
                answer = self._local_recent_history_answer(state)
            self._stop_loop(state, "recent_history_available", "completed")
            self._record_skill_validation(state)
            async for chunk in self._stream_answer(answer):
                yield chunk
            yield self._event(
                "final",
                {
                    "run_id": "",
                    "route": "direct_answer",
                    "tool_calls": [],
                    "agent_events": state.get("agent_events") or [],
                    "answer": answer,
                    "citations": [],
                    "completion": self._completion_summary(state),
                    **self._runtime_metadata(state),
                    **self._evaluation_result(state),
                },
            )
            return

        # 明确需要旧原文时，先读正确的历史工具再总结；仅根据摘要猜测旧原句容易遗漏限制。
        required_memory_tool = str((memory_policy or {}).get("tool") or "") or None
        if required_memory_tool and required_memory_tool in raw_tools:
            async for event in self._prefetch_memory_context(state, required_memory_tool):
                yield event
            state["final_answer"] = await self._final_from_observations(state)
            answer = str(state.get("final_answer") or "").strip()
            if not answer:
                answer = await self._fallback_answer(state)
            self._stop_loop(state, "required_memory_context_loaded", self._tool_completion_status(state))
            self._record_skill_validation(state)
            async for chunk in self._stream_answer(answer):
                yield chunk
            yield self._event(
                "final",
                {
                    "run_id": "",
                    "route": state.get("route") or required_memory_tool,
                    "tool_calls": state.get("tool_calls") or [],
                    "agent_events": state.get("agent_events") or [],
                    "answer": answer,
                    "citations": state.get("citations") or [],
                    "completion": self._completion_summary(state),
                    **self._runtime_metadata(state),
                    **self._evaluation_result(state),
                },
            )
            return

        # 项目概览优先准备项目自身上下文再回答，而不是让模型只凭仓库名称推测用途。
        if self._asks_project_overview(message):
            async for event in self._prefetch_project_context(state):
                yield event
            state["final_answer"] = await self._final_from_observations(state)
            answer = str(state.get("final_answer") or "").strip()
            if not answer:
                answer = await self._fallback_answer(state)
            self._stop_loop(state, "project_context_loaded", self._tool_completion_status(state))
            self._record_skill_validation(state)
            async for chunk in self._stream_answer(answer):
                yield chunk
            yield self._event(
                "final",
                {
                    "run_id": "",
                    "route": state.get("route") or "workspace_read_file",
                    "tool_calls": state.get("tool_calls") or [],
                    "agent_events": state.get("agent_events") or [],
                    "answer": answer,
                    "citations": state.get("citations") or [],
                    "completion": self._completion_summary(state),
                    **self._runtime_metadata(state),
                    **self._evaluation_result(state),
                },
            )
            return

        # 没有模型客户端时不能进入自由工具决策循环，只能使用已取得证据或明确提示配置不足。
        if self.model is None:
            answer = self._local_prefetched_evidence_answer(state) or (
                "当前没有取得足够证据。请先确认仓库已连接并完成同步；开放式对话还需要配置 LLM。"
            )
            evidence = (state.get("context") or {}).get("evidence")
            has_evidence = isinstance(evidence, list) and any(isinstance(item, dict) for item in evidence)
            self._stop_loop(
                state,
                "prefetched_evidence_available" if has_evidence else "llm_not_configured",
                "completed" if has_evidence else "blocked",
            )
            self._record_skill_validation(state)
            async for chunk in self._stream_answer(answer):
                yield chunk
            yield self._event(
                "final",
                {
                    "run_id": "",
                    "route": "prefetched_evidence" if has_evidence else "direct_answer",
                    "tool_calls": [],
                    "agent_events": state.get("agent_events") or [],
                    "answer": answer,
                    "citations": self._prefetched_evidence_citations(state),
                    "completion": self._completion_summary(state),
                    **self._runtime_metadata(state),
                    **self._evaluation_result(state),
                },
            )
            return

        # 到这里才进入一般性的“模型选择工具”循环；前面的特定回忆/概览路径可以提前返回。
        graph = self._build_agent_graph(self._tool_model(tools))
        # 图会依次产生模型/工具阶段事件，边运行边向调用方转发，而不是等全部工具结束才展示进度。
        async for event in self._stream_agent_graph(graph, state):
            yield event

        answer = str(state.get("final_answer") or "").strip()
        if not answer:
            answer = await self._fallback_answer(state)
        # 若图结束后仍未登记停止原因，以当前工具结果补一个明确收尾状态，避免 final 仍显示 running。
        if state.get("completion_status") == "running":
            self._stop_loop(state, "fallback_answer_ready", self._tool_completion_status(state))
        self._record_skill_validation(state)
        async for chunk in self._stream_answer(answer):
            yield chunk
        yield self._event(
            "final",
            {
                "run_id": "",
                "route": state.get("route") or "direct_answer",
                "tool_calls": state.get("tool_calls") or [],
                "agent_events": state.get("agent_events") or [],
                "answer": answer,
                "citations": state.get("citations") or [],
                "completion": self._completion_summary(state),
                **self._runtime_metadata(state),
                **self._evaluation_result(state),
            },
        )

    # 把模型决策、工具执行和预算耗尽建成三个节点，再用条件边构成循环。
    # START -> model_decision：每次先让模型选择回答，或提出一个/多个工具调用。
    # model_decision -> tool_action：有 pending 才执行；没有 pending 就结束。
    # tool_action -> tool_action：同轮还有其他调用时继续消费，不必额外请求模型。
    # tool_action -> model_decision：这一批工具执行完后，用新的观察再做一轮决策。
    # tool_action -> max_steps_stop：决策轮数达到预算且仍未得到最终回答时，明确结束。
    # compile 只把图定义变成可执行对象；真正运行发生在后面的 astream。
    def _build_agent_graph(self, tool_model: Any):
        async def model_decision_node(state: ConversationState) -> dict[str, Any]:
            return await self._graph_model_decision(state, tool_model)

        graph = StateGraph(ConversationState)
        graph.add_node("model_decision", model_decision_node)
        graph.add_node("tool_action", self._graph_tool_action)
        graph.add_node("max_steps_stop", self._graph_max_steps_stop)
        graph.add_edge(START, "model_decision")
        graph.add_conditional_edges(
            "model_decision",
            self._route_after_model_decision,
            {"tools": "tool_action", "end": END},
        )
        graph.add_conditional_edges(
            "tool_action",
            self._route_after_tool_action,
            {"tools": "tool_action", "model": "model_decision", "limit": "max_steps_stop"},
        )
        graph.add_edge("max_steps_stop", END)
        # name 便于诊断识别执行图；compile 不会立即发起模型或工具请求。
        return graph.compile(name="chat_agent_tool_calling_graph")

    # 使用 LangGraph debug 流同时观察“任务开始”和“任务完成”两个时点。
    # 工具节点开始时先发 tool_use，界面才能在工具尚未返回时显示运行中，而不是等结束才出现。
    # 任务完成时把节点返回的状态字段合回 current_state 和调用方传入的 state。
    # stream_events 只用于对外转发，不当作下一节点的长期状态字段累积，避免重复发旧事件。
    # recursion_limit 是图运行的防御上限；业务层的 max_steps 仍通过条件路由控制正常收尾。
    async def _stream_agent_graph(self, graph: Any, state: ConversationState) -> AsyncIterator[dict[str, Any]]:
        current_state: ConversationState = dict(state)
        config = {"recursion_limit": self.max_steps * 8 + 20}
        # 逐个消费图事件才能在工具开始时立刻发出预告，在工具完成后再发结果；开始和完成是两个时点。
        async for graph_event in graph.astream(current_state, config=config, stream_mode="debug"):
            if not isinstance(graph_event, dict):
                continue
            payload = graph_event.get("payload") if isinstance(graph_event.get("payload"), dict) else {}
            node_name = str(payload.get("name") or "")
            if graph_event.get("type") == "task" and node_name == "tool_action":
                preview = self._preview_tool_use_event(payload.get("input"))
                if preview:
                    yield self._event("tool_use", preview)
                continue
            # debug 流还包含检查点等内部事件，不能把所有 payload 都当作对外业务事件发送。
            if graph_event.get("type") != "task_result":
                continue
            result = payload.get("result")
            if not isinstance(result, dict):
                continue
            # 将节点实际返回的状态增量回写两份运行状态；stream_events 单独转发，不留在长期状态里重复消费。
            for key, value in result.items():
                if key != "stream_events":
                    current_state[key] = value
                    state[key] = value
            # 一个节点可能产生多条业务事件，逐条按顺序发给接口，不把整个事件数组误当单个 SSE 事件。
            for event in result.get("stream_events") or []:
                if isinstance(event, dict):
                    yield event

    # 模型决策节点：在状态副本中推进 step，组装系统/用户提示，记录输入估算，再调用绑定工具的模型。
    # 正常响应分成两类：没有 tool_calls 就把正文作为最终答案；有调用就保存到 pending 等待执行。
    # 正文与调用同时出现时，正文作为公开说明事件展示；它不是工具已经执行成功的证明。
    # 若供应商报上下文过长，只允许一次 aggressive 提示投影重试；再次失败则把异常交给上层。
    # 相同工具和参数重复超过阈值时，记录 blocked 与指纹，防止无新信息的循环持续消耗预算。
    # 返回值只包含应合并的状态字段与本节点新事件，不替换服务端注入的全部运行上下文。
    async def _graph_model_decision(self, state: ConversationState, tool_model: Any) -> dict[str, Any]:
        next_state: ConversationState = dict(state)
        step = int(next_state.get("step") or 0) + 1  # 进入一次模型决策才加 1；一次响应内提出多次工具调用也只算这一轮决策
        next_state["step"] = step
        self._append_agent_event(next_state, "agent_step", {"step": step, "phase": "model_decision"})
        system_prompt = self._tool_calling_system_prompt()
        user_prompt = self._tool_calling_user_prompt(next_state)
        # 系统规则和本轮用户上下文使用不同角色包装，随后一并交给已绑定工具声明的模型。
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        estimated_prompt_tokens = self._record_prompt_diagnostics(
            next_state,
            phase="model_decision",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        try:
            response = await tool_model.ainvoke(messages)
        except Exception as exc:
            if not is_context_overflow_error(exc) or next_state.get("context_overflow_recovery_attempted"):
                raise
            # 记录已恢复标记，让下一次溢出直接暴露，而不是反复压缩重试耗尽预算。
            next_state["context_overflow_recovery_attempted"] = True
            next_state["runtime_context_stats"] = {
                **(next_state.get("runtime_context_stats") or {}),
                "reactive_compaction": "aggressive_retry",
                "overflow_error": str(exc),
            }
            self._append_agent_event(
                next_state,
                "context_overflow_recovery",
                {"phase": "model_decision", "strategy": "aggressive_runtime_projection"},
            )
            user_prompt = self._tool_calling_user_prompt(next_state, aggressive=True)
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            estimated_prompt_tokens = self._record_prompt_diagnostics(
                next_state,
                phase="model_decision_reactive_retry",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            response = await tool_model.ainvoke(messages)
        self._record_model_usage(
            next_state,
            response,
            phase="model_decision",
            estimated_prompt_tokens=estimated_prompt_tokens,
        )
        # 先取可见正文，再独立检查结构化工具调用；正文出现不代表本轮已经可以结束。
        response_text = self._message_content(response).strip()
        # 从结构化 tool_calls 字段读取意图，不靠解析回答正文里“我要调用...”这样的自然语言来执行。
        raw_tool_calls = getattr(response, "tool_calls", None) or []
        tool_calls = [dict(call) for call in raw_tool_calls if isinstance(call, dict)]
        stream_events: list[dict[str, Any]] = []  # 仅收集本节点新产生的对外事件，避免把历史事件反复重发
        # 同一响应可同时给公开说明和工具调用；这里的 thought 事件是可见响应文本，不是隐藏推理。
        if response_text and tool_calls:
            self._append_agent_event(next_state, "thought", {"content": response_text})
            stream_events.append(self._event("thought", {"content": response_text}))
        # 模型本轮没再请求工具，才按最终回答路径收尾；没有调用不等于一定有足够证据，仍需结合回答与上游约束。
        if not tool_calls:
            next_state["final_answer"] = response_text
            next_state["pending_tool_calls"] = []
            self._stop_loop(next_state, "model_final_answer", "completed")
        else:
            repeated = self._repeated_tool_call(tool_calls, next_state)
            # 重复调用保护命中后不执行这批意图，清空 pending 并给出 blocked 状态，防止相同参数空转。
            if repeated:
                next_state["pending_tool_calls"] = []
                next_state["route"] = "tool_loop_guard"
                next_state["final_answer"] = (
                    "我停止了这轮 Agent Loop，因为模型连续请求相同工具和参数，"
                    "继续执行只会重复消耗预算。请根据已有工具结果调整问题，或让我换一种检索/分析路径。"
                )
                next_state["observations"] = [
                    *(next_state.get("observations") or []),
                    {
                        "tool_name": repeated["tool_name"],
                        "arguments": repeated["arguments"],
                        "observation": {
                            "route": "tool_loop_guard",
                            "answer_preview": "Detected repeated identical tool call and stopped the loop.",
                            "repeat_count": repeated["repeat_count"],
                            "fingerprint": repeated["fingerprint"],
                        },
                    },
                ]
                self._stop_loop(
                    next_state,
                    "repeated_tool_call",
                    "blocked",
                    {
                        "tool_name": repeated["tool_name"],
                        "arguments": repeated["arguments"],
                        "repeat_count": repeated["repeat_count"],
                    },
                )
                return self._graph_state_update(next_state, stream_events)
            # 仅登记待执行意图。真正执行放在工具节点，保持“做决定”和“产生副作用”两个阶段可观察。
            next_state["pending_tool_calls"] = tool_calls
        return self._graph_state_update(next_state, stream_events)

    # 每个节点消费 pending 队首的一次调用，剩余调用原样留给下一次工具节点。
    # _execute_tool_call 负责参数规范、执行与轨迹登记；本节点负责将结果包装成可见事件。
    # tool_result 用来更新同一调用的界面状态，action/observation 是另两种公开事件形态。
    # 当前实现按队列逐个执行，并未因为模型同一响应提出多个调用就自动并发运行它们。
    async def _graph_tool_action(self, state: ConversationState) -> dict[str, Any]:
        next_state: ConversationState = dict(state)
        pending = list(next_state.get("pending_tool_calls") or [])
        if not pending:
            return self._graph_state_update(next_state, [])
        # 一个工具节点只消费队首调用，剩余调用继续留在 pending；同轮多个调用并未在这里 asyncio.gather 并发执行。
        tool_call = pending[0]
        next_state["pending_tool_calls"] = pending[1:]
        public_action, action_input, observation, _tool_message, _tool_use_event, tool_result_event = await self._execute_tool_call(tool_call, next_state)
        stream_events = [
            self._event("tool_result", tool_result_event),
            self._event("action", {"tool_name": public_action, "arguments": action_input}),
            self._event("observation", {"tool_name": public_action, "content": observation}),
        ]
        return self._graph_state_update(next_state, stream_events)

    # 达到步数上限后用已有观察收尾；显式标记停止原因，不能把“没继续执行”伪装成全部完成。
    async def _graph_max_steps_stop(self, state: ConversationState) -> dict[str, Any]:
        next_state: ConversationState = dict(state)
        next_state["route"] = "tool_loop_budget_exhausted"
        next_state["final_answer"] = (
            "这轮 Agent Loop 已达到步骤预算上限。下面的结论只能基于已经完成的工具调用和观察结果，"
            "如果还需要继续，我会从当前证据继续缩小范围。"
        )
        self._stop_loop(next_state, "max_steps_exhausted", "budget_exhausted", {"max_steps": self.max_steps})
        return self._graph_state_update(next_state, [])

    def _route_after_model_decision(self, state: ConversationState) -> str:
        return "tools" if state.get("pending_tool_calls") else "end"

    # 路由优先级：先处理同轮剩余工具，再检查模型决策预算，最后才允许回模型。
    # 例如 step=3 且 pending 还有两条时，会先执行两条；step 不会因每次工具执行而自增。
    # 只有没有剩余调用、达到 max_steps 且没有答案时才选择 limit 路径。
    # 这种顺序保证模型已提出的一组调用不会被误判成额外的模型决策轮。
    def _route_after_tool_action(self, state: ConversationState) -> str:
        if state.get("pending_tool_calls"):
            return "tools"
        if int(state.get("step") or 0) >= self.max_steps and not str(state.get("final_answer") or "").strip():
            return "limit"
        return "model"

    def _graph_state_update(self, state: ConversationState, stream_events: list[dict[str, Any]]) -> dict[str, Any]:
        keys = [
            "observations",
            "agent_events",
            "tool_calls",
            "citations",
            "pending_tool_calls",
            "route",
            "step",
            "final_answer",
            "stop_reason",
            "completion_status",
            "tool_call_counts",
            "evaluation_trace",
            "model_usage",
            "prompt_diagnostics",
            "runtime_context_stats",
            "context_overflow_recovery_attempted",
        ]
        update = {key: state[key] for key in keys if key in state}
        update["stream_events"] = stream_events
        return update

    def _stop_loop(
        self,
        state: ConversationState,
        reason: str,
        completion_status: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state["stop_reason"] = reason
        state["completion_status"] = completion_status
        return self._append_agent_event(
            state,
            "agent_loop_stop",
            {"reason": reason, "completion_status": completion_status, **(extra or {})},
        )

    # 以“规范工具名 + 排序后的参数 JSON”作为一次调用的指纹。
    # 对参数键排序，使 {a:1,b:2} 与 {b:2,a:1} 被视为同一请求，而不会绕过重复检测。
    # 计数写入 state，跨多轮模型决策累计；超过 max_repeated_tool_calls 才返回阻塞信息。
    # 不同 query 或不同文件窗口的补查属于不同指纹，不会因为工具名相同就全部禁止。
    # 它是循环止损机制，不是外部写接口的幂等保证，也不能代替权限检查。
    def _repeated_tool_call(self, tool_calls: list[dict[str, Any]], state: ConversationState) -> dict[str, Any] | None:
        counts = dict(state.get("tool_call_counts") or {})  # 本轮累计调用计数的副本
        # 逐条检查同轮意图，因为模型一次响应可给出多个调用，不能只对第一条做重复保护。
        for tool_call in tool_calls:
            safe_action, public_action, _tool_use_id, action_input = self._tool_call_parts(tool_call, state)
            fingerprint = self._tool_call_fingerprint(safe_action, action_input)
            repeat_count = counts.get(fingerprint, 0) + 1
            counts[fingerprint] = repeat_count
            if repeat_count > self.max_repeated_tool_calls:
                state["tool_call_counts"] = counts
                return {
                    "tool_name": public_action,
                    "arguments": action_input,
                    "fingerprint": fingerprint,
                    "repeat_count": repeat_count,
                }
        state["tool_call_counts"] = counts
        return None

    def _tool_call_fingerprint(self, safe_action: str, action_input: dict[str, Any]) -> str:
        return f"{safe_action}:{json.dumps(action_input, ensure_ascii=False, sort_keys=True, default=str)}"

    def _completion_summary(self, state: ConversationState) -> dict[str, Any]:
        tool_calls = [call for call in state.get("tool_calls") or [] if isinstance(call, dict)]
        error_calls = [call for call in tool_calls if call.get("status") == "error"]
        status = state.get("completion_status") or "completed"
        if status == "completed" and error_calls:
            status = "completed_with_tool_errors"
        return {
            "status": status,
            "reason": state.get("stop_reason") or "model_final_answer",
            "steps": int(state.get("step") or 0),
            "tool_call_count": len(tool_calls),
            "tool_error_count": len(error_calls),
        }

    def _tool_completion_status(self, state: ConversationState) -> str:
        calls = [call for call in state.get("tool_calls") or [] if isinstance(call, dict)]
        if calls and not any(call.get("status") == "success" for call in calls):
            return "blocked"
        return "completed"

    def _preview_tool_use_event(self, raw_state: Any) -> dict[str, Any] | None:
        if not isinstance(raw_state, dict):
            return None
        pending = raw_state.get("pending_tool_calls")
        if not isinstance(pending, list) or not pending or not isinstance(pending[0], dict):
            return None
        safe_action, public_action, tool_use_id, action_input = self._tool_call_parts(pending[0], raw_state)
        return {
            "type": "tool_use",
            "step": raw_state.get("step", 0),
            "id": tool_use_id,
            "tool_name": public_action,
            "arguments": action_input,
            **self._skill_event_metadata(public_action, raw_state),
        }

    async def _execute_tool_call_events(
        self,
        tool_call: dict[str, Any],
        state: ConversationState,
    ) -> AsyncIterator[dict[str, Any]]:
        safe_action, public_action, tool_use_id, action_input, tool_use_event = self._prepare_tool_call(tool_call, state)
        yield self._event("tool_use", tool_use_event)
        observation, _tool_message, tool_result_event = await self._complete_tool_call(
            safe_action,
            public_action,
            tool_use_id,
            action_input,
            state,
        )
        yield self._event("tool_result", tool_result_event)
        yield self._event("action", {"tool_name": public_action, "arguments": action_input})
        yield self._event("observation", {"tool_name": public_action, "content": observation})

    async def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        state: ConversationState,
    ) -> tuple[str, dict[str, Any], dict[str, Any], ToolMessage, dict[str, Any], dict[str, Any]]:
        safe_action, public_action, tool_use_id, action_input, tool_use_event = self._prepare_tool_call(tool_call, state)
        observation, tool_message, tool_result_event = await self._complete_tool_call(
            safe_action,
            public_action,
            tool_use_id,
            action_input,
            state,
        )
        return public_action, action_input, observation, tool_message, tool_use_event, tool_result_event

    def _prepare_tool_call(
        self,
        tool_call: dict[str, Any],
        state: ConversationState,
    ) -> tuple[str, str, str, dict[str, Any], dict[str, Any]]:
        safe_action, public_action, tool_use_id, action_input = self._tool_call_parts(tool_call, state)
        tool_use_event = self._append_agent_event(
            state,
            "tool_use",
            {"id": tool_use_id, "tool_name": public_action, "arguments": action_input, **self._skill_event_metadata(public_action, state)},
        )
        return safe_action, public_action, tool_use_id, action_input, tool_use_event

    # 从模型调用结构中提取工具名、调用 ID 和参数，统一成后续执行需要的形状。
    # safe_action 是发给供应商的兼容名称，public_action 是项目内部带命名空间的原工具名。
    # tool_use_id 把开始事件、结束事件和 ToolMessage 配对，参数则交给规范化函数补上下文。
    # 这一步只解释调用，不执行工具；未知名称会在 _complete_tool_call 里形成结构化错误。
    def _tool_call_parts(
        self,
        tool_call: dict[str, Any],
        state: ConversationState | dict[str, Any],
    ) -> tuple[str, str, str, dict[str, Any]]:
        safe_action = str(tool_call.get("name") or "").strip()
        public_action = self._public_tool_name(safe_action, state)  # type: ignore[arg-type]
        tool_use_id = str(tool_call.get("id") or f"tool_use_{len(state.get('agent_events') or []) + 1}")
        raw_args = tool_call.get("args") if isinstance(tool_call, dict) else {}
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                raw_args = {}
        action_input = self._normalize_action_input(public_action, raw_args if isinstance(raw_args, dict) else {}, state)  # type: ignore[arg-type]
        return safe_action, public_action, tool_use_id, action_input

    # 一次工具执行的完整边界，输入是规范名称、调用 ID、参数及当前状态。
    # 先检查显式 Skill 的允许工具；不允许时生成 permission_denied，而不是仍调用后再隐藏结果。
    # 再检查工具是否真实注册；模型编出的函数名只会成为 unknown_tool，不会被动态导入或执行。
    # 允许的调用进入有限恢复流程，结果统一带调用轨迹、错误类型、尝试次数和 Skill 元数据。
    # 之后采集评测证据、生成公开摘要，把观察、调用记录和引用写回本轮 state。
    # 最后构造带相同 tool_call_id 的 ToolMessage，保证结果能关联到对应的模型请求。
    # 本函数不负责数据库 commit；持久化整轮消息与运行记录仍由 API 层控制。
    async def _complete_tool_call(
        self,
        safe_action: str,
        public_action: str,
        tool_use_id: str,
        action_input: dict[str, Any],
        state: ConversationState,
    ) -> tuple[dict[str, Any], ToolMessage, dict[str, Any]]:
        tools = state.get("tools") or {}
        skill_violation = self._explicit_skill_tool_violation(public_action, state)
        # 先做显式 Skill 工具边界检查；不通过就生成错误观察，下面的真正工具执行分支不会运行。
        if skill_violation:
            error = skill_violation
            error_kind: ToolErrorKind = "permission_denied"
            observation = {
                "route": "tool_error",
                "answer": error,
                "citations": [],
                "tool_calls": [
                    {
                        "tool_name": public_action,
                        "arguments": action_input,
                        "status": "error",
                        "error": error,
                        "error_kind": error_kind,
                        "retryable": False,
                        "attempts": 1,
                        "tool_use_id": tool_use_id,
                    }
                ],
                "error_kind": error_kind,
                "retryable": False,
            }
        # 未知工具返回结构化错误供模型恢复，不能动态执行模型提供的任意函数名。
        elif safe_action not in tools:
            error = f"未知工具：{public_action}。可用工具：{', '.join(state.get('tool_name_map', {}).values())}"
            error_kind: ToolErrorKind = "unknown_tool"
            observation = {
                "route": "tool_error",
                "answer": error,
                "citations": [],
                "tool_calls": [
                    {
                        "tool_name": public_action,
                        "arguments": action_input,
                        "status": "error",
                        "error": error,
                        "error_kind": error_kind,
                        "retryable": False,
                        "attempts": 1,
                        "tool_use_id": tool_use_id,
                    }
                ],
                "error_kind": error_kind,
                "retryable": False,
            }
        else:
            observation = await self._invoke_tool_with_recovery(
                tools[safe_action],
                public_action,
                tool_use_id,
                action_input,
                state,
            )

        # 部分简单工具只返回 answer；为它们补标准成功轨迹，使统一界面和统计都有可用记录。
        if not observation.get("tool_calls"):
            observation["tool_calls"] = [
                {
                    "tool_name": public_action,
                    "arguments": action_input,
                    "status": "success",
                    "tool_use_id": tool_use_id,
                    **self._skill_event_metadata(public_action, state),
                }
            ]
        # 某个业务工具内部可能产生多条子调用轨迹，逐条补调用身份、重试与 Skill 信息，而非只处理第一条。
        for call in observation.get("tool_calls", []):
            if isinstance(call, dict):
                call.setdefault("tool_use_id", tool_use_id)
                if observation.get("attempts") is not None:
                    call.setdefault("attempts", observation.get("attempts"))
                if observation.get("error_kind") is not None:
                    call.setdefault("error_kind", observation.get("error_kind"))
                if observation.get("retryable") is not None:
                    call.setdefault("retryable", observation.get("retryable"))
                call.update(self._skill_event_metadata(str(call.get("tool_name") or public_action), state))
        self._capture_evaluation_observation(
            state,
            tool_name=public_action,
            arguments=action_input,
            observation=observation,
        )
        # 对外观察摘要与工具原始结构分开，避免把无限长日志直接塞入下一轮提示和前端事件。
        summary = self._summarize_observation(observation)
        status = "error" if any(call.get("status") == "error" for call in observation.get("tool_calls", []) if isinstance(call, dict)) else "success"
        tool_result_event = self._append_agent_event(
            state,
            "tool_result",
            {
                "tool_use_id": tool_use_id,
                "tool_name": public_action,
                "status": status,
                "route": observation.get("route"),
                "error_kind": observation.get("error_kind"),
                "retryable": observation.get("retryable"),
                "content": summary,
                **self._skill_event_metadata(public_action, state),
            },
        )
        state["observations"] = [
            *(state.get("observations") or []),
            {"tool_name": public_action, "arguments": action_input, "observation": summary},
        ]
        # 追加到既有调用而非覆盖，最终报告才能包含整轮所有工具；引用也按同样思路累积。
        state["tool_calls"] = [*(state.get("tool_calls") or []), *observation.get("tool_calls", [])]
        state["citations"] = [*(state.get("citations") or []), *observation.get("citations", [])]
        extra_events = observation.get("agent_events")
        if isinstance(extra_events, list):
            state["agent_events"] = [*(state.get("agent_events") or []), *[event for event in extra_events if isinstance(event, dict)]]
        observation_route = str(observation.get("route") or "")
        has_successful_tool = any(call.get("status") == "success" for call in state.get("tool_calls") or [])
        # 后来的工具错误不应抹掉此前已有成功业务路由，但全程失败时需要保留 tool_error 便于诊断。
        if observation_route and (observation_route != "tool_error" or not has_successful_tool):
            state["route"] = observation_route

        # 通过 tool_call_id 关联本次调用的结果；这条关联是工具协议的基本约束。
        tool_message = ToolMessage(
            content=json.dumps(summary, ensure_ascii=False),
            tool_call_id=tool_use_id,
        )
        return summary, tool_message, tool_result_event

    # 执行包装工具并统一处理异常，最多尝试 1 + tool_retry_attempts 次。
    # 正常字典结果保留业务字段；非字典结果包装为可读 answer，调用方不用再猜返回类型。
    # 异常先分类，再由“错误类型 + 动作名称”共同决定能否重试，而非见到异常就重发。
    # 当前可重试类别是超时、限流和临时网络故障；权限错误、目标不存在等应让上层调整输入。
    # 每次重试写 tool_retry 事件，最终失败也返回结构化 observation，使模型知道失败原因。
    # 重试策略不会自动回滚已经发生的外部副作用，所以部分动作明确排除在自动重试之外。
    async def _invoke_tool_with_recovery(
        self,
        tool: StructuredTool,
        public_action: str,
        tool_use_id: str,
        action_input: dict[str, Any],
        state: ConversationState,
    ) -> dict[str, Any]:
        max_attempts = 1 + self.tool_retry_attempts  # 首次调用也算一次尝试
        last_error: Exception | None = None  # 尚未发生异常；只有进入 except 后才保存最后一次错误，用于最终失败说明
        last_error_kind: ToolErrorKind = "tool_runtime_error"  # 尚未分类时的通用错误标签，真实异常发生后会更新为更具体类别
        retryable = False  # 是否重试由错误类型与动作一起决定
        # 循环表示同一次工具调用的有限重试，不是重新规划任务；从 1 起计数，便于记录第几次实际执行。
        for attempt in range(1, max_attempts + 1):
            try:
                observation = await tool.ainvoke(action_input)
                if isinstance(observation, dict):
                    # 工具已有 attempts 时尊重原记录，没有时补当前次数，成功后立即返回而不再继续重试。
                    observation.setdefault("attempts", attempt)
                    return observation
                return {
                    "route": public_action,
                    "answer": str(observation),
                    "citations": [],
                    "tool_calls": [],
                    "attempts": attempt,
                }
            except Exception as exc:
                last_error = exc
                last_error_kind = self._classify_tool_error(exc)
                # 同样是 timeout，只读检索通常可重试，有副作用的动作则要更保守地处理。
                retryable = self._is_retriable_tool_error(last_error_kind, public_action)
                # 达到尝试上限或不可重试时立即退出，避免把权限错误/无数据等问题重试成无意义循环。
                if not retryable or attempt >= max_attempts:
                    break
                self._append_agent_event(
                    state,
                    "tool_retry",
                    {
                        "tool_use_id": tool_use_id,
                        "tool_name": public_action,
                        "attempt": attempt,
                        "next_attempt": attempt + 1,
                        "error_kind": last_error_kind,
                        "error": str(exc),
                    },
                )
        # 走到这里说明循环未成功返回，用最后一个真实错误组织统一失败观察，供模型调整后续动作。
        error_text = str(last_error or "unknown tool failure")
        return {
            "route": "tool_error",
            "answer": f"工具 {public_action} 执行失败：{error_text}",
            "citations": [],
            "tool_calls": [
                {
                    "tool_name": public_action,
                    "arguments": action_input,
                    "status": "error",
                    "error": error_text,
                    "error_kind": last_error_kind,
                    "retryable": retryable,
                    "attempts": max_attempts if retryable else 1,
                    "tool_use_id": tool_use_id,
                }
            ],
            "error_kind": last_error_kind,
            "retryable": retryable,
            "attempts": max_attempts if retryable else 1,
        }

    # 把供应商/网络库的不同异常统一为项目内部错误类别。
    # 先识别真实 TimeoutError，再对异常文本中的状态码和常见词做兼容匹配。
    # 例如 429 -> rate_limited，401/403 -> permission_denied，404 -> data_not_found。
    # 这是一组启发式映射，不是所有 SDK 异常的精确继承树；未命中时保留 tool_runtime_error。
    # 分类的目的是帮助恢复决策与诊断，而不是改变原工具的业务返回值。
    def _classify_tool_error(self, exc: Exception) -> ToolErrorKind:
        if isinstance(exc, TimeoutError):
            return "timeout"
        text = str(exc).lower()
        if any(token in text for token in ["timeout", "timed out", "deadline"]):
            return "timeout"
        if any(token in text for token in ["rate limit", "429", "too many requests"]):
            return "rate_limited"
        if any(token in text for token in ["connection", "temporarily", "unavailable", "reset by peer", "503", "502", "504"]):
            return "transient_network"
        if any(token in text for token in ["permission", "forbidden", "unauthorized", "401", "403"]):
            return "permission_denied"
        if any(token in text for token in ["not found", "missing", "404"]):
            return "data_not_found"
        if any(token in text for token in ["validation", "schema", "json"]):
            return "model_output_invalid"
        return "tool_runtime_error"

    # 是否重试不仅看网络异常，还看工具动作；有副作用的动作不能照搬只读查询的重试策略。
    def _is_retriable_tool_error(self, error_kind: ToolErrorKind, public_action: str) -> bool:
        if error_kind not in {"timeout", "rate_limited", "transient_network"}:
            return False
        if public_action in {"safety_agent.classify_action_risk", "report_agent.generate_weekly_report", "workflow.run_engineering_review"}:
            return False
        return True

    # bind_tools 把当前工具的名称、说明和参数 schema 交给模型，让它输出原生 tool_calls。
    # 传给模型的是工具声明，不是 Python 函数源码；真正调用仍发生在服务端执行器中。
    # 开放式工具选择要求存在模型，因此无模型时这里会抛错；上层提前分流可避免走到此处。
    def _tool_model(self, tools: dict[str, StructuredTool]):
        if self.model is None:
            raise RuntimeError("ChatAgent requires an LLM for open-ended tool selection")
        # 工具函数仍留在服务端；模型收到的是工具声明，并返回调用名称与 JSON 参数。
        return self.model.bind_tools(list(tools.values()))

    def _tool_calling_system_prompt(self) -> str:
        return (
            "你是 DevFlow AI 的对话控制器。这个稳定前缀会被缓存。"
            "请像长任务软件 Agent 一样工作：先规划，调用工具，读取 tool_result 观察结果，修正计划，"
            "持续推进直到用户请求真正完成，或步骤预算耗尽。"
            "当需要仓库、记忆、工作区代码或写操作安全上下文时，使用原生工具调用。"
            "绝不要编造工具结果。"
            "工程结论必须说明 WHY，并区分证据、推断和不确定项；证据不足时直接说缺什么，不要硬猜。"
            "当输出审查结论时，不要用 looks good、没什么问题这类空话代替检查；必须给出风险、级别和下一步。"
            "当回答交接、总结或工作流结论时，尽量保留 What、Why、Tradeoff、Open Questions、Next Action。"
            "当用户询问负责人、已分配人员、责任人或任务分配时，只能从明确的 DevFlow team_members "
            "或工具返回的现有 GitHub assignees 中推荐人员。仓库文档、Issue 文本、代码、示例、模型名、"
            "角色设定、团队比喻和记忆片段都不能当成人，除非它们同时出现在 DevFlow team_members 中。"
            "如果没有配置团队成员，请说明这一点，并要求用户先添加开发者后再分配任务。"
            "当用户请求 GitHub 写入、关闭、打标签、评论、发布、回复或发送时，必须先调用安全工具；"
            "除非存在单独的已确认写入工具，否则只返回草稿。"
            "对于合并就绪度、当前阻塞、优先级规划，或需要 Issue + PR + CI + RAG 综合判断的跨领域工程决策，"
            "优先使用 workflow.run_engineering_review 工具。"
        )

    # 把本轮问题与服务端上下文投影成模型的用户提示，工具预算由 context budget 提供。
    # 历史、证据、团队信息、已激活 Skill 和工具观察按各自规则整理，不直接把 state 整体序列化。
    # aggressive=True 只在溢出恢复时进一步缩短历史/工具投影；state 中的执行事实仍保留。
    # 问题本身是当前任务锚点，不能为了节省 token 只留下检索材料而丢失用户要求。
    # 该函数返回的字符串会真实发给模型；新增教学解释必须写在 # 注释里，不能改提示正文。
    def _tool_calling_user_prompt(self, state: ConversationState, *, aggressive: bool = False) -> str:
        prompt_observations, prompt_events, runtime_stats = self._runtime_tool_context(
            state,
            aggressive=aggressive,
        )
        state["runtime_context_stats"] = {
            **(state.get("runtime_context_stats") or {}),
            **runtime_stats,
        }
        context = state.get("context") or {}
        server_context = self._server_context(context)
        history = self._history_messages(context)
        if aggressive:
            budget = default_context_budget()
            server_context = clip_to_token_budget(server_context, max(512, budget.available_prompt_tokens // 3))
            history = self._compress_history_for_retry(history, max(512, budget.recent_tokens // 2))
        return (
            "上下文使用规则：如果服务端提供的最近消息已经包含回答所需的完整原文，直接基于这些消息回答，"
            "不要为了展示工具调用而重复读取记忆。只有目标消息不在最近窗口、内容被截断或压缩、用户明确要求"
            "核实原话，或者问题跨越会话时，才调用一个最匹配的 MCP 记忆工具。当前会话原始消息使用 "
            "devflow_get_thread_context；封存会话和跨会话摘要使用 devflow_read_session_events；项目长期记忆、"
            "历史决策和仓库证据使用 devflow_search_evidence。询问当前会话原话时，不要额外调用 "
            "devflow_search_evidence。\n\n"
            "硬性规则：如果用户询问这个项目/仓库做什么、技术栈、架构、启动方式、依赖或源码行为，"
            "回答前必须检查工作区文件。优先读取 README、package 或 manifest 文件；如果工作区文件不可用，要明确说明。\n\n"
            "涉及源码的问题，优先使用 workspace.search_code、workspace.read_file 或 workspace.list_files。"
            "涉及已索引的仓库历史和上传知识，使用 rag.search_similar_documents 或 devflow_search_evidence。\n\n"
            "混合 RAG 约定：服务端上下文中的 [Related Evidence] 是根据原始问题完成的调用前检索。"
            "如果这些证据不足、互相冲突，或问题包含多个子问题，请改写为更具体的 query 调用 "
            "rag.search_similar_documents；可以用不同 query 连续补查，并在最终回答前比较多次 tool_result。\n\n"
            "工作流硬性规则：如果用户请求多 Agent 审查、合并就绪度、当前阻塞、工程状态、跨 Agent 综合，"
            "或依赖 Issue、PR、CI 与仓库证据共同判断的决策，必须先调用 workflow.run_engineering_review，"
            "再给出最终建议。\n\n"
            "分配硬性规则：如果用户询问谁应该负责、处理、接手或被分配某个 Issue/PR/任务，"
            "可选人员只能来自下方列出的 DevFlow team_members，以及工具明确返回的现有 GitHub assignees。"
            "不要把仓库内容、项目角色设定、模型名、示例、作者或虚构标签当成团队成员。"
            "如果 team_members 为空且没有现有 assignee，请回答当前还没有添加项目成员，"
            "并要求用户在团队面板中添加开发者。可以说明可能需要的能力方向，但不要点名具体人员。\n\n"
            "循环约定：每次工具调用都会先记录为 tool_use，再记录为 tool_result。决定下一步前必须阅读之前的 tool_result 内容。"
            "如果目录列表里出现值得继续查看的 README、manifest、代码文件、Issue、PR 或 CI 记录，不要停在目录列表。"
            "除非新证据使重复调用有必要，不要用相同参数重复调用同一个工具。\n\n"
            f"当前 Agent 配置的运行模型：{settings.llm_model}\n\n"
            f"可用原生工具：\n{self._tool_help(state)}\n\n"
            f"已注册的 DevFlow 技能：\n{self._registered_skills_context(state.get('context') or {})}\n\n"
            f"本轮按需加载的 Skill 完整指令：\n{self._active_skill_prompt(state.get('context') or {})}\n\n"
            f"DevFlow 团队上下文：\n{self._team_assignment_context(state.get('context') or {})}\n\n"
            f"服务端上下文：\n{server_context}\n\n"
            f"上下文覆盖度：\n{json.dumps(self._history_coverage(context), ensure_ascii=False)}\n\n"
            f"最近消息：\n{json.dumps(history, ensure_ascii=False)}\n\n"
            f"已有观察结果（已按总预算投影，完整结果保存在会话转录）：\n{json.dumps(prompt_observations, ensure_ascii=False)}\n\n"
            f"工具调用/结果索引：\n{json.dumps(prompt_events, ensure_ascii=False)}\n\n"
            f"用户消息：{state.get('message') or ''}"
        )

    # 补齐服务端掌握的上下文并规范化参数；让不同工具收到稳定的输入形状。
    def _normalize_action_input(self, action: str, action_input: dict[str, Any], state: ConversationState) -> dict[str, Any]:
        normalized = dict(action_input)
        message = str(state.get("message") or "")
        context = state.get("context") or {}

        if action in {
            "repo_health.generate_summary",
            "rag.search_similar_documents",
            "issue_agent.analyze_issue",
            "pr_review_agent.analyze_pr",
            "ci_debug_agent.analyze_workflow_run",
            "report_agent.generate_weekly_report",
            "safety_agent.classify_action_risk",
            "workflow.run_engineering_review",
            "workspace.search_code",
        }:
            normalized.setdefault("message", message)
        if action == "workspace.search_code":
            normalized.setdefault("query", message)
        if action == "rag.search_similar_documents":
            normalized.setdefault("query", str(normalized.get("message") or message))
        if action == "workspace.read_file" and not normalized.get("path") and self._asks_project_overview(message):
            normalized["path"] = "README.md"
        if action == "devflow_search_evidence":
            normalized.setdefault("query", message)
        if action == "devflow_get_thread_context":
            normalized.setdefault("limit", 20)
        if action == "devflow_read_session_events":
            normalized.setdefault("limit", 5)
        if action.startswith("devflow_") and context.get("repo_id"):
            normalized["repo_id"] = context["repo_id"]
            normalized.pop("repo_full_name", None)
        if action.startswith("devflow_") and context.get("conversation_id"):
            normalized["conversation_id"] = context["conversation_id"]
        return normalized

    def _asks_project_overview(self, message: str) -> bool:
        lowered = message.lower()
        if any(token in lowered for token in ["issue", "pull request", "pr #", "ci", "未关闭", "待处理"]):
            return False
        return any(
            token in lowered
            for token in [
                "what is this project",
                "what does this project",
                "project about",
                "repository about",
                "tech stack",
                "architecture",
                "setup",
                "项目介绍",
                "项目概览",
                "这个项目",
                "该项目",
                "这个仓库",
                "该仓库",
                "\u505a\u4ec0\u4e48",
                "\u67b6\u6784",
                "\u6280\u672f\u6808",
            ]
        )

    def _required_memory_tool(self, message: str, context: dict[str, Any] | None = None) -> str | None:
        policy = self._memory_context_policy(message, context or {})
        return str((policy or {}).get("tool") or "") or None

    # 先判断问题需要的是“刚才的内容”“更早原文”还是其他任务，再选合适的记忆路径。
    # 最近历史足够且精确时可直接回答，减少不必要的历史工具调用。
    # 需要被省略/封存的原文时，选择线程上下文或会话事件工具，回到持久 transcript 查证。
    # 判断会结合 history_coverage：仅有一段看似相关历史，不代表它覆盖用户要找的完整范围。
    # 返回的是策略元数据；run_stream 负责据此调用工具并决定是否提前结束本轮。
    def _memory_context_policy(self, message: str, context: dict[str, Any]) -> dict[str, Any] | None:
        lowered = message.lower()
        for tool_name in [
            "devflow_search_evidence",
            "devflow_get_thread_context",
            "devflow_read_session_events",
        ]:
            if tool_name in lowered:
                return {"mode": "required_tool", "tool": tool_name, "reason": "explicit_tool_request"}
        if any(token in lowered for token in ["历史证据", "历史决策", "项目记忆", "historical evidence", "past decision"]):
            return {"mode": "required_tool", "tool": "devflow_search_evidence", "reason": "project_history_request"}
        if any(
            token in lowered
            for token in [
                "封存会话",
                "会话摘要",
                "上次会话",
                "上个会话",
                "跨会话",
                "session digest",
                "session events",
                "previous session",
                "previous conversation",
            ]
        ):
            return {"mode": "required_tool", "tool": "devflow_read_session_events", "reason": "cross_session_request"}

        immediate_recall = any(
            token in lowered
            for token in [
                "刚刚说",
                "刚才说",
                "上一条消息",
                "上一句话",
                "上一句",
                "what did i just say",
                "my last message",
            ]
        )
        broad_recall = immediate_recall or any(
            token in lowered
            for token in [
                "之前说过",
                "上次讨论",
                "聊天历史",
                "会话历史",
                "你记得",
                "前面说过",
                "what did we say",
                "earlier in this chat",
            ]
        )
        if not broad_recall:
            return None

        requires_exact_verification = any(
            token in lowered
            for token in ["原话", "逐字", "准确核实", "精确核实", "审计", "verbatim", "exact quote", "verify the record"]
        )
        if requires_exact_verification:
            return {"mode": "required_tool", "tool": "devflow_get_thread_context", "reason": "exact_verification_requested"}

        if self._recent_history_can_answer(message, context, immediate=immediate_recall):
            return {
                "mode": "direct_recent_history",
                "tool": None,
                "reason": "exact_recent_messages_available",
            }
        return {"mode": "required_tool", "tool": "devflow_get_thread_context", "reason": "recent_history_incomplete"}

    def _recent_history_can_answer(self, message: str, context: dict[str, Any], *, immediate: bool) -> bool:
        history = self._history_messages(context)
        prior_user_index = self._last_prior_user_index(history, message)
        if prior_user_index is None:
            return False
        coverage = self._history_coverage(context)
        if not coverage:
            return True
        if immediate:
            exact_suffix = int(coverage.get("exact_suffix_messages") or 0)
            distance_from_end = len(history) - prior_user_index
            return exact_suffix >= distance_from_end
        return bool(coverage.get("complete_and_exact"))

    def _last_prior_user_index(self, history: list[dict[str, str]], message: str) -> int | None:
        normalized_current = self._normalize_history_text(message)
        indexes = [index for index, item in enumerate(history) if item.get("role") == "user"]
        if indexes and self._normalize_history_text(history[indexes[-1]].get("content") or "") == normalized_current:
            indexes.pop()
        return indexes[-1] if indexes else None

    def _normalize_history_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    async def _direct_answer_from_recent_history(self, state: ConversationState) -> str:
        if self.model is None:
            return self._local_recent_history_answer(state)
        history = self._history_messages(state.get("context") or {})
        try:
            system_prompt = (
                "请直接根据提供的近期原始消息回答，不要声称调用了任何工具。"
                "最后一条与当前问题相同的 user 消息是本轮问题，不要把它当成用户要回忆的上一条内容。"
                "只回答上下文能证明的内容；如果用户问上一条消息，优先简洁复述上一条 user 消息。"
            )
            user_prompt = (
                f"近期原始消息：{json.dumps(history, ensure_ascii=False)}\n"
                f"当前用户问题：{state.get('message') or ''}"
            )
            estimated = self._record_prompt_diagnostics(
                state,
                phase="direct_recent_history",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            response = await self.model.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            self._record_model_usage(
                state,
                response,
                phase="direct_recent_history",
                estimated_prompt_tokens=estimated,
            )
            return self._message_content(response).strip() or self._local_recent_history_answer(state)
        except Exception:
            return self._local_recent_history_answer(state)

    def _local_recent_history_answer(self, state: ConversationState) -> str:
        history = self._history_messages(state.get("context") or {})
        prior_user_index = self._last_prior_user_index(history, str(state.get("message") or ""))
        if prior_user_index is None:
            return "当前上下文里没有可确认的上一条用户消息。"
        content = history[prior_user_index].get("content") or ""
        return f"你刚刚说的是：{content}"

    def _history_coverage(self, context: dict[str, Any]) -> dict[str, Any]:
        compression_stats = context.get("compression_stats") or {}
        if not isinstance(compression_stats, dict):
            return {}
        coverage = compression_stats.get("history_coverage") or {}
        return coverage if isinstance(coverage, dict) else {}

    async def _prefetch_memory_context(self, state: ConversationState, public_name: str) -> AsyncIterator[dict[str, Any]]:
        tools = state.get("tools") or {}
        tool_names = state.get("tool_name_map") or {}
        safe_name = next((safe for safe, public in tool_names.items() if public == public_name), None)
        if not safe_name or safe_name not in tools:
            return
        message = str(state.get("message") or "")
        args: dict[str, Any]
        if public_name == "devflow_search_evidence":
            args = {"query": message, "limit": 5}
        elif public_name == "devflow_get_thread_context":
            args = {"keyword": "", "limit": 20}
        else:
            args = {"limit": 5}
        async for event in self._execute_tool_call_events(
            {"name": safe_name, "args": args, "id": f"required_{public_name}_{len(state.get('agent_events') or [])}"},
            state,
        ):
            yield event

    async def _prefetch_project_context(self, state: ConversationState) -> AsyncIterator[dict[str, Any]]:
        tools = state.get("tools") or {}
        tool_names = state.get("tool_name_map") or {}
        safe_by_public = {public: safe for safe, public in tool_names.items()}

        async def call(public_name: str, args: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
            safe_name = safe_by_public.get(public_name)
            if not safe_name or safe_name not in tools:
                return
            async for event in self._execute_tool_call_events(
                {"name": safe_name, "args": args, "id": f"prefetch_{public_name}_{len(state.get('agent_events') or [])}"},
                state,
            ):
                yield event

        async for event in call("workspace.list_files", {"path": ".", "limit": 120}):
            yield event
        observed_paths: set[str] = set()
        for item in state.get("observations") or []:
            if not isinstance(item, dict) or not isinstance(item.get("observation"), dict):
                continue
            preview = str(item["observation"].get("answer_preview") or "")
            observed_paths.update(token.strip("'\"`[](),").lower() for token in preview.split())
        for path in ["README.md", "README.zh-CN.md", "package.json", "SETUP.md"]:
            if path.lower() in observed_paths:
                async for event in call("workspace.read_file", {"path": path, "line_count": 220}):
                    yield event

    # 把路由注入的异步业务函数逐一包装成 LangChain StructuredTool。
    # 每个包装都绑定对应的原函数和当前服务端 context，模型只提交 action_input。
    # 闭包中的默认参数用于固定本轮循环对应的工具，避免多个包装最后都指向同一个循环变量。
    # 统一 GenericToolArgs 提供可见 schema，工具说明再补入 Skill 入口等元数据。
    # 返回字典以兼容工具名为键，便于模型响应的名称回查到真正的执行对象。
    def _build_langchain_tools(self, raw_tools: dict[str, ConversationTool], context: dict[str, Any]) -> dict[str, StructuredTool]:
        wrapped: dict[str, StructuredTool] = {}
        # 每个服务端业务函数包装成一个可声明给模型的工具，循环内要固定当前函数，避免闭包最终全指向最后一个。
        for name, func in raw_tools.items():
            async def call_tool(_func: ConversationTool = func, **kwargs: Any) -> dict[str, Any]:
                input_data = {key: value for key, value in kwargs.items() if value is not None}
                return await _func(input_data, context)

            safe_name = self._safe_tool_name(name)
            call_tool.__name__ = safe_name
            wrapped[safe_name] = StructuredTool.from_function(
                coroutine=call_tool,
                name=safe_name,
                description=self._tool_description(name, context),
                args_schema=GenericToolArgs,
                infer_schema=False,
            )
        return wrapped

    def _tool_name_map(self, raw_tools: dict[str, ConversationTool]) -> dict[str, str]:
        return {self._safe_tool_name(name): name for name in raw_tools}

    def _safe_tool_name(self, name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]", "__", name)

    def _public_tool_name(self, safe_name: str, state: ConversationState) -> str:
        return (state.get("tool_name_map") or {}).get(safe_name, safe_name)

    def _tool_description(self, name: str, context: dict[str, Any] | None = None) -> str:
        descriptions = {
            "devflow_search_evidence": "MCP 记忆工具。搜索持久记忆、会话摘要、仓库证据、Issue、PR、CI 和 Agent 运行记录。",
            "devflow_get_thread_context": "MCP 记忆工具。跨压缩边界读取当前仓库会话的完整原始聊天转录，支持消息 UUID 和字符偏移分页。",
            "devflow_read_session_events": "MCP 记忆工具。读取已封存的会话摘要、原始消息和工具事件。",
            "workspace.list_files": "列出所选仓库本地 checkout 下的文件。只读，用于探索源码树。",
            "workspace.read_file": "从所选仓库 checkout 中读取文本文件，可指定 start_line 和 line_count。只读。",
            "workspace.search_code": "在所选仓库 checkout 中搜索代码或文档文本。只读。",
            "repo_health.generate_summary": "总结当前仓库健康状态。",
            "rag.search_similar_documents": "搜索项目历史与非结构化知识，包括 Issue 描述、PR 说明与 Review 讨论、失败 CI 日志、项目文档、上传知识和已批准的长期记忆。可传入改写后的 query，并用不同 query 连续调用以补查和交叉验证。源码优先使用 workspace 工具，团队成员和实时状态使用对应结构化工具。",
            "issue_agent.analyze_issue": "分析最新同步的 Issue。",
            "pr_review_agent.analyze_pr": "分析最新同步的 PR。",
            "ci_debug_agent.analyze_workflow_run": "分析最新失败的 CI 运行。",
            "report_agent.generate_weekly_report": "生成仓库报告。",
            "safety_agent.classify_action_risk": "安全起草有风险的写操作或外部发送动作。",
            "workflow.run_engineering_review": "运行 Planner -> 专用 Agent -> Observer -> Synthesis 工作流，用于 PR 就绪度、阻塞项、优先级和多 Agent 仓库审查等跨领域工程决策。",
        }
        suffixes = {}
        if context:
            raw_suffixes = context.get("skill_description_by_tool") or {}
            if isinstance(raw_suffixes, dict):
                suffixes = raw_suffixes
        return f"公开工具：{name}。{descriptions.get(name, 'DevFlow 工具。')}{suffixes.get(name, '')}"

    def _tool_help(self, state: ConversationState) -> str:
        tools = state.get("tools") or {}
        return "\n".join(f"- {name}: {tool.description}" for name, tool in tools.items())

    # 收尾时依据已经执行的工具结果生成答案；模型不可用或无有效输出时走本地观察结果兜底。
    async def _final_from_observations(self, state: ConversationState) -> str:
        if self.model is None:
            return self._local_observation_answer(state)
        try:
            system_prompt = (
                "请使用用户的语言回答，并且只能基于观察结果、tool_result 记录和上下文。"
                "关键判断要说明 WHY；把证据、推断、不确定项和下一步区分开。"
                "如果是审查或合并建议，不能用 looks good、没什么问题一笔带过，必须保留风险和阻塞项。"
                "对于负责人、已分配人员、责任人或任务分配问题，只能从明确的 DevFlow team_members "
                "或工具返回的现有 GitHub assignees 中推荐人员。仓库文档、Issue 文本、代码、示例、"
                "角色设定、模型名和团队比喻都不能当成人。如果没有配置团队成员，也没有观察到现有 assignee，"
                "请说明当前还没有添加项目成员，并要求用户在团队面板中添加开发者。"
            )
            prompt_observations, prompt_events, runtime_stats = self._runtime_tool_context(state)
            state["runtime_context_stats"] = {**(state.get("runtime_context_stats") or {}), **runtime_stats}
            user_prompt = (
                f"用户消息：{state.get('message') or ''}\n"
                f"当前 Agent 配置的运行模型：{settings.llm_model}\n"
                f"DevFlow 团队上下文：{self._team_assignment_context(state.get('context') or {})}\n"
                f"上下文：{self._server_context(state.get('context') or {})}\n"
                f"观察结果：{json.dumps(prompt_observations, ensure_ascii=False)}\n"
                f"工具调用/结果索引：{json.dumps(prompt_events, ensure_ascii=False)}"
            )
            estimated = self._record_prompt_diagnostics(
                state,
                phase="final_from_observations",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            response = await self.model.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            self._record_model_usage(
                state,
                response,
                phase="final_from_observations",
                estimated_prompt_tokens=estimated,
            )
            return self._message_content(response)
        except Exception:
            return self._local_observation_answer(state)

    def _local_observation_answer(self, state: ConversationState) -> str:
        observations = state.get("observations") or []
        excerpts: list[str] = []
        for item in observations:
            if not isinstance(item, dict):
                continue
            value = item.get("answer") or item.get("content") or item.get("result") or item.get("message")
            if not value and isinstance(item.get("observation"), dict):
                nested = item["observation"]
                if nested.get("route") == "tool_error":
                    continue
                value = nested.get("answer_preview") or nested.get("answer") or nested.get("content")
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            text = " ".join(str(value or "").split())
            if text:
                excerpts.append(text[:600])
        if not excerpts:
            evidence_answer = self._local_prefetched_evidence_answer(state)
            if evidence_answer:
                return evidence_answer
            context_text = " ".join(self._server_context(state.get("context") or {}).split())
            if context_text:
                return "已读取当前仓库上下文。\n\n" + context_text[:1200]
            return "当前没有取得足够证据。请先确认仓库已连接并完成同步；开放式对话还需要配置 LLM。"
        return "已读取当前仓库证据。\n\n" + "\n\n".join(
            f"{index + 1}. {excerpt}" for index, excerpt in enumerate(excerpts[:4])
        )

    def _local_prefetched_evidence_answer(self, state: ConversationState) -> str:
        context = state.get("context") or {}
        raw_evidence = context.get("evidence")
        if not isinstance(raw_evidence, list):
            return ""
        evidence = [item for item in raw_evidence if isinstance(item, dict)]
        if not evidence:
            return ""

        message = str(state.get("message") or "").lower()
        asks_open_issues = "issue" in message and any(
            token in message for token in ["未关闭", "待处理", "open", "开放"]
        )
        if asks_open_issues:
            open_issues = [
                item
                for item in evidence
                if str(item.get("source_type") or "") == "issue"
                and str((item.get("metadata") or {}).get("state") or "").lower() == "open"
            ]
            if open_issues:
                lines = ["根据本轮预检索证据，当前仓库有这些未关闭的 Issue："]
                for item in open_issues[:8]:
                    title = str(item.get("title") or "未命名 Issue").strip()
                    snippet = " ".join(str(item.get("snippet") or "").split())
                    lines.append(f"- {title}" + (f"：{snippet[:240]}" if snippet else ""))
                lines.append("\n数据来自当前仓库已同步的 Issue 证据；本轮没有调用工作区源码工具。")
                return "\n".join(lines)

        rendered = [self._render_evaluation_context(item) for item in evidence[:8]]
        rendered = [item for item in rendered if item]
        if not rendered:
            return ""
        return "根据当前仓库已同步的证据：\n\n" + "\n\n".join(
            f"{index + 1}. {item}" for index, item in enumerate(rendered)
        )

    def _prefetched_evidence_citations(self, state: ConversationState) -> list[dict[str, str]]:
        context = state.get("context") or {}
        evidence = context.get("evidence")
        if not isinstance(evidence, list):
            return []
        return [
            {
                "type": str(item.get("source_type") or "evidence"),
                "id": str(item.get("source_id") or item.get("id") or ""),
                "title": str(item.get("title") or "未命名证据"),
            }
            for item in evidence[:8]
            if isinstance(item, dict)
        ]

    def _append_agent_event(self, state: ConversationState, event_type: str, data: dict[str, Any]) -> dict[str, Any]:
        event = {
            "type": event_type,
            "step": state.get("step", 0),
            **data,
        }
        state["agent_events"] = [*(state.get("agent_events") or []), event]
        return event

    def _tool_event_transcript(self, state: ConversationState) -> list[dict[str, Any]]:
        return [
            event
            for event in state.get("agent_events") or []
            if event.get("type") in {"tool_use", "tool_result", "tool_retry", "agent_step", "agent_loop_stop"}
        ]

    # 从完整运行状态派生“下一轮模型可见的工具投影”，不直接修改已保存的轨迹。
    # 先保留最近且有用的观察、工具名、状态和关键字段，再按 token 预算裁剪过长文本。
    # 多次搜索会留下不同查询与观察；投影需要保留关联，避免把 A 查询的结果当成 B 查询的证据。
    # aggressive 路径缩得更紧，用于供应商报告超窗后的单次恢复。
    # 界面/封存记录与提示投影是不同用途：提示短一点不应导致审计现场也只剩缩略文本。
    def _runtime_tool_context(
        self,
        state: ConversationState,
        *,
        aggressive: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """Project complete tool state into a bounded prompt-only view.

        The full observations and events remain in ``state`` and are sealed in
        ChatSession.  Only this derived view is shortened for the next model
        call, matching Claude Code's separation between durable transcript and
        active context.
        """
        raw_observations = [item for item in state.get("observations") or [] if isinstance(item, dict)]
        budget = default_context_budget().tool_observation_tokens
        if aggressive:
            budget = max(512, budget // 4)
        per_item_budget = max(160, min(1400, budget // max(len(raw_observations), 1)))
        remaining = budget  # 本轮工具观察区还有多少额度，开始未装入观察，剩余量等于全部预算
        kept_reversed: list[dict[str, Any]] = []  # 从最新观察开始倒序收集，最后还原原来的执行顺序
        omitted = 0  # 省略观察的诊断计数起点；尚未筛选任何观察时为 0
        # 优先给最新工具观察分配额度，模型下一步通常更依赖刚取得的信息；旧完整结果仍留在 state。
        for item in reversed(raw_observations):
            # 建立投影副本再裁剪，不能直接缩短 state 里的完整观察，否则封存时也只剩摘要。
            next_item = dict(item)
            observation = next_item.get("observation")
            if isinstance(observation, dict):
                next_observation = dict(observation)
                preview = str(
                    next_observation.get("answer_preview")
                    or next_observation.get("answer")
                    or next_observation.get("content")
                    or ""
                )
                if preview:
                    next_observation["answer_preview"] = clip_to_token_budget(preview, per_item_budget)
                    # answer 已裁成 answer_preview 后移除原大字段，避免同一个结果既以缩略版又以原文重复进入提示。
                    next_observation.pop("answer", None)
                    next_observation.pop("content", None)
                next_item["observation"] = next_observation
            else:
                next_item["observation"] = clip_to_token_budget(str(observation or ""), per_item_budget)

            # 估算完整投影对象的成本，包括字段名和参数，不只估算其中 answer_preview 的正文。
            serialized = json.dumps(next_item, ensure_ascii=False, default=str)
            cost = estimate_tokens(serialized)
            # 已经有较新观察且本条装不下时跳过，继续看更早但可能更短的观察，所以用 continue 而不是立即结束。
            if cost > remaining and kept_reversed:
                omitted += 1
                continue
            # 首条也过大时再压成最小可读形状，至少保住最近工具身份和部分结果，不让整个观察区直接为空。
            if cost > remaining:
                next_item = {
                    "tool_name": str(item.get("tool_name") or "tool"),
                    "observation": clip_to_token_budget(serialized, max(120, remaining)),
                    "compacted": True,
                }
                cost = estimate_tokens(json.dumps(next_item, ensure_ascii=False))
            kept_reversed.append(next_item)
            # 每接受一条就扣实际序列化估算成本，并将剩余额度限制为非负。
            remaining = max(0, remaining - cost)
            if remaining <= 0:
                omitted += max(0, len(raw_observations) - len(kept_reversed))
                break

        prompt_observations = list(reversed(kept_reversed))
        if omitted:
            prompt_observations.insert(
                0,
                {
                    "compacted": True,
                    "omitted_old_observations": omitted,
                    "recovery": "完整工具结果保存在本轮 ChatSession transcript，可按 session 读取。",
                },
            )

        prompt_events: list[dict[str, Any]] = []
        # 除了正文观察，还生成较小的事件索引，保留工具名、参数、状态等定位线索，避免压缩后不知道做过什么。
        for event in self._tool_event_transcript(state):
            summary = {
                key: value
                for key, value in event.items()
                if key
                in {
                    "type",
                    "step",
                    "id",
                    "tool_use_id",
                    "tool_name",
                    "arguments",
                    "status",
                    "route",
                    "error_kind",
                    "retryable",
                    "reason",
                }
            }
            if "arguments" in summary:
                args_text = json.dumps(summary["arguments"], ensure_ascii=False, default=str)
                if estimate_tokens(args_text) > 240:
                    summary["arguments"] = clip_to_token_budget(args_text, 240)
            prompt_events.append(summary)

        original_tokens = estimate_tokens(json.dumps(raw_observations, ensure_ascii=False, default=str))
        prompt_tokens = estimate_tokens(json.dumps(prompt_observations, ensure_ascii=False, default=str))
        return prompt_observations, prompt_events, {
            "mode": "runtime_tool_projection",
            "aggressive": aggressive,
            "observation_budget_tokens": budget,
            "original_observation_tokens": original_tokens,
            "prompt_observation_tokens": prompt_tokens,
            "estimated_tokens_saved": max(0, original_tokens - prompt_tokens),
            "source_observations": len(raw_observations),
            "prompt_observations": len(prompt_observations),
            "omitted_observations": omitted,
            "tool_event_index_entries": len(prompt_events),
        }

    def _compress_history_for_retry(
        self,
        history: list[dict[str, str]],
        max_tokens: int,
    ) -> list[dict[str, str]]:
        kept: list[dict[str, str]] = []  # 仅本次重试要发送的历史副本，持久消息列表不在这里删除
        used = 0  # 超窗重试的历史投影已用额度；这一轮重新选择历史，因此从 0 计数
        # 重试时从最新历史往回装，给旧消息更少机会，优先保住当前问题附近的连续语境。
        for item in reversed(history):
            content = clip_to_token_budget(item.get("content") or "", max(120, min(700, max_tokens // 3)))
            cost = estimate_tokens(item.get("role")) + estimate_tokens(content)
            # 已有内容后再加入当前条会超限就停止，单条先裁剪，总量再校验，二者共同控制输入。
            if kept and used + cost > max_tokens:
                break
            kept.append({"role": item.get("role") or "user", "content": content})
            used += cost
        # 选择阶段按新到旧，但模型输入仍需按旧到新的正常对话顺序组织。
        return list(reversed(kept))

    # 记录估算的输入预算用于排查上下文压力；估算 token 与供应商返回的真实用量不是同一指标。
    def _record_prompt_diagnostics(
        self,
        state: ConversationState,
        *,
        phase: str,
        system_prompt: str,
        user_prompt: str,
    ) -> int:
        system_tokens = estimate_tokens(system_prompt)
        user_tokens = estimate_tokens(user_prompt)
        tool_schema_tokens = self._tool_schema_estimated_tokens(state)
        total = system_tokens + user_tokens + tool_schema_tokens
        pressure = calculate_context_pressure(total)
        diagnostic = {
            "phase": phase,
            "estimated_prompt_tokens": total,
            "system_tokens": system_tokens,
            "user_context_tokens": user_tokens,
            "tool_schema_tokens": tool_schema_tokens,
            "pressure": pressure.as_dict(),
        }
        state["prompt_diagnostics"] = [*(state.get("prompt_diagnostics") or []), diagnostic]
        return total

    def _tool_schema_estimated_tokens(self, state: ConversationState) -> int:
        total = 0
        for tool in (state.get("tools") or {}).values():
            total += estimate_tokens(str(getattr(tool, "name", "")))
            total += estimate_tokens(str(getattr(tool, "description", "")))
            args_schema = getattr(tool, "args_schema", None)
            try:
                schema = args_schema.model_json_schema() if args_schema else {}
            except (AttributeError, TypeError, ValueError):
                schema = {}
            total += estimate_tokens(json.dumps(schema, ensure_ascii=False, default=str))
        return total

    def _record_model_usage(
        self,
        state: ConversationState,
        response: Any,
        *,
        phase: str,
        estimated_prompt_tokens: int,
    ) -> None:
        usage = self._normalize_model_usage(response)
        usage.update(
            {
                "phase": phase,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "source": "provider" if usage.get("input_tokens") is not None else "estimate",
            }
        )
        if usage.get("input_tokens") is None:
            usage["input_tokens"] = estimated_prompt_tokens
        if usage.get("total_tokens") is None:
            usage["total_tokens"] = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
        state["model_usage"] = [*(state.get("model_usage") or []), usage]

    def _normalize_model_usage(self, response: Any) -> dict[str, Any]:
        raw = getattr(response, "usage_metadata", None)
        if raw is not None and not isinstance(raw, dict) and hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        raw = raw if isinstance(raw, dict) else {}
        response_metadata = getattr(response, "response_metadata", None)
        response_metadata = response_metadata if isinstance(response_metadata, dict) else {}
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
        token_usage = token_usage if isinstance(token_usage, dict) else {}
        input_details = raw.get("input_token_details") or {}
        input_details = input_details if isinstance(input_details, dict) else {}
        return {
            "input_tokens": raw.get("input_tokens", token_usage.get("prompt_tokens")),
            "output_tokens": raw.get("output_tokens", token_usage.get("completion_tokens")),
            "total_tokens": raw.get("total_tokens", token_usage.get("total_tokens")),
            "cache_read_input_tokens": input_details.get("cache_read", token_usage.get("cache_read_input_tokens", 0)),
            "cache_creation_input_tokens": input_details.get("cache_creation", token_usage.get("cache_creation_input_tokens", 0)),
            "model": response_metadata.get("model_name") or settings.llm_model,
        }

    def _runtime_metadata(self, state: ConversationState) -> dict[str, Any]:
        return {
            "model_usage": state.get("model_usage") or [],
            "context_diagnostics": {
                "prompt_calls": state.get("prompt_diagnostics") or [],
                "runtime_tool_context": state.get("runtime_context_stats") or {},
                "reactive_overflow_retry": bool(state.get("context_overflow_recovery_attempted")),
            },
        }

    async def _fallback_answer(self, state: ConversationState) -> str:
        return await self._final_from_observations(state)

    def _history_messages(self, context: dict[str, Any]) -> list[dict[str, str]]:
        raw_messages = context.get("server_messages") or context.get("messages") or []
        if not isinstance(raw_messages, list):
            return []
        messages: list[dict[str, str]] = []
        for item in raw_messages:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content})
        return messages

    def _server_context(self, context: dict[str, Any]) -> str:
        value = context.get("server_context")
        return str(value) if value else ""

    def _registered_skills_context(self, context: dict[str, Any]) -> str:
        value = context.get("registered_skills")
        return str(value) if value else "当前没有注册 DevFlow 技能。"

    # 从注册中心选择本次任务真正激活的 Skill；目录摘要与完整执行指令按需分层加载。
    def _activate_skills(self, message: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        registry = context.get("_skill_registry")
        if not isinstance(registry, SkillRegistry):
            return []
        requested_name = context.get("requested_skill") or context.get("skill_name")
        activations = registry.activate(
            message,
            requested_name=str(requested_name).strip() if requested_name else None,
        )
        return [activation.model_dump(mode="json") for activation in activations]

    def _active_skills(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        skills = context.get("active_skills") or []
        return [item for item in skills if isinstance(item, dict)] if isinstance(skills, list) else []

    # 只把本轮激活的 Skill 正文与受限资源组织成执行提示，而非把整个 Skill 目录塞给模型。
    # 资源带路径与内容，便于模型区分“执行指令”与“被引用的辅助材料”。
    # 先由注册中心验证资源边界再加载，工具执行阶段仍需检查显式 Skill 合同。
    # 因此完整指令加载、实际工具约束、运行后验证是相互补充的三个步骤。
    def _active_skill_prompt(self, context: dict[str, Any]) -> str:
        skills = self._active_skills(context)
        if not skills:
            return "本轮没有匹配到需要加载完整正文的 Skill；仅保留轻量目录摘要。"
        blocks = []
        for skill in skills:
            resources = skill.get("resources") if isinstance(skill.get("resources"), list) else []
            resource_sections = []
            for resource in resources:
                if not isinstance(resource, dict) or not str(resource.get("content") or "").strip():
                    continue
                resource_sections.append(f"### 资源：{resource.get('path')}\n{resource.get('content')}")
            resource_text = "\n\n按需读取的资源：\n" + "\n\n".join(resource_sections) if resource_sections else ""
            blocks.append(
                f"## {skill.get('skill_title')} ({skill.get('skill_name')} v{skill.get('skill_version')})\n"
                f"激活方式：{skill.get('activation_mode')}；原因：{skill.get('activation_reason')}\n"
                f"允许工具：{', '.join(str(item) for item in skill.get('tools') or []) or '无'}\n"
                f"输出契约：{skill.get('output_contract')}；安全级别：{skill.get('safety_level')}\n\n"
                f"{str(skill.get('instructions') or '').strip()}"
                f"{resource_text}"
            )
        return clip_to_token_budget("\n\n".join(blocks), 6000, suffix="\n[Skill 指令因上下文预算被截断]")

    # 用户显式选择 Skill 时检查允许工具，防止执行偏离该 Skill 约定的操作。
    # 这是一层运行约束，不替代 API 层的权限检查或外部操作的人工确认。
    def _explicit_skill_tool_violation(self, tool_name: str, state: ConversationState) -> str | None:
        context = state.get("context") or {}
        explicit = [
            skill
            for skill in self._active_skills(context)
            if skill.get("activation_mode") == "explicit"
        ]
        if not explicit:
            return None
        allowed = {
            str(tool)
            for skill in explicit
            for tool in skill.get("tools") or []
        }
        if tool_name in allowed:
            return None
        skill_names = "、".join(str(skill.get("skill_name")) for skill in explicit)
        return f"工具 {tool_name} 不在显式 Skill（{skill_names}）声明的允许工具列表中，已拒绝执行。"

    # 运行结束后用实际 tool_calls 检查本次激活 Skill 的执行情形。
    # 将激活信息、执行工具和输出合同相关检查写入事件，供前端/评测核对。
    # “Skill 被选中”只说明指令已进入上下文；本记录帮助判断是否真的按入口和约束执行。
    # 它记录验证结果，不会倒放或撤销已经执行的工具，因此前置约束仍不可省略。
    def _record_skill_validation(self, state: ConversationState) -> None:
        context = state.get("context") or {}
        active = self._active_skills(context)
        if not active:
            return
        observed_tools = [
            str(event.get("tool_name") or "")
            for event in state.get("agent_events") or []
            if isinstance(event, dict) and event.get("type") == "tool_use" and event.get("tool_name")
        ]
        successful_tools = [
            str(event.get("tool_name") or "")
            for event in state.get("agent_events") or []
            if isinstance(event, dict)
            and event.get("type") == "tool_result"
            and event.get("status") == "success"
            and event.get("tool_name")
        ]
        explicit_allowed = {
            str(tool)
            for skill in active
            if skill.get("activation_mode") == "explicit"
            for tool in skill.get("tools") or []
        }
        explicit_violations = [tool for tool in observed_tools if explicit_allowed and tool not in explicit_allowed]
        # 对每个激活 Skill 分别核对入口与真实成功调用；某个 Skill 成功不代表其他激活 Skill 也完成了。
        for skill in active:
            allowed = {str(tool) for tool in skill.get("tools") or []}
            used = [tool for tool in observed_tools if tool in allowed]
            entrypoint_called = str(skill.get("entrypoint") or "") in used
            successful_used = [tool for tool in successful_tools if tool in allowed]
            entrypoint_succeeded = str(skill.get("entrypoint") or "") in successful_used
            delegated_tasks = [
                event
                for event in state.get("agent_events") or []
                if isinstance(event, dict)
                and event.get("type") == "workflow_task_result"
                and event.get("skill_name") == skill.get("skill_name")
                and event.get("status") == "success"
            ]
            # 先判断显式约束违规，再判断成功；不能因为某个辅助工具成功就掩盖越界行为。
            if explicit_violations and skill.get("activation_mode") == "explicit":
                status = "violated"
            elif entrypoint_succeeded or successful_used or delegated_tasks:
                status = "completed"
            elif used:
                status = "failed"
            else:
                status = "not_executed"
            self._append_agent_event(
                state,
                "skill_validation",
                {
                    "skill_name": skill.get("skill_name"),
                    "skill_title": skill.get("skill_title"),
                    "skill_version": skill.get("skill_version"),
                    "status": status,
                    "instructions_loaded": bool(str(skill.get("instructions") or "").strip()),
                    "instruction_digest": skill.get("instruction_digest"),
                    "declared_steps": skill.get("workflow_steps") or [],
                    "observed_tools": used,
                    "entrypoint_called": entrypoint_called,
                    "entrypoint_succeeded": entrypoint_succeeded,
                    "delegated_task_ids": [str(event.get("task_id") or "") for event in delegated_tasks],
                    "tool_contract_violations": explicit_violations if skill.get("activation_mode") == "explicit" else [],
                    "step_verification": (
                        "entrypoint_completed"
                        if entrypoint_succeeded
                        else "delegated_entrypoint_completed"
                        if delegated_tasks
                        else "supporting_tools_observed"
                        if successful_used
                        else "skill_tool_failed"
                        if used
                        else "no_skill_tool_observed"
                    ),
                },
            )

    def _skill_event_metadata(self, tool_name: str, state: ConversationState) -> dict[str, Any]:
        context = state.get("context") or {}
        active = self._active_skills(context)
        selected = next((skill for skill in active if skill.get("entrypoint") == tool_name), None)
        if selected is None:
            selected = next((skill for skill in active if tool_name in (skill.get("tools") or [])), None)
        if selected is None:
            selected = next((skill for skill in active if skill.get("activation_mode") == "explicit"), None)
        if selected is not None:
            return {
                "skill_name": selected.get("skill_name"),
                "skill_title": selected.get("skill_title"),
                "skill_version": selected.get("skill_version"),
                "skill_steps": selected.get("workflow_steps") or [],
                "skill_activation_mode": selected.get("activation_mode"),
                "skill_activation_reason": selected.get("activation_reason"),
                "skill_instruction_digest": selected.get("instruction_digest"),
            }
        # 激活 Skill 都没能匹配时才查预设映射，元数据用于关联轨迹，不会据此扩大工具权限。
        mapping = context.get("skill_trace_by_tool") or {}
        if not isinstance(mapping, dict):
            return {}
        metadata = mapping.get(tool_name)
        if not isinstance(metadata, dict):
            return {}
        return {
            key: value
            for key, value in metadata.items()
            if key in {"skill_name", "skill_title", "skill_version", "skill_steps"}
        }

    def _team_assignment_context(self, context: dict[str, Any]) -> str:
        members = self._team_members(context)
        if not members:
            return "team_members: []（当前仓库还没有配置 DevFlow 团队成员。）"
        return "team_members: " + json.dumps(members, ensure_ascii=False)

    def _team_members(self, context: dict[str, Any]) -> list[dict[str, str]]:
        raw_members = context.get("team_members") or []
        if not isinstance(raw_members, list):
            return []
        members: list[dict[str, str]] = []
        # 逐条校验并统一成员字段，跳过缺名或非字典项，避免不规则客户端数据直接进入团队提示。
        for member in raw_members:
            if not isinstance(member, dict):
                continue
            name = str(member.get("name") or "").strip()
            if not name:
                continue
            members.append(
                {
                    "name": name,
                    "role": str(member.get("role") or "").strip(),
                    "strengths": str(member.get("strengths") or "").strip(),
                    "techStack": str(member.get("techStack") or member.get("tech_stack") or "").strip(),
                }
            )
        # 限制本轮团队上下文规模，不是删除数据库中的其他成员。
        return members[:20]

    def _summarize_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        answer = str(observation.get("answer") or "")
        budget = default_context_budget()
        limit = budget.tool_observation_tokens if str(observation.get("route") or "").startswith("workspace_") else max(300, budget.tool_observation_tokens // 3)
        compressed = clip_to_token_budget(answer, limit)
        return {
            "route": observation.get("route"),
            "answer_preview": compressed,
            "compression": {
                "mode": "tool_observation_budget",
                "original_estimated_tokens": estimate_tokens(answer),
                "compressed_estimated_tokens": estimate_tokens(compressed),
                "budget_tokens": limit,
            },
            "citations_count": len(observation.get("citations", [])),
        }

    def _initial_evaluation_trace(self, message: str, context: dict[str, Any]) -> dict[str, Any]:
        evidence = context.get("evidence") if isinstance(context.get("evidence"), list) else []
        contexts: list[str] = []
        sources: list[dict[str, Any]] = []
        for item in evidence:
            if not isinstance(item, dict):
                continue
            rendered_context = self._render_evaluation_context(item)
            if rendered_context:
                contexts.append(rendered_context)
            sources.append(
                {
                    "id": str(item.get("id") or ""),
                    "source_id": str(item.get("source_id") or ""),
                    "source_type": str(item.get("source_type") or ""),
                    "title": str(item.get("title") or ""),
                    "score": item.get("score"),
                }
            )
        retrievals = []
        if contexts or sources:
            retrievals.append(
                {
                    "stage": "prefetch",
                    "query": message,
                    "contexts": contexts,
                    "sources": sources,
                }
            )
        return {
            "target": "ChatAgent",
            "retrievals": retrievals,
            "tool_observations": [],
        }

    # 仅在评测运行需要时，从真正工具观察中收集答案可见的证据和调用轨迹。
    # 检索候选保留标题、来源、片段等字段，后续 Judge 应评估模型确实看过的内容。
    # 查询改写和二次检索也应进入同一 trace，不能只用首次预检索代替整个 Agent 的证据集合。
    # 这些记录用于离线评分，不会反过来作为新证据改变当前工具的实际结果。
    def _capture_evaluation_observation(
        self,
        state: ConversationState,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        observation: dict[str, Any],
    ) -> None:
        if not (state.get("context") or {}).get("evaluation_mode"):
            return
        trace = dict(state.get("evaluation_trace") or {})
        tool_observations = list(trace.get("tool_observations") or [])
        tool_observations.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "status": "error"
                if any(
                    call.get("status") == "error"
                    for call in observation.get("tool_calls", [])
                    if isinstance(call, dict)
                )
                else "success",
                "content": str(observation.get("answer") or ""),
            }
        )
        trace["tool_observations"] = tool_observations

        raw_sources = observation.get("retrieval_results")
        sources = [dict(item) for item in raw_sources if isinstance(item, dict)] if isinstance(raw_sources, list) else []
        contexts = [self._render_evaluation_context(item) for item in sources]
        contexts = [item for item in contexts if item]
        if not contexts:
            contexts = [
                str(item).strip()
                for item in observation.get("retrieved_contexts", [])
                if str(item).strip()
            ]
        if tool_name == "rag.search_similar_documents" or contexts or sources:
            retrievals = list(trace.get("retrievals") or [])
            retrievals.append(
                {
                    "stage": "agent_tool",
                    "query": str(arguments.get("query") or arguments.get("message") or ""),
                    "contexts": contexts,
                    "sources": sources,
                }
            )
            trace["retrievals"] = retrievals
        state["evaluation_trace"] = trace

    @staticmethod
    def _render_evaluation_context(item: dict[str, Any]) -> str:
        """Render the same title/source/snippet evidence shape shown to ChatAgent."""

        source_type = str(item.get("source_type") or "").strip()
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or item.get("content") or "").strip()
        heading = " ".join(part for part in ([f"[{source_type}]" if source_type else "", title]) if part)
        if heading and snippet:
            return f"{heading}: {snippet}"
        return heading or snippet

    def _evaluation_result(self, state: ConversationState) -> dict[str, Any]:
        if not (state.get("context") or {}).get("evaluation_mode"):
            return {}
        trace = dict(state.get("evaluation_trace") or {})
        contexts: list[str] = []
        seen: set[str] = set()
        for retrieval in trace.get("retrievals") or []:
            if not isinstance(retrieval, dict):
                continue
            for item in retrieval.get("contexts") or []:
                text = str(item).strip()
                if text and text not in seen:
                    seen.add(text)
                    contexts.append(text)
        trace["retrieved_contexts"] = contexts
        trace["agent_events"] = self._tool_event_transcript(state)
        return {"evaluation_trace": trace}

    def _message_content(self, message: BaseMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
            return "".join(parts)
        return str(content or "")

    # 把已经得到的完整 answer 按固定字符块转成 delta 事件。
    # 注意这不等于供应商逐 token 生成流：这里可能是在模型/工具流程完成后对现有字符串分块。
    # 前端需要最后的 final 事件取得完整答案与持久化标识，不能仅靠 delta 数量判断执行成功。
    async def _stream_answer(self, answer: str) -> AsyncIterator[dict[str, Any]]:
        for part in self._chunk_text(answer):
            yield self._event("delta", {"content": part})

    def _chunk_text(self, text: str, size: int = 36) -> list[str]:
        if not text:
            return []
        return [text[index : index + size] for index in range(0, len(text), size)]

    def _event(self, event: str, data: dict[str, Any]) -> dict[str, Any]:
        return {"event": event, "data": data}


# SSE 一条事件由 event/data 行和空行结尾组成；JSON 保留中文，前端按空行分帧。
def encode_sse(event: dict[str, Any]) -> str:
    return f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"

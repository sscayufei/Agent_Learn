# 对话请求、最终响应、历史与工具轨迹契约；run_id 标识一次执行，message_id 标识持久消息，二者不要混用。
from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    repo_id: UUID  # 当前仓库内部 UUID
    conversation_id: UUID | None = None  # 仓库中的独立会话；空值走兼容默认会话
    message: str  # 原始用户消息，不应被工具查询改写覆盖
    context: dict = Field(default_factory=dict)  # 客户端辅助信息，服务端保留键会被剔除


class ChatPlanExecuteRequest(ChatRequest):
    workflow_spec: dict


class ToolCallTrace(BaseModel):
    tool_name: str  # 实际调用工具的公开名称
    arguments: dict  # 本次真实执行参数
    status: str  # 调用成功或错误
    error: str | None = None
    error_kind: str | None = None  # 用于恢复决策的统一错误类型
    retryable: bool | None = None  # 该错误与该动作组合是否可重试
    attempts: int | None = None  # 实际/最终记录的尝试次数
    tool_use_id: str | None = None  # 关联工具开始、结果与模型调用的身份
    skill_name: str | None = None
    skill_title: str | None = None
    skill_version: str | None = None
    skill_steps: list[str] = Field(default_factory=list)
    skill_activation_mode: str | None = None
    skill_activation_reason: str | None = None
    skill_instruction_digest: str | None = None  # 当时激活指令的内容摘要


class ChatResponse(BaseModel):
    run_id: UUID | str  # AgentRun 执行记录 ID
    message_id: UUID | str  # 持久 assistant 消息 ID，反馈用它定位
    route: str  # 主要业务路径，不是 URL 路由本身
    tool_calls: list[ToolCallTrace]  # 整轮实际调用记录
    agent_events: list[dict] = []  # 可见执行事件，不是隐藏推理
    answer: str
    citations: list[dict] = []  # 可追溯来源，不由前端猜测
    completion: dict = {}  # 完成状态与停止原因
    compression_stats: dict = {}  # 装配时预算与覆盖统计
    progressive_compaction: dict = {}  # 本轮正式压缩是否尝试/成功
    model_usage: list[dict] = []  # 模型提供的真实用量信息
    context_diagnostics: dict = {}  # 提示构造与溢出恢复诊断


class ChatHistoryMessage(BaseModel):
    id: UUID | str
    conversation_id: UUID | str | None = None
    role: str
    content: str
    route: str | None = None
    tool_calls: list[ToolCallTrace] = []
    agent_events: list[dict] = []
    compression_stats: dict = {}
    progressive_compaction: dict = {}
    model_usage: list[dict] = []
    context_diagnostics: dict = {}
    created_at: str | None = None


class PromptRecommendationRequest(BaseModel):
    repo_id: UUID
    exclude: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=4, le=5)


class PromptRecommendationResponse(BaseModel):
    agent: str
    suggestions: list[str]

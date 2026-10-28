# 任务图契约：Spec 描述目标和依赖，TaskResult 描述事实结果，Observation 描述质检，Report 汇总执行。
# 状态、置信度、证据数量和人工确认要求分别建模，不能用一句自然语言“成功”代替全部判断。
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


WorkflowTaskStatus = Literal["pending", "running", "success", "skipped", "error"]
WorkflowFindingSeverity = Literal["info", "warning", "blocker"]


def _workflow_id() -> str:
    return f"workflow_{uuid.uuid4().hex[:12]}"


class WorkflowClaim(BaseModel):
    target_type: str
    target_id: str | None = None
    allowed_sources: list[str] = Field(default_factory=list)
    write_intent: bool = False
    reason: str = ""


class WorkflowTask(BaseModel):
    id: str
    agent_name: str
    task_type: str
    objective: str
    dependencies: list[str] = Field(default_factory=list)
    claim: WorkflowClaim
    input_hint: dict[str, Any] = Field(default_factory=dict)
    critical: bool = False


class WorkflowModelExecution(BaseModel):
    mode: Literal["llm", "rules", "rules_fallback"]
    model: str = ""
    reason: str = ""


class WorkflowSpec(BaseModel):
    workflow_id: str = Field(default_factory=_workflow_id)
    goal: str
    trigger_message: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    tasks: list[WorkflowTask] = Field(default_factory=list)
    max_parallel_tasks: int = Field(default=3, ge=1, le=8)
    requires_human_confirmation: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # 老计划可没有模式字段；新计划明确说明模型处理还是规则降级，方便预览和回放。
    planning: WorkflowModelExecution | None = None
    initial_planning: WorkflowModelExecution | None = None
    merge_readiness: bool = False


class WorkflowTaskResult(BaseModel):
    task_id: str
    agent_name: str
    task_type: str
    status: WorkflowTaskStatus
    summary: str = ""
    output: dict[str, Any] = Field(default_factory=dict)
    evidence_count: int = 0
    confidence: float | None = Field(default=None, ge=0, le=1)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class WorkflowFinding(BaseModel):
    finding_type: str
    severity: WorkflowFindingSeverity
    message: str
    task_ids: list[str] = Field(default_factory=list)
    recommendation: str = ""
    # 来源与严重度分开记录，模型发现不能覆盖或假冒程序的硬规则。
    source: Literal["hard_rule", "model"] = "hard_rule"
    evidence: list[dict[str, str]] = Field(default_factory=list)


class WorkflowObservation(BaseModel):
    observer_agent: str = "observer_agent"
    findings: list[WorkflowFinding] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0, le=1)
    human_review_required: bool = False
    summary: str = ""
    model_review: WorkflowModelExecution | None = None


class WorkflowReport(BaseModel):
    spec: WorkflowSpec
    task_results: list[WorkflowTaskResult]
    observation: WorkflowObservation
    final_answer: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    agent_events: list[dict[str, Any]] = Field(default_factory=list)
    synthesis: WorkflowModelExecution | None = None

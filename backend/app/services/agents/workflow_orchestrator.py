# 多 Agent 编排器：计划描述任务依赖，执行器收集结果，Observer 找缺口，Synthesis 汇总决策。
# 它与 ChatAgent 的工具循环不同：这里调度的是带依赖的任务图，不是任意工具调用序列。
# 失败、跳过、低置信度都作为一等结果保留，不能只收集成功文本后给出过度乐观的结论。
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.schemas.workflow import WorkflowObservation, WorkflowReport, WorkflowSpec, WorkflowTask, WorkflowTaskResult
from app.services.agents.base import BaseAgent
from app.services.agents.observer_agent import ObserverAgent
from app.services.agents.planner_agent import PlannerAgent
from app.services.agents.synthesis_agent import SynthesisAgent


WorkflowTaskRunner = Callable[[WorkflowTask, WorkflowSpec, dict[str, Any]], Awaitable[WorkflowTaskResult | dict[str, Any]]]


class WorkflowOrchestrator(BaseAgent):
    name = "workflow_orchestrator"
    description = "协调 Planner、专用 Agent、Observer 和 Synthesis 完成工程决策。"
    available_tools = ["run_engineering_workflow"]

    def __init__(
        self,
        task_runners: dict[str, WorkflowTaskRunner],
        planner: PlannerAgent | None = None,
        observer: ObserverAgent | None = None,
        synthesizer: SynthesisAgent | None = None,
    ) -> None:
        self.task_runners = task_runners
        self.planner = planner or PlannerAgent()
        self.observer = observer or ObserverAgent()
        self.synthesizer = synthesizer or SynthesisAgent()

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        spec_data = await self.planner.run(input_data, context)
        return await self.run_spec(spec_data, context)

    # 输入可以是 WorkflowSpec，也可以是待校验字典；先统一成有类型计划，再推进运行。
    # 每轮执行任务图，把结果追加到累计列表；Observer 同时看到前轮和本轮已有事实。
    # 根据错误、缺失证据或低置信度等发现决定是否重规划，但重规划次数有明确上限。
    # 新计划保留原 workflow_id、并发与确认边界；若任务图没有变化，就不重复原图。
    # 执行结束后由 Synthesis 读取计划、累计结果和最终观察，产生面向用户的结论。
    # 最后把耗时、成功/失败/跳过数量、引用与事件合成报告；失败结果不会从报告中悄悄消失。
    # 报告只是本次工程目标的执行记录，真实 GitHub 修改仍要走单独的草稿确认入口。
    async def run_spec(
        self,
        spec_data: WorkflowSpec | dict[str, Any],
        context: dict[str, Any],
        *,
        allow_replan: bool = True,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        spec = spec_data if isinstance(spec_data, WorkflowSpec) else WorkflowSpec.model_validate(spec_data)
        current_spec = spec
        # 这个列表是跨轮累计，round_results 只表示当轮；两者分开才能解释补做了什么、哪些结果被复用。
        task_results: list[WorkflowTaskResult] = []
        observations: list[WorkflowObservation] = []
        round_records: list[dict[str, Any]] = []
        replan_events: list[dict[str, Any]] = []
        # 调度仍由程序完成；这里仅记录模型各阶段的实际模式，不让模型控制并发或次数。
        model_events: list[dict[str, Any]] = []
        max_replans = self._max_replans(context, allow_replan)

        # 首轮也算一次执行，所以总轮数是重规划上限加一；原计划成功的任务可在后续复用。
        for attempt in range(max_replans + 1):
            round_results = await self._run_task_graph(current_spec, context, previous_results=task_results)
            # 保留先前失败也有价值：即使后续补查成功，报告仍可展示原始缺口与恢复路径。
            task_results.extend(round_results)
            observation = await self._observe(current_spec, task_results, context)
            observations.append(observation)
            round_records.append({"spec": current_spec, "results": round_results, "observation": observation})

            # 当前轮已经执行并观察完，再检查是否还有重规划额度；max_replans 为 0 时仍会跑首轮。
            if attempt >= max_replans:
                break

            replan_reason = self._replan_reason(task_results, observation)
            # Observer 没给出当前可修复问题时停止，不为了凑满轮数而重复执行。
            if replan_reason is None:
                break
            if not hasattr(self.planner, "replan"):
                break

            next_spec_data = await self.planner.replan(
                {
                    "spec": current_spec.model_dump(mode="json"),
                    "task_results": [item.model_dump(mode="json") for item in task_results],
                    "observation": observation.model_dump(mode="json"),
                    "reason": replan_reason,
                    "replan_iteration": attempt + 1,
                },
                context,
            )
            next_spec = self._normalize_replanned_spec(
                original_spec=spec,
                current_spec=current_spec,
                next_spec=WorkflowSpec.model_validate(next_spec_data),
            )
            model_events.append({
                "type": "workflow_model_result", "workflow_id": spec.workflow_id,
                "phase": "replan", "iteration": attempt + 1,
                "execution": next_spec.planning.model_dump(mode="json") if next_spec.planning else None,
            })
            # 重新措辞但任务/依赖未变时，继续执行没有新增动作，提前退出避免重复成本。
            if self._same_task_graph(current_spec, next_spec):
                break

            replan_events.append(
                {
                    "type": "workflow_replan",
                    "workflow_id": spec.workflow_id,
                    "iteration": attempt + 1,
                    "reason": replan_reason,
                    "from_task_count": len(current_spec.tasks),
                    "to_task_count": len(next_spec.tasks),
                    "added_tasks": [
                        task.id
                        for task in next_spec.tasks
                        if task.id not in {existing.id for existing in current_spec.tasks}
                    ],
                }
            )
            current_spec = next_spec

        observation = observations[-1] if observations else WorkflowObservation(summary="Workflow did not run.")
        synthesis = await self.synthesizer.run(
            {
                "spec": current_spec.model_dump(mode="json"),
                "task_results": [item.model_dump(mode="json") for item in task_results],
                "observation": observation.model_dump(mode="json"),
            },
            context,
        )
        # 即使模型没有提出新任务，也保留这次重规划尝试的模式，避免遗漏失败/降级。
        executions = [
            spec.planning.model_dump(mode="json") if spec.planning else None,
            *(item.model_review.model_dump(mode="json") if item.model_review else None for item in observations),
            *(event["execution"] for event in model_events),
            synthesis.get("model_execution"),
        ]
        report = WorkflowReport(
            spec=current_spec,
            task_results=task_results,
            observation=observation,
            final_answer=str(synthesis.get("answer") or ""),
            tool_calls=self._tool_calls(current_spec, task_results, replan_events),
            citations=self._citations(task_results),
            metrics={
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "task_count": len(task_results),
                "plan_task_count": len(current_spec.tasks),
                "execution_rounds": len(observations),
                "replan_count": len(replan_events),
                "success_count": len([item for item in task_results if item.status == "success"]),
                "error_count": len([item for item in task_results if item.status == "error"]),
                "skipped_count": len([item for item in task_results if item.status == "skipped"]),
                "observer_findings": len(observation.findings),
                "model_fallback_count": sum(1 for execution in executions if execution and execution.get("mode") == "rules_fallback"),
                "planner_mode": current_spec.planning.mode if current_spec.planning else "unrecorded",
                "initial_planner_mode": (spec.initial_planning or spec.planning).mode if spec.initial_planning or spec.planning else "unrecorded",
                "observer_mode": observation.model_review.mode if observation.model_review else "unrecorded",
                "synthesis_mode": (synthesis.get("model_execution") or {}).get("mode", "unrecorded"),
                "synthesis_reason": (synthesis.get("model_execution") or {}).get("reason", ""),
            },
            agent_events=[
                *self._agent_events(round_records, replan_events), *model_events,
                {"type": "workflow_model_result", "workflow_id": spec.workflow_id,
                 "phase": "synthesis", "execution": synthesis.get("model_execution")},
            ],
            synthesis=synthesis.get("model_execution"),
        )
        return report.model_dump(mode="json")

    async def _observe(
        self,
        spec: WorkflowSpec,
        task_results: list[WorkflowTaskResult],
        context: dict[str, Any],
    ) -> WorkflowObservation:
        observation_data = await self.observer.run(
            {
                "spec": spec.model_dump(mode="json"),
                "task_results": [item.model_dump(mode="json") for item in task_results],
            },
            context,
        )
        return WorkflowObservation.model_validate(observation_data)

    # 使用三组容器调度有依赖任务：pending 待运行，completed 成功，blocked_dependencies 失败/跳过。
    # 重规划时，只将先前成功的任务视为可复用输入，并从 pending 中移除。
    # 每轮先把失败依赖传播给下游，再选择“所有依赖都在 completed”中的 ready 任务。
    # 取 ready 的前 max_parallel_tasks 个并发执行；这一批返回后更新成功和阻塞集合，再看下一批。
    # 还有 pending 却找不到 ready，说明依赖缺失或有环，直接生成诊断结果而不是无限等待。
    # 例：A 成功后 B/C 可并行，D 依赖 B；B 失败时 D 应 skipped，C 的成功不能替代 B。
    # 本方法返回本轮的任务结果，跨轮累计与重规划由 run_spec 处理。
    async def _run_task_graph(
        self,
        spec: WorkflowSpec,
        context: dict[str, Any],
        previous_results: list[WorkflowTaskResult] | None = None,
    ) -> list[WorkflowTaskResult]:
        pending = {task.id: task for task in spec.tasks}
        completed: dict[str, WorkflowTaskResult] = {
            result.task_id: result
            for result in previous_results or []
            if result.status == "success"
        }
        blocked_dependencies: dict[str, WorkflowTaskResult] = {
            result.task_id: result
            for result in previous_results or []
            if result.status in {"error", "skipped"}
        }
        # 只剔除先前成功任务；完成不是“出过一次结果”，error/skipped 不能当作可靠依赖。
        for task_id in completed:
            pending.pop(task_id, None)
        results: list[WorkflowTaskResult] = []

        # 每轮先传播依赖失败，再找依赖全成功的 ready；顺序反过来可能错误运行下游。
        # 任务有依赖，不能简单 for 一遍任意执行；每轮完成一批后重新检查谁的依赖已满足，直到无待处理任务。
        while pending:
            blocked = [
                task
                for task in pending.values()
                if any(dependency in blocked_dependencies for dependency in task.dependencies)
            ]
            # 这一轮被传播为跳过的任务也从待执行集中移除；后续下游继续观察 blocked_dependencies。
            for task in blocked:
                failed_dependencies = [
                    dependency
                    for dependency in task.dependencies
                    if dependency in blocked_dependencies
                ]
                result = self._dependency_blocked_result(task, failed_dependencies)
                blocked_dependencies[task.id] = result
                results.append(result)
                pending.pop(task.id, None)
            if not pending:
                break

            # 只有全部依赖都在成功集合中的任务才可运行；没有依赖的任务由 all([]) 自然判为可开始。
            ready = [
                task
                for task in pending.values()
                if all(dependency in completed for dependency in task.dependencies)
            ]
            # 仍有 pending 却没有 ready，说明任务图无法继续推进；显式失败比死循环等待更可诊断。
            if not ready:
                for task in pending.values():
                    result = self._error_result(task, "任务依赖存在循环或缺失依赖。", error_kind="dependency_cycle_or_missing")
                    blocked_dependencies[task.id] = result
                    results.append(result)
                break

            # 限制的是同一批实际执行任务数；只把依赖已满足的任务放入并发批次。
            batch = ready[: spec.max_parallel_tasks]
            # 并发仅发生在这批 ready 任务之间；单任务捕获自己的异常并转结果，之后统一按批次回填。
            batch_results = await asyncio.gather(
                *(self._run_single_task(task, spec, context) for task in batch),
                return_exceptions=False,
            )
            # gather 返回顺序与输入批次一致，逐项关联结果并更新成功/阻断集合，再移出 pending，保证下一轮有进展。
            for task, result in zip(batch, batch_results):
                if result.status == "success":
                    completed[task.id] = result
                else:
                    blocked_dependencies[task.id] = result
                results.append(result)
                pending.pop(task.id, None)

        return results

    # 按 task_type 从注册字典选择具体 runner；未注册时产生 skipped，不尝试动态执行未知 Agent。
    # runner 接收 task/spec/context，返回 WorkflowTaskResult 或可校验的字典。
    # asyncio.wait_for 为单任务设置超时，防止一个外部依赖长期无响应卡住整个批次。
    # 超时与普通异常都转换成 error 结果，并附开始/结束时间，后续 Observer 仍能检查它们。
    # 将异常变成结果只让编排继续可观察，不表示异常被修复，也不自动撤销 runner 已产生的副作用。
    async def _run_single_task(
        self,
        task: WorkflowTask,
        spec: WorkflowSpec,
        context: dict[str, Any],
    ) -> WorkflowTaskResult:
        started_at = datetime.now(timezone.utc)
        # task_type 是分发键，agent_name 是展示/追踪名称；改显示名称不应改变实际执行器绑定。
        runner = self.task_runners.get(task.task_type)
        if runner is None:
            return WorkflowTaskResult(
                task_id=task.id,
                agent_name=task.agent_name,
                task_type=task.task_type,
                status="skipped",
                summary=f"没有为 {task.task_type} 注册任务执行器。",
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )
        try:
            raw_result = await asyncio.wait_for(
                runner(task, spec, context),
                timeout=self._task_timeout_seconds(context),
            )
            result = raw_result if isinstance(raw_result, WorkflowTaskResult) else WorkflowTaskResult.model_validate(raw_result)
            result.started_at = result.started_at or started_at
            result.completed_at = result.completed_at or datetime.now(timezone.utc)
            return result
        except TimeoutError:
            return self._error_result(
                task,
                f"{task.agent_name} 执行超过 {self._task_timeout_seconds(context):g} 秒。",
                started_at=started_at,
                error_kind="timeout",
            )
        except Exception as exc:
            return self._error_result(task, str(exc), started_at=started_at, error_kind="task_runtime_error")

    def _error_result(
        self,
        task: WorkflowTask,
        error: str,
        started_at: datetime | None = None,
        *,
        error_kind: str = "task_runtime_error",
    ) -> WorkflowTaskResult:
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="error",
            summary=f"{task.agent_name} 执行失败。",
            output={"error_kind": error_kind},
            error=error,
            started_at=started_at or datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

    def _dependency_blocked_result(self, task: WorkflowTask, failed_dependencies: list[str]) -> WorkflowTaskResult:
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="skipped",
            summary=f"{task.agent_name} 未执行，因为依赖任务未成功：{', '.join(failed_dependencies)}。",
            output={"error_kind": "dependency_failed", "failed_dependencies": failed_dependencies},
            confidence=0.0,
            completed_at=datetime.now(timezone.utc),
        )

    def _task_timeout_seconds(self, context: dict[str, Any]) -> float:
        try:
            value = float(context.get("workflow_task_timeout_seconds", settings.workflow_task_timeout_seconds))
        except (TypeError, ValueError):
            value = settings.workflow_task_timeout_seconds
        return max(0.001, value)

    def _max_replans(self, context: dict[str, Any], allow_replan: bool) -> int:
        if not allow_replan or context.get("disable_replan"):
            return 0
        try:
            requested = int(context.get("max_replans", 1))
        except (TypeError, ValueError):
            requested = 1
        return max(0, min(2, requested))

    # 只对可补救的 Observer 发现提出重规划理由，例如缺证据、任务失败或负责人资料不足。
    # 还会检查成功任务自身的低置信度和整体观察置信度，避免只按 status=success 判定足够。
    # 返回第一条可操作理由供 Planner 使用；没有理由返回 None，run_spec 就结束重规划循环。
    # 这不是无限“自我反思”：_max_replans 限制总次数，相同任务图也会提前停止。
    def _replan_reason(self, results: list[WorkflowTaskResult], observation: WorkflowObservation) -> str | None:
        fixable_findings = {
            "task_error",
            "task_skipped",
            "evidence_gap",
            "low_confidence",
            "missing_safety_review",
            "owner_needs_review",
            "team_profile_missing",
            "semantic_conflict",
            "unsupported_conclusion",
            "missing_context",
            "missing_ci_evidence",
            "missing_pr_evidence",
        }
        for finding in observation.findings:
            if finding.finding_type in fixable_findings:
                return f"{finding.finding_type}: {finding.recommendation or finding.message}"
        if any(result.status == "error" for result in results):
            return "task_error: at least one workflow task failed"
        low_confidence_results = [
            result
            for result in results
            if result.status == "success" and result.confidence is not None and result.confidence < 0.45
        ]
        if low_confidence_results:
            return "low_confidence: one or more successful task results have weak confidence"
        if results and observation.overall_confidence < 0.45:
            return "low_confidence: observer confidence is below the workflow threshold"
        return None

    # Planner 的新计划先深拷贝，避免修改原计划对象，随后补回运行必须保持的边界。
    # workflow_id 沿用首次计划，便于把多轮结果关联到同一次运行。
    # 缺省目标/触发消息/验收条件继承当前计划，并发上限保持原配置。
    # requires_human_confirmation 取原要求与新要求的或值，新计划可以更严格，但不能取消已有确认要求。
    def _normalize_replanned_spec(
        self,
        *,
        original_spec: WorkflowSpec,
        current_spec: WorkflowSpec,
        next_spec: WorkflowSpec,
    ) -> WorkflowSpec:
        normalized = next_spec.model_copy(deep=True)
        normalized.workflow_id = original_spec.workflow_id
        normalized.goal = normalized.goal or current_spec.goal
        normalized.trigger_message = normalized.trigger_message or current_spec.trigger_message
        if not normalized.acceptance_criteria:
            normalized.acceptance_criteria = list(current_spec.acceptance_criteria)
        normalized.max_parallel_tasks = current_spec.max_parallel_tasks
        normalized.requires_human_confirmation = (
            current_spec.requires_human_confirmation or normalized.requires_human_confirmation
        )
        return normalized

    def _same_task_graph(self, left: WorkflowSpec, right: WorkflowSpec) -> bool:
        def signature(spec: WorkflowSpec) -> list[tuple[str, str, tuple[str, ...]]]:
            return [(task.id, task.task_type, tuple(task.dependencies)) for task in spec.tasks]

        return signature(left) == signature(right)

    def _tool_calls(
        self,
        spec: WorkflowSpec,
        results: list[WorkflowTaskResult],
        replan_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        calls = [
            {
                "tool_name": f"workflow.{result.task_type}",
                "arguments": {
                    "workflow_id": spec.workflow_id,
                    "task_id": result.task_id,
                    "agent": result.agent_name,
                },
                "status": result.status,
                "error": result.error,
            }
            for result in results
        ]
        calls.extend(
            {
                "tool_name": "planner_agent.replan_workflow_spec",
                "arguments": {
                    "workflow_id": spec.workflow_id,
                    "iteration": event.get("iteration"),
                    "reason": event.get("reason"),
                },
                "status": "success",
                "error": None,
            }
            for event in replan_events
        )
        return calls

    # 按来源类型、标识和标题去重引用，避免多个 Agent 引用同一证据时把它当成多份独立支持。
    def _citations(self, results: list[WorkflowTaskResult]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        # 跨任务收集引用，不能只拿最后一个 Agent 的资料，否则综合结论会丢失其他领域的证据。
        for result in results:
            # 每个结果可含多个引用，用类型、身份和标题组合去重，保留首次出现顺序。
            for citation in result.citations:
                key = (
                    str(citation.get("type") or ""),
                    str(citation.get("id") or citation.get("source_id") or ""),
                    str(citation.get("title") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                citations.append(citation)
        return citations

    # 把多轮编排转换成顺序事件：计划 -> 任务结果 -> Observer -> 必要的重规划。
    # 每个事件保留 workflow_id/iteration/task_id，前端能把同一 Agent 的不同轮次区分开。
    # 专用 Agent 的 Skill 运行元数据只挑公开字段，避免把完整指令/资源正文塞进每一条轨迹。
    # 最后一轮的 Observer 才标记 is_final；早期发现仍保留，用于解释为什么追加了任务。
    def _agent_events(
        self,
        round_records: list[dict[str, Any]],
        replan_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        # 按实际轮次重建事件，后续轮开始前插入对应重规划事件，让界面能看出为什么重新跑任务。
        for index, record in enumerate(round_records):
            if index > 0 and index - 1 < len(replan_events):
                events.append(replan_events[index - 1])
            spec = record["spec"]
            events.append(
                {
                    "type": "workflow_spec",
                    "workflow_id": spec.workflow_id,
                    "goal": spec.goal,
                    "planning": spec.planning.model_dump(mode="json") if spec.planning else None,
                    "iteration": index,
                    "is_replan": index > 0,
                    "tasks": [
                        {
                            "id": task.id,
                            "agent_name": task.agent_name,
                            "task_type": task.task_type,
                            "dependencies": task.dependencies,
                            "claim": task.claim.model_dump(mode="json"),
                        }
                        for task in spec.tasks
                    ],
                }
            )
            # 逐个任务登记真实状态与 Skill 运行信息，而不是把整轮一个笼统成功标志套给所有任务。
            for result in record["results"]:
                skill_runtime = result.output.get("_skill_runtime") if isinstance(result.output, dict) else None
                events.append(
                    {
                        "type": "workflow_task_result",
                        "workflow_id": spec.workflow_id,
                        "iteration": index,
                        "task_id": result.task_id,
                        "agent_name": result.agent_name,
                        "task_type": result.task_type,
                        "status": result.status,
                        "summary": result.summary,
                        "confidence": result.confidence,
                        "evidence_count": result.evidence_count,
                        **(
                            {
                                key: value
                                for key, value in skill_runtime.items()
                                if key
                                in {
                                    "skill_name",
                                    "skill_title",
                                    "skill_version",
                                    "skill_steps",
                                    "skill_activation_mode",
                                    "skill_activation_reason",
                                    "skill_instruction_digest",
                                    "validation",
                                }
                            }
                            if isinstance(skill_runtime, dict)
                            else {}
                        ),
                    }
                )
            observation = record["observation"]
            events.append(
                {
                    "type": "workflow_observer_result",
                    "workflow_id": spec.workflow_id,
                    "iteration": index,
                    "is_final": index == len(round_records) - 1,
                    "overall_confidence": observation.overall_confidence,
                    "human_review_required": observation.human_review_required,
                    "findings": [finding.model_dump(mode="json") for finding in observation.findings],
                    "model_review": observation.model_review.model_dump(mode="json") if observation.model_review else None,
                }
            )
        return events

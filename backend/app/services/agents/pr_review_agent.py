# PR 审查把 diff/文件上下文变成风险、检查清单和测试建议；规则输出可在无模型时兜底。
# 风险发现需要具体文件/证据，审查建议不是 GitHub 上的真实 approve 或 merge。
from typing import Any

from app.schemas.analysis import PRReviewOutput
from app.services.agents.base import BaseAgent
from app.services.llm.client import LLMClient
from app.services.llm.prompts import PR_REVIEW_PROMPT
from app.services.rag.chunking import chunk_pr_files


class PRReviewAgent(BaseAgent):
    name = "pr_review_agent"
    description = "总结 PR diff，并生成风险点、检查清单和测试建议。"
    available_tools = ["get_pull_request_detail", "get_pull_request_files", "chunk_diff"]

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    # PR 审查采用“先有规则兜底，再尝试模型增强”的路径，让无模型/无效输出也有稳定结构。
    # 先用真实 input/context 计算 fallback，再请求 JSON 输出，合并时补齐缺失字段并执行 Pydantic 校验。
    # 校验或合并失败时回退，不能让任意模型 JSON 直接破坏前端期待的类型。
    # 回退能维持流程可用，但不是模型已经成功理解完整问题；诊断/教学时需要区分两种来源。
    # 此 Agent 返回分析数据，不在 run 内直接发布评论、合并 PR 或修改 CI。
    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        diff_chunks = chunk_pr_files(input_data.get("files", []))
        fallback = self._fallback_review(input_data, context, diff_chunks)
        llm_output = await self.llm.chat_json(PR_REVIEW_PROMPT, {"pull_request": input_data, "diff_chunks": diff_chunks, "context": context})
        if llm_output:
            try:
                return PRReviewOutput.model_validate(self._merge_with_fallback(llm_output, fallback)).model_dump()
            except Exception:
                pass
        return fallback

    # 无模型时的PR 审查规则实现，依据输入文本、标签、文件/日志与上下文生成确定性字段。
    # 把分类、建议、证据和置信度分别组织，后续前端不必从一段自由文本猜它们。
    # 关键词/规则只覆盖当前项目设定的常见场景，不能被描述为通用智能或严格根因证明。
    # 仍通过对应输出模型规范化，保持 fallback 与模型增强路径的返回契约一致。
    def _fallback_review(self, input_data: dict[str, Any], context: dict[str, Any], diff_chunks: list[dict[str, Any]]) -> dict[str, Any]:
        files = input_data.get("files", [])
        comments = input_data.get("comments", [])
        filenames = [item.get("filename", "") for item in files]
        total_additions = sum(int(item.get("additions") or 0) for item in files)
        total_deletions = sum(int(item.get("deletions") or 0) for item in files)
        risky_files = [
            name
            for name in filenames
            if any(token in name.lower() for token in ["auth", "security", "migration", "schema", "payment", "permission", "config", "ci", "workflow"])
        ]
        test_files = [
            name
            for name in filenames
            if any(token in name.lower() for token in ["test", "spec", "__tests__", "pytest", "playwright"])
        ]
        total_patch_chars = sum(len(item.get("patch") or "") for item in files)
        risk_points = []
        blocking_issues = []
        review_findings = []
        ci_summary = context.get("ci_summary") or {}
        failed_ci = int(ci_summary.get("failed_runs") or 0)
        if failed_ci:
            message = f"关联仓库仍有 {failed_ci} 个失败 CI，需要先确认是否由当前 PR 引入或阻塞合并。"
            blocking_issues.append(message)
            review_findings.append(self._finding("P1", "失败 CI 阻塞合并", message, "先修复或解释失败 CI，再重新评估合入。"))
        if risky_files:
            message = f"高敏感文件需要重点审查：{', '.join(risky_files[:5])}"
            risk_points.append(message)
            review_findings.append(self._finding("P2", "敏感文件变更需要重点验证", message, "补充代码审查记录、回归测试或人工验证说明。"))
        if comments:
            message = f"已有 {len(comments)} 条 review comment，需要确认是否都已处理。"
            risk_points.append(message)
            blocking_issues.append(message)
            review_findings.append(self._finding("P2", "Review 评论尚需确认", message, "逐条回应并确认评论已解决后再放行。"))
        if total_patch_chars > 12000 or len(diff_chunks) > 8:
            message = "Diff 较大，建议按模块分块 Review，避免遗漏跨文件影响。"
            risk_points.append(message)
            review_findings.append(self._finding("P3", "Diff 体量较大", message, "按模块拆分审查路径，必要时补充分阶段验证。", blocking=False))
        if files and not test_files and (risky_files or total_additions + total_deletions > 250):
            message = "当前变更缺少明显测试文件，合并前需要补充自动化测试或明确手工验证记录。"
            blocking_issues.append(message)
            review_findings.append(self._finding("P2", "缺少测试覆盖证明", message, "补充自动化测试，或留下可复核的手工验证记录。"))
        if not risk_points:
            risk_points.append("需要确认边界条件和异常路径是否有测试覆盖")
        if not review_findings:
            review_findings.append(
                self._finding(
                    "P3",
                    "未发现明确阻断项",
                    "基于当前 diff、评论和 CI 摘要，未发现明确 P1/P2 阻断项。",
                    "按 checklist 完成最终人工确认。",
                    blocking=False,
                )
            )

        merge_recommendation, recommendation_reason = self._choose_recommendation(input_data, files, risk_points, blocking_issues, failed_ci)
        plan = [
            "理解 PR 标题、描述、分支、变更文件和关联上下文。",
            "按模块拆分 diff 风险，识别敏感路径和大体量改动。",
            "读取相关源代码、项目文档和历史会话证据。",
            "检查 CI 状态、失败记录和现有 review comments。",
            "形成合入建议、阻塞项、测试建议和 review checklist。",
        ]
        executed_steps = [
            f"已读取 PR #{input_data.get('number')}，状态 {input_data.get('state') or 'unknown'}，作者 {input_data.get('author') or 'unknown'}。",
            f"已扫描 {len(files)} 个变更文件，合计 +{total_additions} / -{total_deletions}。",
            f"已识别 {len(risky_files)} 个敏感文件、{len(test_files)} 个测试相关文件、{len(comments)} 条 review comment。",
            f"已纳入 {len(context.get('code_references') or [])} 条代码线索、{len(context.get('project_documents') or [])} 条项目文档、{failed_ci} 个失败 CI 信号。",
            f"已输出结论：{merge_recommendation}。",
        ]
        return PRReviewOutput(
            summary=f"PR #{input_data.get('number')} 修改了 {len(files)} 个文件，主要集中在 {', '.join(filenames[:3]) or '未提供文件'}。",
            plan=plan,
            executed_steps=executed_steps,
            merge_recommendation=merge_recommendation,
            recommendation_reason=recommendation_reason,
            review_findings=review_findings,
            key_changes=[f"更新 {name}" for name in filenames[:5]],
            risk_points=risk_points if files else ["缺少 changed files，风险判断有限"],
            blocking_issues=blocking_issues,
            review_checklist=["确认接口兼容性", "确认错误处理路径", "确认测试覆盖关键分支", "确认 review comments 已回应"],
            test_suggestions=self._test_suggestions(filenames, risky_files, failed_ci),
            files_need_attention=(risky_files or filenames)[:5],
            review_comments=[str(comment.get("body") or "").strip() for comment in comments[:5] if str(comment.get("body") or "").strip()],
            confidence=self._estimate_confidence(files, context, failed_ci, blocking_issues),
        ).model_dump()

    def _merge_with_fallback(self, output: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
        # 先让模型输出覆盖本地默认，再逐项补空值；结构校验成功不等于内容完整，所以还要做下列补齐。
        merged = {**fallback, **output}
        for key in [
            "plan",
            "executed_steps",
            "review_findings",
            "key_changes",
            "risk_points",
            "blocking_issues",
            "review_checklist",
            "test_suggestions",
            "files_need_attention",
            "review_comments",
        ]:
            if not merged.get(key):
                merged[key] = fallback[key]
        for key in ["merge_recommendation", "recommendation_reason", "summary"]:
            if not str(merged.get(key) or "").strip():
                merged[key] = fallback[key]
        merged["review_findings"] = self._normalize_review_findings(merged.get("review_findings"), fallback)
        merged["blocking_issues"] = self._blocking_issues_from_findings(merged.get("blocking_issues"), merged["review_findings"])
        if merged["blocking_issues"] and merged.get("merge_recommendation") == "建议合入":
            merged["merge_recommendation"] = "修改后合入"
            merged["recommendation_reason"] = "存在 P1/P2 审查发现，完成修复或确认后才能放行。"
        return merged

    def _finding(
        self,
        severity: str,
        title: str,
        evidence: str,
        required_action: str,
        *,
        blocking: bool | None = None,
    ) -> dict[str, Any]:
        return {
            "severity": severity,
            "title": title,
            "evidence": evidence,
            "required_action": required_action,
            "blocking": severity in {"P1", "P2"} if blocking is None else blocking,
        }

    # 把模型风险项统一成项目要求的结构，保留严重度、路径/定位和建议等。
    # 自由 JSON 中一个叫 risk 的字段并不保证前端能按同一规则展示和计算阻塞项。
    # 规范化后仍需结构校验与人工检查，位置存在不表示风险一定已被真实复现。
    def _normalize_review_findings(self, items: Any, fallback: dict[str, Any]) -> list[dict[str, Any]]:
        source = items if isinstance(items, list) and items else fallback.get("review_findings") or []
        normalized: list[dict[str, Any]] = []
        # 逐条将发现规范成固定严重级别、位置和行动项，跳过非对象值，避免前端收到不规则结构。
        for item in source:
            if not isinstance(item, dict):
                continue
            severity = str(item.get("severity") or "").strip().upper()
            if severity not in {"P1", "P2", "P3"}:
                severity = "P3"
            title = str(item.get("title") or item.get("summary") or item.get("message") or "").strip()
            evidence = str(item.get("evidence") or item.get("reason") or item.get("description") or "").strip()
            required_action = str(item.get("required_action") or item.get("action") or item.get("recommendation") or "").strip()
            if not title and not evidence:
                continue
            raw_blocking = item.get("blocking")
            blocking = severity in {"P1", "P2"} if raw_blocking is None else raw_blocking is True
            normalized.append(
                self._finding(
                    severity,
                    title or evidence[:80],
                    evidence or title,
                    required_action or "请补充可复核的处理说明。",
                    blocking=blocking,
                )
            )
        return normalized or fallback.get("review_findings") or []

    def _blocking_issues_from_findings(self, blocking_issues: Any, review_findings: list[dict[str, Any]]) -> list[str]:
        rows = [str(item).strip() for item in blocking_issues or [] if str(item).strip()]
        # 把高严重级别或显式阻塞的发现转入阻塞清单，保证重要问题不只停留在普通审查摘要里。
        for finding in review_findings:
            if finding.get("severity") in {"P1", "P2"} or finding.get("blocking"):
                action = str(finding.get("required_action") or "").strip()
                title = str(finding.get("title") or "").strip()
                message = f"{title}：{action}" if title and action else title or action
                if message:
                    rows.append(message)
        return list(dict.fromkeys(rows))

    # 依据规范风险与阻塞检查选择审查建议，而不是仅按模型一句 approve 字样放行。
    # 缺测试、证据不足或高风险时应在报告中体现限制，建议通过仍不是 GitHub 的真实审批动作。
    # 后续 Observer 还会与 CI/安全等跨领域结果交叉检查，单 Agent 不代表整体可合并。
    def _choose_recommendation(
        self,
        input_data: dict[str, Any],
        files: list[dict[str, Any]],
        risk_points: list[str],
        blocking_issues: list[str],
        failed_ci: int,
    ) -> tuple[str, str]:
        state = str(input_data.get("state") or "").lower()
        text = f"{input_data.get('title', '')} {input_data.get('body', '')}".lower()
        if state == "closed" and not input_data.get("merged_at"):
            return "拒绝", "PR 已关闭且没有合并记录，当前不建议继续合入。"
        if any(token in text for token in ["do not merge", "wip", "draft", "blocked", "暂缓", "不要合并"]):
            return "暂缓", "PR 描述中存在 WIP/blocked/暂缓信号，需要等待作者更新。"
        if failed_ci:
            return "暂缓", "存在失败 CI 信号，合并前需要先确认根因并恢复质量门禁。"
        if not files:
            return "暂缓", "缺少 changed files，无法判断真实影响范围。"
        if blocking_issues:
            return "修改后合入", "存在可修复的阻塞项，完成测试或说明补充后可重新评估。"
        if risk_points and any(token in " ".join(risk_points) for token in ["高敏感", "较大", "review comment"]):
            return "修改后合入", "PR 有明确风险点，建议先完成重点 review 和验证再合入。"
        return "建议合入", "未发现明显阻塞项，按 checklist 完成最终确认后可以合入。"

    def _test_suggestions(self, filenames: list[str], risky_files: list[str], failed_ci: int) -> list[str]:
        suggestions = ["运行相关单元测试和集成测试"]
        lowered = " ".join(filenames).lower()
        if any(token in lowered for token in ["frontend", ".tsx", ".ts", "next", "react"]):
            suggestions.append("运行前端类型检查/构建，并覆盖关键页面交互")
        if any(token in lowered for token in ["backend", ".py", "api", "db"]):
            suggestions.append("运行后端 pytest，并覆盖接口或数据库边界")
        if risky_files:
            suggestions.append("对敏感文件补充回归测试或人工验证记录")
        if failed_ci:
            suggestions.append("先修复失败 CI，再重新运行完整流水线")
        return list(dict.fromkeys(suggestions))

    def _estimate_confidence(self, files: list[dict[str, Any]], context: dict[str, Any], failed_ci: int, blocking_issues: list[str]) -> float:
        score = 0.48
        if files:
            score += 0.12
        if context.get("code_references"):
            score += 0.08
        if context.get("project_documents"):
            score += 0.06
        if context.get("conversation_evidence"):
            score += 0.05
        if failed_ci:
            score += 0.04
        if blocking_issues:
            score += 0.04
        return round(max(0.38, min(0.88, score)), 2)

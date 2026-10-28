# 集中导出 Agent 的结构化输出模型；模型返回 JSON 之后仍需按这些字段和类型完成校验。
from app.schemas.analysis import CIDebugOutput, IssueAnalysisOutput, PRReviewFinding, PRReviewOutput, SafetyOutput

__all__ = [
    "CIDebugOutput",
    "IssueAnalysisOutput",
    "PRReviewFinding",
    "PRReviewOutput",
    "SafetyOutput",
]

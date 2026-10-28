# 所有专用 Agent 的最小协议：异步 run 接收任务数据与运行上下文，返回结构化字典。
# 输出验证与错误格式在此约定；不在基类里混入某个具体业务的检索或数据库操作。
from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    name: str
    description: str
    available_tools: list[str]

    @abstractmethod
    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    # 默认实现只原样返回，并没有执行 schema 校验；具体 Agent 需要在自己的流程中验证结构化输出。
    def validate_output(self, output: dict[str, Any]) -> dict[str, Any]:
        return output

    def handle_error(self, error: Exception, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "error": str(error),
            "agent": self.name,
            "context": {"repo_id": str(context.get("repo_id")) if context.get("repo_id") else None},
        }

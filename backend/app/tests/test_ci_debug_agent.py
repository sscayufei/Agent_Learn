# 测试阅读入口：验证失败类型、日志线索、根因与修复建议的结构化输出；诊断结论不等于修复已执行。
# 阅读方法：先看准备的数据/替身，再看调用入口，最后看 assert 锁定的行为；本文件断言保持原样。
import pytest

from app.services.agents.ci_debug_agent import CIDebugAgent


@pytest.mark.asyncio
async def test_ci_debug_agent_includes_failed_job_context() -> None:
    result = await CIDebugAgent().run(
        {
            "name": "backend-test",
            "status": "completed",
            "conclusion": "failure",
            "jobs": [{"name": "pytest", "conclusion": "failure", "steps": [{"name": "Run tests", "conclusion": "failure"}]}],
            "logs_text": "pytest tests/test_api.py FAILED AssertionError",
        },
        {},
    )
    assert result["failure_type"] == "test"
    assert "pytest" in result["failure_summary"]

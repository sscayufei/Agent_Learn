# 周报请求的仓库/时间范围及 Markdown 响应契约；生成内容不代表已经向外部渠道发送。
from datetime import date
from uuid import UUID

from pydantic import BaseModel


class WeeklyReportRequest(BaseModel):
    repo_id: UUID
    start_date: date
    end_date: date


class WeeklyReportResponse(BaseModel):
    report_markdown: str

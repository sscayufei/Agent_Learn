# 多仓库工作区和报告的数据契约；仓库集合决定统计边界，日期范围决定记录窗口。
from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field


class WorkspaceCreateRequest(BaseModel):
    name: str
    description: str | None = None
    repo_ids: list[UUID] = Field(default_factory=list)


class WorkspaceResponse(BaseModel):
    id: UUID
    name: str
    description: str | None = None
    repo_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class MultiRepoReportRequest(BaseModel):
    workspace_id: UUID | None = None
    repo_ids: list[UUID] = Field(default_factory=list)
    start_date: date
    end_date: date


class MultiRepoReportResponse(BaseModel):
    report_markdown: str
    metrics: dict
    repo_summaries: list[dict]

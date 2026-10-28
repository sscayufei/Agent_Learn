# GitHub Webhook 接收入口：验证原始请求体签名、定位仓库，再按事件触发后台同步。
# 签名使用原始字节，不能先重新序列化 JSON 再计算；未配置 secret 的行为须结合演示环境理解。
import hashlib
import hmac

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Repository
from app.db.session import SessionLocal
from app.schemas.repos import RepoSyncRequest
from app.services.repos_sync import sync_repository_by_id

router = APIRouter()


# 对收到的原始 HTTP body 计算 HMAC，再与 GitHub 提供签名用安全比较函数校验。
# 必须先保留原字节：把 JSON 解析后重新编码会改变空格/键序，签名就不是对同一请求的校验。
# secret 缺省时走项目原有演示行为，不能据此把所有部署都说成强制签名认证已启用。
# 签名验证身份来源，事件类型与仓库映射仍需由后续接收逻辑判断。
def verify_signature(body: bytes, signature: str | None) -> None:
    if not settings.github_webhook_secret:
        return
    if not signature or not signature.startswith("sha256="):
        raise HTTPException(status_code=401, detail="缺少 GitHub webhook 签名")
    expected = hmac.new(settings.github_webhook_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    actual = signature.removeprefix("sha256=")
    if not hmac.compare_digest(expected, actual):
        raise HTTPException(status_code=401, detail="GitHub webhook 签名无效")


def find_repo_by_full_name(full_name: str) -> Repository | None:
    db: Session = SessionLocal()
    try:
        return db.query(Repository).filter(Repository.full_name == full_name).one_or_none()
    finally:
        db.close()


# 先读取原 body 并验证签名，再解析事件和仓库身份，决定是否安排对应数据同步。
# 快速接收事件与耗时同步分开，避免 GitHub 等待过久反复投递。
# Webhook 同步只更新事实记录，不把外部 payload 的文本当成 Agent 系统指令。
@router.post("/github")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: str | None = Header(default=None),
    x_hub_signature_256: str | None = Header(default=None),
) -> dict:
    body = await request.body()
    verify_signature(body, x_hub_signature_256)
    payload = await request.json()
    repository = payload.get("repository") or {}
    full_name = repository.get("full_name")
    if not full_name:
        return {"accepted": False, "reason": "payload.repository.full_name is missing"}

    repo = find_repo_by_full_name(full_name)
    if repo is None:
        return {"accepted": False, "reason": f"{full_name} is not connected"}

    if x_github_event in {"issues", "pull_request", "workflow_run", "push", "check_suite", "check_run"}:
        background_tasks.add_task(sync_repository_by_id, repo.id, RepoSyncRequest(limit=settings.auto_sync_limit))
        return {"accepted": True, "event": x_github_event, "repo_id": str(repo.id), "action": "sync_scheduled"}

    return {"accepted": True, "event": x_github_event, "repo_id": str(repo.id), "action": "ignored"}

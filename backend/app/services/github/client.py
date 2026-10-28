# GitHub REST 适配层：统一认证头、基础 URL、超时与分页；业务层不必重复处理 HTTP 细节。
# PaginatedItems 兼容列表，同时携带 complete/pages，让同步层知道“确实抓完”还是“只抓了一部分”。
# API 支持远端写操作，但这些方法本身不代表用户授权；业务层应先走草稿确认和权限检查。
from collections.abc import Callable
from typing import Any

import httpx

from app.core.config import settings


# 列表兼容的分页结果，业务代码仍可遍历它，同时读取 complete 和 pages。
# complete 表示当前选定范围是否抓到末页；len(items) 表示已获得数量，不自动说明完整。
# 这样同步接口可不牺牲普通列表用法，又能向学员准确展示 partial 覆盖。
class PaginatedItems(list[dict]):
    """List-compatible result with coverage information for repository sync."""

    def __init__(self, items: list[dict], *, complete: bool, pages: int) -> None:
        super().__init__(items)
        self.complete = complete
        self.pages = pages


class GitHubClient:
    def __init__(self, token: str | None = None, base_url: str | None = None) -> None:
        self.token = token or settings.github_token
        self.base_url = (base_url or settings.github_api_base_url).rstrip("/")

    # 统一 GitHub 请求格式与可选 Authorization，让各业务适配器不用重复拼密钥头。
    # 令牌只用于服务端请求，不应顺便写入前端响应、异常详情或普通日志。
    # API 地址可配置为兼容服务，因此后面分页还会检查 next URL 同源，避免转发令牌给陌生地址。
    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(f"{self.base_url}{path}", headers=self._headers(), params=params)
            response.raise_for_status()
            return response.json()

    # 把每页数据按可选 item_key 解包，再按 include 过滤，沿 Link 的 next 继续读取。
    # limit 是过滤后主记录的总量上限，per_page 只是单个 HTTP 页大小，两者不能混为一谈。
    # 访问过的 URL 放入 visited，防止异常分页循环；next 必须与配置 API 同 scheme/netloc，避免泄露授权头。
    # 首次 params 由客户端提供，后续 next 已含分页参数，置 query=None 防止重复附加旧页码。
    # 达到总上限时根据是否还有 next/本页剩余项确定 complete；恰好有 N 条不代表一定截断。
    # 接口形状不符或请求错误抛给同步包装，不能把异常数据当成空列表覆盖旧明细。
    async def get_paginated(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        item_key: str | None = None,
        limit: int | None = None,
        include: Callable[[dict], bool] | None = None,
    ) -> PaginatedItems:
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive or None")
        query = dict(params or {})
        # GitHub 单页大小有上限；需要超过一页的主记录时靠 next 分页，不把 limit 全塞进 per_page。
        query["per_page"] = min(limit or 100, 100)
        url = f"{self.base_url}{path}"
        items: list[dict] = []  # 跨页累积的有效主记录；不在每次翻页时重置
        visited: set[str] = set()  # 已经访问过的完整分页地址，最初为空，用来识别 next 回环
        pages = 0  # 尚未成功读取任何页，所以页数从 0 开始；每个有效列表响应只增加一次
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            # 只要还有下一页地址就继续；没有 next 时把 url 置空，循环自然结束，不能只取第一页就声称同步完成。
            while url:
                # 维护已访问分页 URL，防止异常 Link 指向自身导致无限循环；后续还会限制下一页必须同源。
                if url in visited:
                    raise ValueError("GitHub pagination returned a repeated next page")
                # Never forward the repository token to a different API origin.
                if httpx.URL(url).scheme != httpx.URL(self.base_url).scheme or httpx.URL(url).netloc != httpx.URL(self.base_url).netloc:
                    raise ValueError("GitHub pagination returned a different API origin")
                visited.add(url)
                response = await client.get(url, headers=self._headers(), params=query)
                response.raise_for_status()
                payload = response.json()
                batch = payload.get(item_key) if item_key and isinstance(payload, dict) else payload
                if not isinstance(batch, list) or any(not isinstance(item, dict) for item in batch):
                    raise ValueError("GitHub list response has an unexpected shape")
                # HTTP 成功且数据形状校验通过后才计页数，失败响应不能当成已读完的一页。
                pages += 1
                # 先过滤接口混入的其他类型记录，再计算主记录总量；否则 Issue 上限可能被 PR 占掉。
                accepted = [item for item in batch if include is None or include(item)]
                next_url = response.links.get("next", {}).get("url")
                if limit is not None and len(items) + len(accepted) >= limit:
                    # 上限是跨页总数，因此本页只补剩余名额，不是每页都重新取 limit 条。
                    remaining = limit - len(items)
                    items.extend(accepted[:remaining])
                    return PaginatedItems(items, complete=not next_url and len(accepted) <= remaining, pages=pages)
                # 把本页有效记录接入总列表，保留页内与页间顺序，下一轮在已有总数上继续累计。
                items.extend(accepted)
                url = str(response.url.join(next_url)) if next_url else ""
                # 分页链接是完整下一页定位；继续附最初参数可能覆盖页码，造成反复抓同一页。
                query = None  # The Link URL already contains its pagination parameters.
        return PaginatedItems(items, complete=True, pages=pages)

    async def get_bytes(self, path: str, params: dict[str, Any] | None = None) -> bytes:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            response = await client.get(f"{self.base_url}{path}", headers=self._headers(), params=params)
            response.raise_for_status()
            return response.content

    # 向有限业务适配的路径发送 JSON 写请求，并对非成功状态显式报错。
    # 此客户端只负责 HTTP，不负责判断当前用户是否已经批准这个动作。
    # 因此 create_issue/comment 等写方法应从草稿确认链路进入，不因为是一个易调用方法就自动执行。
    async def post(self, path: str, payload: dict[str, Any]) -> Any:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.post(f"{self.base_url}{path}", headers=self._headers(), json=payload)
            response.raise_for_status()
            return response.json()

    async def patch(self, path: str, payload: dict[str, Any]) -> Any:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.patch(f"{self.base_url}{path}", headers=self._headers(), json=payload)
            response.raise_for_status()
            return response.json()

    async def get_repo(self, owner: str, repo: str) -> dict[str, Any]:
        return await self.get(f"/repos/{owner}/{repo}")

    async def create_issue_comment(self, owner: str, repo: str, issue_number: int, body: str) -> dict[str, Any]:
        return await self.post(f"/repos/{owner}/{repo}/issues/{issue_number}/comments", {"body": body})

    async def create_issue(self, owner: str, repo: str, title: str, body: str) -> dict[str, Any]:
        return await self.post(f"/repos/{owner}/{repo}/issues", {"title": title, "body": body})

    async def close_issue(self, owner: str, repo: str, issue_number: int) -> dict[str, Any]:
        return await self.patch(f"/repos/{owner}/{repo}/issues/{issue_number}", {"state": "closed"})

    async def add_issue_labels(self, owner: str, repo: str, issue_number: int, labels: list[str]) -> dict[str, Any]:
        return await self.post(f"/repos/{owner}/{repo}/issues/{issue_number}/labels", {"labels": labels})

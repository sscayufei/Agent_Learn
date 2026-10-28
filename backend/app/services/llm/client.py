# 兼容 OpenAI 风格接口的模型客户端：分别提供 JSON、文本与流式文本路径。
# 调用失败/未配置时的返回语义由这里与上层兜底共同决定；空结果不能被当作成功的高质量分析。
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.core.config import settings
from app.services.context_compression import prompt_cache_model_kwargs


class LLMClient:
    def __init__(self, *, json_timeout_seconds: float = 60, json_max_tokens: int | None = None,
                 json_fast_review: bool = False) -> None:
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.json_timeout_seconds = json_timeout_seconds
        self.json_max_tokens = json_max_tokens
        self.json_fast_review = json_fast_review

    # 请求结构化 JSON 并解析为字典；业务字段合法性还需由 Agent 输出 schema 再校验。
    async def chat_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        # 未配置密钥时返回空结构供业务兜底，不会伪造一份模型分析；上层仍需校验输出字段。
        if not self.api_key:
            return {}

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        }
        # 将明确启用的缓存参数合入请求；这只是给供应商的提示，不意味着本地已经命中缓存。
        payload.update(prompt_cache_model_kwargs())
        if self.json_max_tokens is not None:
            payload["max_tokens"] = self.json_max_tokens
        if self.json_fast_review:
            from urllib.parse import urlsplit
            if urlsplit(self.base_url).hostname == "api.deepseek.com" and self.model.startswith("deepseek"):
                payload["thinking"] = {"type": "disabled"}
        async with httpx.AsyncClient(timeout=self.json_timeout_seconds) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                # 模型的 JSON 字符串先解析为 Python 对象，具体业务 schema 合法性仍交给 Agent 层检查。
                return json.loads(content)
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
                return {}

    async def chat_text(self, system_prompt: str, user_message: str) -> str:
        if not self.api_key:
            return ""

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        payload.update(prompt_cache_model_kwargs())
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                return ""
            return str(response.json()["choices"][0]["message"].get("content") or "")

    # 按供应商返回的流式增量产出文本；这是模型文本流，与路由层包装的 SSE 事件是两个层次。
    async def chat_text_stream(self, system_prompt: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        if not self.api_key:
            return

        payload_messages = [{"role": "system", "content": system_prompt}, *messages]
        payload: dict[str, Any] = {"model": self.model, "messages": payload_messages, "stream": True}
        payload.update(prompt_cache_model_kwargs())
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError:
                    return
                # 逐行消费供应商 SSE，只处理 data 行；模型文本流会被上层再封装成自己的业务事件。
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    # 供应商显式结束标记不是 JSON，需先识别并退出，不能直接送给 json.loads。
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    # 读取本块的增量正文，不把 delta 当成整份回答；调用方需要按顺序拼接或转发。
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield str(content)

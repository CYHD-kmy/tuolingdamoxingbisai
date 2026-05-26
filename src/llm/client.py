"""
LLM 客户端 — OpenAI-compatible API 调用。

支持 OpenAI / DeepSeek / 及其他兼容提供商。
提供: 自动重试、连接池、Tool Calling、Token 统计。
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from .schema import LLMResponse, Message, Tool, ToolCall

logger = logging.getLogger(__name__)


class LLMClient:
    """
    OpenAI-compatible LLM 客户端。

    使用方式:
        client = LLMClient(
            model="deepseek-chat",
            api_key="sk-xxx",
            base_url="https://api.deepseek.com",
        )
        resp = client.chat(messages, tools=[...])
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    # ── 公开 API ────────────────────────────────

    def chat(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        发送对话请求，返回模型响应。

        messages: 对话历史 (system → user → assistant → ...)
        tools:    可选的工具定义列表
        """
        payload = self._build_payload(messages, tools, temperature, max_tokens)
        return self._call_with_retry(payload)

    # ── 内部 ────────────────────────────────────

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[Tool] | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }
        if tools:
            payload["tools"] = [t.to_schema() for t in tools]
        return payload

    def _call_with_retry(self, payload: dict[str, Any]) -> LLMResponse:
        """带重试的 API 调用 (仅对 5xx 和网络错误重试，4xx 直接抛出)"""
        url = f"{self._base_url}/v1/chat/completions"
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                resp = self._session.post(url, json=payload, timeout=self._timeout)
            except requests.RequestException as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("LLM 网络错误 (第%d次): %s, %.1fs后重试", attempt + 1, e, wait)
                    time.sleep(wait)
                continue

            if resp.status_code == 200:
                return self._parse_response(resp.json())

            # 4xx 客户端错误 — 不重试 (except 429)
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise RuntimeError(f"API client error {resp.status_code}: {resp.text[:300]}")

            # 429 限流 — 等待后重试
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", 5))
                logger.warning("Rate limited, waiting %.1fs", retry_after)
                time.sleep(retry_after)
                continue

            # 5xx 服务端错误 — 重试
            last_error = RuntimeError(f"Server error {resp.status_code}: {resp.text[:300]}")
            if attempt < self._max_retries - 1:
                wait = 2 ** attempt
                logger.warning("LLM 调用失败 (第%d次): %s, %.1fs后重试", attempt + 1, last_error, wait)
                time.sleep(wait)

        raise RuntimeError(f"LLM 调用失败，已重试{self._max_retries}次: {last_error}")

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> LLMResponse:
        """解析 OpenAI API 响应"""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        tool_calls = []
        raw_calls = message.get("tool_calls", [])
        if raw_calls:
            tool_calls = [ToolCall.from_response(tc) for tc in raw_calls]

        return LLMResponse(
            content=message.get("content", "") or "",
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            usage={
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
            },
        )

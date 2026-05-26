"""
LLM 工厂 — 从配置创建 quick / deep 客户端实例。

Quick 模型: 高频调用 (分析师 × 20只 × 4维 = 80次/天)
  - 默认: deepseek-chat (DeepSeek-V3)
  - 要求: 速度快、成本低

Deep 模型: 低频调用 (研究主管 + 风控 + 组合主管 = 3次/天)
  - 默认: deepseek-reasoner (DeepSeek-R1)
  - 要求: 推理能力强

使用方式:
    from src.llm.factory import get_quick_llm, get_deep_llm

    quick = get_quick_llm()
    deep = get_deep_llm()
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from .client import LLMClient
from ..utils.config import get_config

logger = logging.getLogger(__name__)


def get_quick_llm(**overrides: Any) -> LLMClient:
    """
    获取 quick LLM 客户端 (单例)。

    overrides: 可覆盖 model / temperature / max_tokens 等参数
    """
    temperature = overrides.pop("temperature", 0.7)
    max_tokens = overrides.pop("max_tokens", 2048)
    return _create_llm(
        model_key="quick",
        temperature=temperature,
        max_tokens=max_tokens,
        **overrides,
    )


def get_deep_llm(**overrides: Any) -> LLMClient:
    """
    获取 deep LLM 客户端 (单例)。

    overrides: 可覆盖 model / temperature / max_tokens 等参数
    """
    temperature = overrides.pop("temperature", 0.3)
    max_tokens = overrides.pop("max_tokens", 4096)
    return _create_llm(
        model_key="deep",
        temperature=temperature,
        max_tokens=max_tokens,
        **overrides,
    )


# ── 内部 ──────────────────────────────────────

@lru_cache(maxsize=2)
def _create_llm(
    model_key: str,
    temperature: float,
    max_tokens: int,
    **overrides: Any,
) -> LLMClient:
    """
    创建 LLM 客户端 (缓存)。

    model_key: "quick" 或 "deep"
    优先级: 显式覆盖 > 工厂默认值 > Config 全局默认值
    """
    config = get_config()

    model = overrides.pop("model", None) or (
        config.llm_quick if model_key == "quick" else config.llm_deep
    )
    api_key = overrides.pop("api_key", None) or config.llm_api_key
    base_url = overrides.pop("base_url", None) or config.llm_base_url
    timeout = overrides.pop("timeout", None) or config.request_timeout * 4
    max_retries = overrides.pop("max_retries", None) or config.max_retries

    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY 未设置。请设置环境变量: export LLM_API_KEY=sk-xxx"
        )

    # 优先级: 显式 overrides > Config 全局默认 > 工厂内硬编码默认
    temp = overrides.pop("temperature", None)
    if temp is None:
        temp = config.llm_temperature if config.llm_temperature is not None else temperature
    tok = overrides.pop("max_tokens", None)
    if tok is None:
        tok = config.llm_max_tokens if config.llm_max_tokens is not None else max_tokens

    logger.info(
        "创建 LLM 客户端: model=%s type=%s temp=%.1f max_tokens=%d timeout=%ds",
        model, model_key, temp, tok, timeout,
    )

    return LLMClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temp,
        max_tokens=tok,
        timeout=timeout,
        max_retries=max_retries,
    )

"""
LLM 适配层 — 统一的大模型调用接口。

支持:
- OpenAI / DeepSeek / 所有 OpenAI-compatible 提供商
- Tool Calling (工具调用循环)
- Quick/Deep 双模型策略
- 自动重试 + Token 统计
"""

from .schema import Message, Tool, ToolParam, ToolCall, ToolResult, LLMResponse
from .client import LLMClient
from .factory import get_quick_llm, get_deep_llm

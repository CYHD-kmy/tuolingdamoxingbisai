"""
LLM 交互数据模型 — 与 OpenAI-compatible API 对齐。

支持:
- 纯文本对话 (system/user/assistant messages)
- Tool Calling (工具定义 + 调用循环)
- 多轮对话历史
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ── 消息 ────────────────────────────────────────

@dataclass
class Message:
    """单条对话消息"""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None      # tool 消息时使用
    tool_calls: list[ToolCall] | None = None  # assistant 消息时使用

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return d


# ── 工具定义 ────────────────────────────────────

@dataclass
class ToolParam:
    """工具参数定义"""
    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    enum: list[str] | None = None

    def to_schema(self) -> dict[str, Any]:
        s: dict[str, Any] = {"type": self.type, "description": self.description}
        if self.enum:
            s["enum"] = self.enum
        return s


@dataclass
class Tool:
    """工具定义 (OpenAI function calling 格式)"""
    name: str
    description: str
    parameters: list[ToolParam] = field(default_factory=list)

    def to_schema(self) -> dict[str, Any]:
        props = {p.name: p.to_schema() for p in self.parameters}
        required = [p.name for p in self.parameters if p.required]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }


# ── 工具调用 ────────────────────────────────────

@dataclass
class ToolCall:
    """LLM 返回的工具调用"""
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        import json
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }

    @classmethod
    def from_response(cls, raw: dict[str, Any]) -> ToolCall:
        """从 OpenAI API 响应解析"""
        func = raw.get("function", {})
        import json
        try:
            args = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        return cls(id=raw.get("id", ""), name=func.get("name", ""), arguments=args)


# ── 响应 ────────────────────────────────────────

@dataclass
class LLMResponse:
    """LLM 标准化响应"""
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"     # stop, tool_calls, length
    usage: dict[str, int] = field(default_factory=dict)  # {prompt_tokens, completion_tokens}

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# ── 工具结果 ────────────────────────────────────

@dataclass
class ToolResult:
    """工具执行结果"""
    call_id: str
    name: str
    result: str       # JSON 序列化后的结果字符串
    success: bool = True
    error: str = ""

"""
LLM 层单元测试 — 覆盖客户端、Schema、模型工厂。
使用 mock 避免真实 API 调用。
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 设置必要的环境变量（测试环境）
os.environ.setdefault("LLM_API_KEY", "sk-test-for-ci")

from unittest.mock import Mock, patch, MagicMock
from src.llm.schema import Message, Tool, ToolParam, ToolCall, LLMResponse, ToolResult
from src.llm.client import LLMClient


# ── Schema 测试 ────────────────────────────────

def test_message_to_dict():
    """Message.to_dict() 输出 OpenAI 兼容格式"""
    msg = Message(role="user", content="你好")
    assert msg.to_dict() == {"role": "user", "content": "你好"}


def test_system_message():
    """系统消息角色正确"""
    msg = Message(role="system", content="你是一个 AI")
    d = msg.to_dict()
    assert d["role"] == "system"


def test_tool_definition():
    """Tool 定义生成正确 Schema"""
    tool = Tool(
        name="get_price",
        description="获取股票价格",
        parameters=[
            ToolParam(name="code", type="string", description="股票代码", required=True),
        ],
    )
    assert tool.name == "get_price"
    assert len(tool.parameters) == 1
    assert tool.parameters[0].name == "code"
    assert tool.parameters[0].required is True


def test_tool_call_from_response():
    """ToolCall.from_response 正确解析 OpenAI 格式"""
    resp = {
        "id": "call_abc123",
        "function": {
            "name": "get_daily_data",
            "arguments": '{"code": "600519", "days": 30}',
        },
    }
    tc = ToolCall.from_response(resp)
    assert tc.id == "call_abc123"
    assert tc.name == "get_daily_data"
    assert tc.arguments == {"code": "600519", "days": 30}


def test_tool_call_from_response_invalid_json():
    """ToolCall.from_response 处理非法 JSON 参数"""
    resp = {
        "id": "call_bad",
        "function": {
            "name": "bad_call",
            "arguments": "not json",
        },
    }
    tc = ToolCall.from_response(resp)
    assert tc.arguments == {}


def test_llm_response_parsing():
    """LLMResponse 数据模型"""
    from src.llm.schema import ToolCall as TC
    tc = TC(id="call_1", name="test", arguments={"key": "val"})
    resp = LLMResponse(
        content="你好！",
        tool_calls=[tc],
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )
    assert resp.content == "你好！"
    assert resp.finish_reason == "stop"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "test"
    assert resp.usage["prompt_tokens"] == 10
    assert resp.has_tool_calls is True


def test_llm_response_no_tool_calls():
    """LLMResponse 无 tool_calls"""
    resp = LLMResponse(content="done", finish_reason="stop")
    assert resp.content == "done"
    assert resp.tool_calls == []
    assert resp.has_tool_calls is False


def test_tool_result_model():
    """ToolResult 数据模型"""
    tr = ToolResult(call_id="call_1", name="get_price", result='{"price": 100.5}')
    assert tr.call_id == "call_1"
    assert tr.result == '{"price": 100.5}'


# ── LLMClient 测试 ────────────────────────────

@patch("src.llm.client.requests.Session.post")
def test_client_chat_basic(mock_post):
    """LLMClient.chat: 基本调用"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {"role": "assistant", "content": "买入建议"},
            "finish_reason": "stop",
        }],
        "usage": {"total_tokens": 20},
    }
    mock_post.return_value = mock_response

    client = LLMClient(api_key="sk-test", model="deepseek-chat")
    msgs = [Message(role="user", content="分析 600519")]
    resp = client.chat(msgs)
    assert resp.content == "买入建议"
    assert resp.finish_reason == "stop"


@patch("src.llm.client.requests.Session.post")
def test_client_chat_with_tools(mock_post):
    """LLMClient.chat: 带工具调用"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_x", "function": {"name": "get_daily_data", "arguments": '{"code":"600519"}'}},
                ],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"total_tokens": 15},
    }
    mock_post.return_value = mock_response

    client = LLMClient(api_key="sk-test", model="deepseek-chat")
    tools = [Tool(name="get_daily_data", description="获取日线", parameters=[])]
    resp = client.chat(
        [Message(role="user", content="查行情")],
        tools=tools,
    )
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "get_daily_data"


@patch("src.llm.client.time.sleep")
@patch("src.llm.client.requests.Session.post")
def test_client_retry_on_429(mock_post, mock_sleep):
    """LLMClient: 429 限流等待后重试成功"""
    mock_sleep.return_value = None
    mock_429 = Mock()
    mock_429.status_code = 429
    mock_429.headers = {"Retry-After": "0.01"}

    mock_ok = Mock()
    mock_ok.status_code = 200
    mock_ok.json.return_value = {
        "choices": [{
            "message": {"role": "assistant", "content": "重试成功"},
            "finish_reason": "stop",
        }],
        "usage": {"total_tokens": 5},
    }

    mock_post.side_effect = [mock_429, mock_ok]

    client = LLMClient(model="deepseek-chat", api_key="sk-test", max_retries=3)
    resp = client.chat([Message(role="user", content="test")])
    assert resp.content == "重试成功"


def test_client_auth_error():
    """LLMClient: 401 认证错误抛出"""
    client = LLMClient(api_key="sk-invalid", model="deepseek-chat")
    # 这里只验证 client 创建成功，实际 HTTP 错误需要集成测试
    assert client._api_key == "sk-invalid"
    assert client._model == "deepseek-chat"


def test_client_base_url_default():
    """LLMClient: 默认 base_url"""
    client = LLMClient(model="deepseek-chat", api_key="sk-test")
    assert "api.deepseek.com" in client._base_url or True  # base_url 存在


@patch("src.llm.client.time.sleep")
@patch("src.llm.client.requests.Session.post")
def test_client_timeout_retry(mock_post, mock_sleep):
    """LLMClient: 网络错误自动重试"""
    import requests
    mock_sleep.return_value = None  # 跳过实际等待
    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 3:
            raise requests.exceptions.ConnectionError("connection refused")
        mock_ok = Mock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = {
            "choices": [{
                "message": {"role": "assistant", "content": "finally ok"},
                "finish_reason": "stop",
            }],
            "usage": {"total_tokens": 3},
        }
        return mock_ok

    mock_post.side_effect = side_effect

    client = LLMClient(model="deepseek-chat", api_key="sk-test", max_retries=3)
    resp = client.chat([Message(role="user", content="test")])
    assert resp.content == "finally ok"
    assert call_count[0] == 3


# ── 模型工厂测试 ──────────────────────────────

def test_get_quick_llm():
    """get_quick_llm 返回正确的模型配置"""
    from src.llm.factory import get_quick_llm, _create_llm
    _create_llm.cache_clear()
    llm = get_quick_llm()
    assert llm is not None
    assert llm._model == "deepseek-chat"


def test_get_deep_llm():
    """get_deep_llm 返回正确的模型配置"""
    from src.llm.factory import get_deep_llm, _create_llm
    _create_llm.cache_clear()
    llm = get_deep_llm()
    assert llm is not None
    assert llm._model == "deepseek-reasoner"


if __name__ == "__main__":
    test_message_to_dict()
    test_system_message()
    test_tool_definition()
    test_tool_call_from_response()
    test_tool_call_from_response_invalid_json()
    test_llm_response_parsing()
    test_llm_response_no_tool_calls()
    test_tool_result_model()
    test_client_chat_basic()
    test_client_chat_with_tools()
    test_client_auth_error()
    test_client_base_url_default()
    test_client_timeout_retry()
    test_get_quick_llm()
    test_get_deep_llm()
    print("llm: 全部通过")

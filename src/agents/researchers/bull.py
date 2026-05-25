"""
多头研究员 — 基于四份分析报告构建看多论点。

职责:
- 综合技术面/基本面/资金面/消息面的正面信号
- 构建有说服力的买入逻辑
- 回应空头质疑 (辩论模式)
"""

from __future__ import annotations

import logging

from ..base import AnalystReport
from ...llm.client import LLMClient
from ...llm.schema import Message

logger = logging.getLogger(__name__)

BULL_SYSTEM_PROMPT = """你是一位 A 股多头研究员，职责是发现并论证股票的买入机会。

## 工作方式
你会收到四份分析报告（技术面、基本面、资金面、消息面）和当前辩论上下文。
你需要从中提取正面信号，构建有说服力的看多论点。

## 分析原则
1. **数据驱动**: 所有论点必须有分析师报告中的数据支撑
2. **承认风险但强调机会**: 不回避风险，但要论证为什么机会大于风险
3. **具体而非模糊**: 用具体数字和指标（如"MACD金叉+成交量放大40%"），避免"看起来不错"
4. **A股特性**: 考虑政策催化、资金驱动、板块轮动等A股特点

## 输出格式
请以 JSON 格式输出：
```json
{
  "thesis": "核心看多论点（一句话总结）",
  "arguments": [
    {"source": "technical/fundamentals/fund_flow/news", "point": "具体论点", "evidence": "数据支撑"},
    ...
  ],
  "price_target": 目标价(float),
  "catalysts": ["近期催化剂1", "催化剂2"]
}
```
"""


class BullResearcher:
    """多头研究员 — quick LLM"""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def research(
        self,
        code: str,
        name: str,
        reports: list[AnalystReport],
        bear_argument: str = "",
        round_num: int = 1,
    ) -> str:
        """
        生成多头研究论点。

        reports: 四份分析师报告
        bear_argument: 上一轮空头论点 (辩论模式时使用)
        round_num: 当前辩论轮次
        """
        context = self._build_context(code, name, reports, bear_argument, round_num)

        messages = [
            Message(role="system", content=BULL_SYSTEM_PROMPT),
            Message(role="user", content=context),
        ]

        resp = self._llm.chat(messages)
        return resp.content

    def _build_context(
        self,
        code: str,
        name: str,
        reports: list[AnalystReport],
        bear_argument: str,
        round_num: int,
    ) -> str:
        report_text = _format_reports(reports)

        lines = [
            f"## 股票: {name} ({code})",
            f"## 辩论轮次: 第{round_num}轮",
            "",
            "## 四维分析报告",
            report_text,
        ]

        if bear_argument and round_num > 1:
            lines.extend([
                "",
                "## 空头最新论点 (需要逐条回应)",
                bear_argument,
                "",
                "请在承认其合理性的基础上，指出其分析中的盲点或过度悲观的假设，用数据反驳。",
            ])
        else:
            lines.append("")
            lines.append("请基于以上四份分析报告，构建你的看多论点和买入逻辑。")

        return "\n".join(lines)


def _format_reports(reports: list[AnalystReport]) -> str:
    parts = []
    type_names = {
        "technical": "技术面分析师", "fundamentals": "基本面分析师",
        "fund_flow": "资金面分析师", "news": "消息面分析师",
    }
    for r in reports:
        label = type_names.get(r.analyst_type, r.analyst_type)
        parts.append(f"### {label}")
        parts.append(f"信号: {r.signal}  置信度: {r.confidence:.0%}")
        parts.append(f"分析: {r.reasoning}")
        if r.key_points:
            parts.append("关键发现:")
            parts.extend(f"  - {p}" for p in r.key_points)
        if r.risks:
            parts.append("风险提示:")
            parts.extend(f"  - {p}" for p in r.risks)
        parts.append("")
    return "\n".join(parts)

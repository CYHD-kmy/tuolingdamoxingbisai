"""
空头研究员 — 基于多头论点进行批判性质疑。

职责:
- 逐条审视多头论点的逻辑漏洞
- 指出被忽略的风险和负面信号
- 提供反向视角，防止确认偏误
"""

from __future__ import annotations

import logging

from ..base import AnalystReport
from ...llm.client import LLMClient
from ...llm.schema import Message

logger = logging.getLogger(__name__)

BEAR_SYSTEM_PROMPT = """你是一位 A 股空头研究员，职责是挑战看多论点，揭示被忽略的风险。

## 工作方式
你会收到四份分析报告和多头研究员的完整论点。
你的任务不是无脑唱空，而是：
1. 找出多头论点中的数据选择性偏差（cherry-picking）
2. 指出被多头忽略或弱化的负面信号
3. 评估风险收益比是否合理
4. 如果确实找不到重大风险，诚实承认

## 分析原则
1. **质疑但不偏执**: 该看多时不要强行唱空，但要对每个论点保持怀疑
2. **关注尾部风险**: 小概率但大影响的事件（黑天鹅）值得关注
3. **风险收益比**: 即使方向正确，如果当前价位已经price in太多利好，上涨空间有限
4. **A股特性**: 关注资金博弈、主力出货、利好兑现即利空等特点

## 输出格式
请以 JSON 格式输出：
```json
{
  "overall_stance": "bearish/slightly_bearish/neutral",
  "critiques": [
    {"target": "针对多头哪个论点", "issue": "具体问题", "severity": "high/medium/low"},
    ...
  ],
  "overlooked_risks": ["被忽略的风险1", "风险2"],
  "risk_reward_assessment": "风险收益比评估（50字以内）"
}
```
"""


class BearResearcher:
    """空头研究员 — quick LLM"""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def research(
        self,
        code: str,
        name: str,
        reports: list[AnalystReport],
        bull_argument: str,
        round_num: int = 1,
    ) -> str:
        """
        生成空头批判意见。

        reports: 四份分析师报告
        bull_argument: 本轮多头的完整论点
        round_num: 当前辩论轮次
        """
        context = self._build_context(code, name, reports, bull_argument, round_num)

        messages = [
            Message(role="system", content=BEAR_SYSTEM_PROMPT),
            Message(role="user", content=context),
        ]

        resp = self._llm.chat(messages)
        return resp.content

    def _build_context(
        self,
        code: str,
        name: str,
        reports: list[AnalystReport],
        bull_argument: str,
        round_num: int,
    ) -> str:
        report_signals = []
        for r in reports:
            signal_icon = {"bullish": "看多", "bearish": "看空", "neutral": "中性"}.get(r.signal, r.signal)
            report_signals.append(f"- {r.analyst_type}: {signal_icon} (置信度:{r.confidence:.0%})")

        lines = [
            f"## 股票: {name} ({code})",
            f"## 辩论轮次: 第{round_num}轮",
            "",
            "## 分析师信号汇总",
            *report_signals,
            "",
            "## 多头论点 (需要逐条审视)",
            bull_argument,
            "",
        ]

        if round_num > 1:
            lines.append(f"这是第{round_num}轮辩论。请在前几轮基础上进一步深挖，不要重复之前的论点。")
        else:
            lines.append("请逐条审视多头论点的逻辑漏洞，指出被忽略的风险信号。")

        return "\n".join(lines)

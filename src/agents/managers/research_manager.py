"""
研究主管 — deep LLM，综合辩论双方论点给出最终研报。

输入: 四份分析师报告 + 完整辩论记录
输出: ResearchVerdict (方向 + 置信度 + 目标价 + 风险等级)
"""

from __future__ import annotations

import json
import logging

from ..base import AnalystReport
from ..models import DebateResult, ResearchVerdict
from ..researchers.engine import debate_result_to_text
from ...llm.client import LLMClient
from ...llm.schema import Message
from ...utils.validators import extract_json

logger = logging.getLogger(__name__)

RESEARCH_MANAGER_PROMPT = """你是一位资深 A 股研究主管，拥有 20 年投研经验。

## 你的职责
审阅四份分析师报告和多空双方辩论记录，给出最终研究结论。

## 决策框架
1. **评估论据质量**: 哪一方的论据更扎实？哪一方在回避关键问题？
2. **信号权重**: 不同市况下各维度权重不同：
   - 震荡市: 技术面 > 资金面 > 消息面 > 基本面
   - 趋势市: 趋势 > 动量 > 量价 > 其他
   - 财报季: 基本面 > 消息面 > 技术面 > 资金面
3. **置信度校准**:
   - 四方一致同向 → 高置信度 (0.75-0.90)
   - 三方同向一方分歧 → 中置信度 (0.60-0.75)
   - 两方对两方 → 低置信度 (0.40-0.60)
   - 不要给出 0.95+ 的极端置信度
4. **风险定价**: 即使看多，也要明确止损条件和风险等级

## 输出格式
请以 JSON 格式输出：
```json
{
  "direction": "buy",
  "confidence": 0.72,
  "target_price": 1850.00,
  "risk_level": "medium",
  "core_reasoning": "综合四维分析和辩论，核心逻辑是...（200字以内）",
  "key_risks": ["风险1", "风险2"],
  "verdict_summary": "一句话总结"
}
```

direction 取值: "buy" / "sell" / "hold"
risk_level 取值: "low" / "medium" / "high"
confidence 取值: 0.0 ~ 1.0
"""


class ResearchManager:
    """研究主管 — deep LLM"""

    def __init__(self, deep_llm: LLMClient) -> None:
        self._llm = deep_llm

    def decide(
        self,
        code: str,
        name: str,
        reports: list[AnalystReport],
        debate: DebateResult,
        current_price: float,
    ) -> ResearchVerdict:
        """
        综合分析和辩论，给出最终研究结论。
        LLM 不可用时自动降级为简单多数投票。
        """
        try:
            return self._decide_with_llm(code, name, reports, debate, current_price)
        except Exception as e:
            logger.warning("ResearchManager: LLM 调用失败 (%s)，降级为投票", e)
            from ..fallback import fallback_verdict
            return fallback_verdict(code, name, reports, current_price)

    def _decide_with_llm(
        self,
        code: str,
        name: str,
        reports: list[AnalystReport],
        debate: DebateResult,
        current_price: float,
    ) -> ResearchVerdict:
        report_text = self._format_reports(reports)
        debate_text = debate_result_to_text(debate)

        user_prompt = f"""## 股票: {name} ({code})
当前价格: {current_price:.2f}

## 四维分析报告
{report_text}

## 多空辩论记录
{debate_text}
请基于以上信息，给出最终研究结论。"""

        messages = [
            Message(role="system", content=RESEARCH_MANAGER_PROMPT),
            Message(role="user", content=user_prompt),
        ]

        resp = self._llm.chat(messages)
        return self._parse_verdict(resp.content, code, name)

    @staticmethod
    def _format_reports(reports: list[AnalystReport]) -> str:
        parts = []
        type_names = {
            "technical": "技术面", "fundamentals": "基本面",
            "fund_flow": "资金面", "news": "消息面",
        }
        for r in reports:
            label = type_names.get(r.analyst_type, r.analyst_type)
            parts.append(
                f"【{label}】信号:{r.signal} 置信度:{r.confidence:.0%}\n"
                f"分析:{r.reasoning}"
            )
            if r.key_points:
                parts.append(f"关键点: {'; '.join(r.key_points[:3])}")
            if r.risks:
                parts.append(f"风险: {'; '.join(r.risks[:3])}")
            parts.append("")
        return "\n".join(parts)

    @staticmethod
    def _parse_verdict(raw: str, code: str, name: str) -> ResearchVerdict:
        """解析研究主管的 JSON 输出"""
        try:
            data = json.loads(extract_json(raw))
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
            logger.warning("ResearchManager: JSON 解析失败，使用保守估计")
            return ResearchVerdict(
                code=code, name=name, direction="hold",
                confidence=0.3, core_reasoning=raw[:200],
            )

        return ResearchVerdict(
            code=code,
            name=name,
            direction=data.get("direction", "hold"),
            confidence=float(data.get("confidence", 0.5)),
            target_price=float(data.get("target_price", 0)),
            risk_level=data.get("risk_level", "medium"),
            core_reasoning=data.get("core_reasoning", data.get("verdict_summary", ""))[:300],
            key_risks=data.get("key_risks", []),
        )

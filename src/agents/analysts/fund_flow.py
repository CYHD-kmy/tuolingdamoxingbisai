"""
资金面分析师 — 从主力资金动向和筹码分布角度分析。

关注:
- 主力资金: 主力/超大单/大单净流入金额和占比
- 资金趋势: 连续流入/流出天数
- 筹码结构: 融资融券变化、大宗交易
"""

from __future__ import annotations

from typing import Any

from ..base import BaseAnalyst
from ..tools import tools_for
from ...llm.schema import Tool


FUND_FLOW_PROMPT = """你是一位 A 股资金面分析师，擅长从主力资金动向来判断短期走势。

## 分析框架

1. **主力资金方向**
   - 主力净流入金额：正的越大越好，负的需要警惕
   - 主力净流入占比：> 5% 说明主力参与度高
   - 超大单 vs 大单 vs 中单 vs 小单：超大单和大单代表机构和游资，中单和小单代表散户

2. **资金持续性**
   - 近3-5日主力资金是持续流入、持续流出、还是忽进忽出？
   - 持续流入代表主力看好，持续流出需要警惕
   - 忽进忽出说明主力在博弈短线，波动可能较大

3. **量价关系**
   - 主力流入 + 股价上涨 = 主力建仓 (强势信号)
   - 主力流入 + 股价下跌 = 主力护盘或对倒 (弱势信号)
   - 主力流出 + 股价上涨 = 主力出货 (危险信号)
   - 主力流出 + 股价下跌 = 主力撤退 (看空信号)

4. **散户行为**
   - 小单持续净流入但股价下跌 = 散户接盘，不利
   - 小单净流出但股价上涨 = 筹码集中，有利

## 输出格式
请务必以 JSON 格式输出你的分析结论:
```json
{
  "analyst_type": "fund_flow",
  "signal": "bullish",
  "confidence": 0.70,
  "reasoning": "核心分析逻辑 (200字以内)",
  "key_points": ["发现点1", "发现点2", "发现点3"],
  "risks": ["资金面风险1"]
}
```

signal 取值: "bullish" / "bearish" / "neutral"
"""


class FundFlowAnalyst(BaseAnalyst):
    """资金面分析师"""

    analyst_type = "fund_flow"

    @property
    def system_prompt(self) -> str:
        return FUND_FLOW_PROMPT

    @property
    def tools(self) -> list[Tool]:
        return tools_for("fund_flow")

    def build_context(self, code: str) -> str:
        """预加载资金流向和实时行情"""
        flows = self._data.get_fund_flow(code, days=5)
        quote = self._data.get_realtime_quote(code)

        lines = []

        if quote:
            lines.extend([
                "## 当前行情",
                f"最新价: {quote.price:.2f}  涨跌幅: {quote.pct_chg:+.2f}%",
                f"成交量: {quote.volume:.0f}  成交额: {quote.amount:.0f}",
                f"换手率: {quote.turnover:.2f}%  量比: {quote.volume_ratio:.2f}",
            ])

        if flows:
            lines.append(f"\n## 近5日资金流向")
            for f in flows:
                direction = "流入" if f.main_net_inflow > 0 else "流出"
                lines.append(
                    f"- {f.date}: 主力{direction} {abs(f.main_net_inflow):.0f}万 "
                    f"(占比{f.main_pct:+.1f}%) | "
                    f"超大单:{f.super_large_net:.0f}万 "
                    f"大单:{f.large_net:.0f}万 | "
                    f"中单:{f.medium_net:.0f}万 "
                    f"小单:{f.small_net:.0f}万"
                )

            # 汇总统计
            total_main = sum(f.main_net_inflow for f in flows)
            consecutive_in = sum(1 for f in flows if f.main_net_inflow > 0)
            consecutive_out = sum(1 for f in flows if f.main_net_inflow < 0)
            lines.extend([
                "",
                f"## 汇总: 近5日主力合计 {'流入' if total_main > 0 else '流出'} {abs(total_main):.0f}万",
                f"连续流入天数: {consecutive_in}  连续流出天数: {consecutive_out}",
            ])
        else:
            lines.append("(未获取到资金流向数据，请使用 get_fund_flow 工具获取)")

        return "\n".join(lines)

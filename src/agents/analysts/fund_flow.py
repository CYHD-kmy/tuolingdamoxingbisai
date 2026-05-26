"""
资金面分析师 — 从主力资金动向和筹码分布角度分析。

关注:
- 主力资金: 主力/超大单/大单净流入金额和占比
- 资金趋势: 连续流入/流出天数
- 筹码结构: 融资融券变化、大宗交易
"""

from __future__ import annotations

from ..base import BaseAnalyst
from ..tools import tools_for
from ...llm.schema import Tool


FUND_FLOW_PROMPT = """你是一位 A 股资金面分析师，擅长从主力资金动向、北向资金和融资融券来判断短期走势。

## 分析框架

1. **主力资金方向**
   - 主力净流入金额：正的越大越好，负的需要警惕
   - 主力净流入占比：> 5% 说明主力参与度高
   - 超大单 vs 大单 vs 中单 vs 小单：超大单和大单代表机构和游资，中单和小单代表散户

2. **北向资金 (沪深股通)**
   - 北向资金全市场是净流入还是净流出？这代表外资整体态度
   - 该股是否被北向持续增持/减持？
   - 北向持股占比变化：增持=外资看好，减持需关注

3. **融资融券**
   - 融资余额变化：持续上升=杠杆资金看多，下降=去杠杆
   - 融资买入额占成交额比重：>10% 表示杠杆活跃
   - 融券余量大幅增加=看空信号

4. **资金持续性**
   - 近3-5日主力资金是持续流入、持续流出、还是忽进忽出？
   - 持续流入代表主力看好，持续流出需要警惕

5. **量价关系**
   - 主力流入 + 股价上涨 = 主力建仓 (强势信号)
   - 主力流入 + 股价下跌 = 主力护盘或对倒 (弱势信号)
   - 主力流出 + 股价上涨 = 主力出货 (危险信号)
   - 主力流出 + 股价下跌 = 主力撤退 (看空信号)

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
        """预加载资金流向、北向资金、融资融券和实时行情"""
        flows = self._data.get_fund_flow(code, days=5)
        quote = self._data.get_realtime_quote(code)
        northbound_flow = self._data.get_northbound_flow(days=5)
        northbound_stock = self._data.get_northbound_stock(code, days=10)
        margin_detail = self._data.get_margin_detail(code, days=10)

        lines = []

        if quote:
            lines.extend([
                "## 当前行情",
                f"最新价: {quote.price:.2f}  涨跌幅: {quote.pct_chg:+.2f}%",
                f"成交量: {quote.volume:.0f}  成交额: {quote.amount:.0f}",
                f"换手率: {quote.turnover:.2f}%  量比: {quote.volume_ratio:.2f}",
            ])

        # ── 主力资金流向 ──
        if flows:
            lines.append(f"\n## 近5日主力资金流向")
            for f in flows:
                direction = "流入" if f.main_net_inflow > 0 else "流出"
                lines.append(
                    f"- {f.date}: 主力{direction} {abs(f.main_net_inflow):.0f}万 "
                    f"(占比{f.main_pct:+.1f}%) | "
                    f"超大单:{f.super_large_net:.0f}万 "
                    f"大单:{f.large_net:.0f}万"
                )
            total_main = sum(f.main_net_inflow for f in flows)
            positive_days = sum(1 for f in flows if f.main_net_inflow > 0)
            lines.append(f"汇总: 近5日主力合计 {'流入' if total_main > 0 else '流出'} {abs(total_main):.0f}万, 净流入{positive_days}天")
        else:
            lines.append("\n(主力资金数据暂未获取)")

        # ── 北向资金 ──
        if northbound_flow:
            lines.append(f"\n## 近5日北向资金 (全市场)")
            for nf in northbound_flow:
                direction = "净流入" if nf.net_inflow > 0 else "净流出"
                lines.append(f"- {nf.date}: {direction} {abs(nf.net_inflow):.0f}万 (沪:{nf.sh_inflow:.0f} 深:{nf.sz_inflow:.0f})")
            total_nb = sum(n.net_inflow for n in northbound_flow)
            lines.append(f"汇总: 近5日北向合计 {'净流入' if total_nb > 0 else '净流出'} {abs(total_nb):.0f}万")

        if northbound_stock:
            lines.append(f"\n## 个股北向持仓变化")
            for d in northbound_stock[-5:]:
                lines.append(f"- {d.get('date', '')}: 持股{d.get('hold_shares',0):.0f}万股, 占比{d.get('hold_pct',0):.2f}%")

        # ── 融资融券 ──
        if margin_detail:
            lines.append(f"\n## 近5日融资融券")
            for m in margin_detail[-5:]:
                lines.append(f"- {m.date}: 融资余额{m.margin_balance:.0f}万, 买入{m.margin_buy:.0f}万, 融券{m.short_balance:.0f}万")
            if len(margin_detail) >= 2:
                mb_change = margin_detail[-1].margin_balance - margin_detail[0].margin_balance
                trend = "上升" if mb_change > 0 else "下降"
                lines.append(f"融资余额趋势: {trend} {abs(mb_change):.0f}万")

        return "\n".join(lines)

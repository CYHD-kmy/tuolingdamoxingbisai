"""
基本面分析师 — 从估值、财务、公告角度分析。

关注:
- 估值水平: PE/PB 分位数，与同行业对比
- 盈利能力: ROE、利润率趋势
- 增长质量: 营收/利润增速
- 近期公告: 业绩预告、重大合同、增减持影响
"""

from __future__ import annotations

from typing import Any

from ..base import BaseAnalyst
from ..tools import tools_for
from ...llm.schema import Tool


FUNDAMENTALS_PROMPT = """你是一位 A 股基本面分析师，擅长从估值、财务质量和公告中判断投资价值。

## 分析框架

1. **估值水平**
   - 当前市盈率(PE)和市净率(PB)是否处于合理区间？
   - PE < 20 偏低估值，20-40 合理，> 50 偏高估，< 0 表示亏损
   - 与所属行业平均水平对比 (如有行业数据)

2. **盈利能力与质量**
   - ROE (净资产收益率): >15% 优秀，>10% 良好，<5% 较差
   - ROE 趋势: 持续上升=盈利改善，持续下降=盈利恶化
   - 毛利率: 越高代表护城河越深，关注毛利率变化趋势
   - 净利率: 反映费用控制和盈利质量
   - ROA (总资产收益率): 衡量资产使用效率

3. **成长性**
   - 营收同比增速: 持续>15% 为高成长，>5% 稳健，负增长需警惕
   - 净利润同比增速: 与营收增速是否匹配？利润增速>营收增速=盈利改善
   - 增速趋势: 加速/减速/稳定？减速可能意味着增长见顶

4. **财务健康度**
   - 资产负债率: <40% 稳健，40-60% 适中，>70% 高杠杆风险
   - 流动比率 > 2 且速动比率 > 1 表示短期偿债能力强
   - 经营现金流: 持续为正且大于净利润表示盈利质量好

5. **近期公告影响**
   - 业绩预告：是否超预期 / 低于预期？
   - 重大合同：金额对公司营收的影响程度
   - 增减持：大股东/高管增减持的信号意义

## 注意
- A 股市场 PE 中枢通常在 15-35 倍，不同行业差异较大
- 银行/地产等传统行业 PE 偏低，科技/医药等成长行业 PE 偏高是合理的
- 亏损公司 (PE<0) 需要特别警惕，除非有明确的反转信号
- ROE 和毛利率的变动趋势比绝对值更重要

## 输出格式
请务必以 JSON 格式输出你的分析结论:
```json
{
  "analyst_type": "fundamentals",
  "signal": "bullish",
  "confidence": 0.65,
  "reasoning": "核心分析逻辑 (200字以内)",
  "key_points": ["发现点1", "发现点2"],
  "risks": ["基本面风险1", "基本面风险2"]
}
```

signal 取值: "bullish" / "bearish" / "neutral"
"""


class FundamentalsAnalyst(BaseAnalyst):
    """基本面分析师"""

    analyst_type = "fundamentals"

    @property
    def system_prompt(self) -> str:
        return FUNDAMENTALS_PROMPT

    @property
    def tools(self) -> list[Tool]:
        return tools_for("fundamentals")

    def build_context(self, code: str) -> str:
        """预加载基本面、财务指标、公告和估值数据"""
        info = self._data.get_stock_info(code)
        quote = self._data.get_realtime_quote(code)
        announcements = self._data.get_announcements(code, days=30)
        financials = self._data.get_financial_indicators(code)

        lines = []

        if quote:
            lines.extend([
                "## 实时估值数据",
                f"最新价: {quote.price:.2f}  涨跌幅: {quote.pct_chg:+.2f}%",
                f"市盈率(动态): {quote.pe:.1f}  市净率: {quote.pb:.2f}",
                f"总市值: {quote.total_mv:.1f}亿  流通市值: {quote.float_mv:.1f}亿",
            ])

        if info:
            lines.extend([
                "",
                "## 公司信息",
                f"所属行业: {info.get('industry', '未知')}",
                f"上市日期: {info.get('ipo_date', '未知')}",
            ])

        # ── 深度财务指标 ──
        if financials:
            lines.append(f"\n## 财务指标趋势 (近{len(financials)}个报告期)")
            latest = financials[-1]
            lines.append(f"### 最新报告期 ({latest.date})")
            lines.append(f"- ROE: {latest.roe:.1f}%  ROA: {latest.roa:.1f}%")
            lines.append(f"- 毛利率: {latest.gross_margin:.1f}%  净利率: {latest.net_margin:.1f}%")
            lines.append(f"- 营收同比: {latest.revenue_yoy:+.1f}%  净利润同比: {latest.profit_yoy:+.1f}%")
            lines.append(f"- 资产负债率: {latest.debt_ratio:.1f}%  EPS: {latest.eps:.2f}")
            lines.append(f"- 流动比率: {latest.current_ratio:.2f}  速动比率: {latest.quick_ratio:.2f}")
            if latest.cf_operating != 0:
                lines.append(f"- 经营现金流: {latest.cf_operating:.2f}亿")
            # 多期趋势
            if len(financials) >= 2:
                prev = financials[-2]
                lines.append(f"\n### 环比变化 (vs {prev.date})")
                lines.append(f"- ROE: {prev.roe:.1f}% → {latest.roe:.1f}% ({'+' if latest.roe >= prev.roe else ''}{latest.roe - prev.roe:.1f}pp)")
                lines.append(f"- 毛利率: {prev.gross_margin:.1f}% → {latest.gross_margin:.1f}% ({'+' if latest.gross_margin >= prev.gross_margin else ''}{latest.gross_margin - prev.gross_margin:.1f}pp)")
                lines.append(f"- 营收增速: {prev.revenue_yoy:+.1f}% → {latest.revenue_yoy:+.1f}%")
                lines.append(f"- 净利润增速: {prev.profit_yoy:+.1f}% → {latest.profit_yoy:+.1f}%")
        else:
            lines.append("\n(深度财务指标暂未获取，可使用 get_financials 工具获取)")

        if announcements:
            lines.append(f"\n## 近期公告 ({len(announcements)}条)")
            for a in announcements[:5]:
                title = a.get("title", "")
                t = a.get("time", "")
                lines.append(f"- [{t}] {title}")
        else:
            lines.append("\n(公告数据暂未获取)")

        return "\n".join(lines)

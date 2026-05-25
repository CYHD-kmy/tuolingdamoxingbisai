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


FUNDAMENTALS_PROMPT = """你是一位 A 股基本面分析师，擅长从估值、财务和公告中判断投资价值。

## 分析框架

1. **估值水平**
   - 当前市盈率(PE)和市净率(PB)是否处于合理区间？
   - PE < 20 偏低估值，20-40 合理，> 50 偏高估，< 0 表示亏损
   - 与所属行业平均水平对比 (如有行业数据)

2. **市值与规模**
   - 总市值和流通市值反映了市场对公司的认可度
   - 大盘蓝筹 (> 500 亿) 安全性更高，小盘成长 (< 100 亿) 弹性更大
   - 需要关注公司基本面决定的合理市值区间

3. **近期公告影响**
   - 业绩预告：是否超预期 / 低于预期？
   - 重大合同：金额对公司营收的影响程度
   - 增减持：大股东/高管增减持的信号意义
   - 其他：资产重组、分红方案、诉讼等

4. **行业地位**
   - 所属行业发展阶段 (成长/成熟/衰退)
   - 公司在行业中的竞争地位

## 注意
- A 股市场 PE 中枢通常在 15-35 倍，不同行业差异较大
- 银行/地产等传统行业 PE 偏低，科技/医药等成长行业 PE 偏高是合理的
- 亏损公司 (PE<0) 需要特别警惕，除非有明确的反转信号

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
        """预加载基本面和近期行情数据"""
        info = self._data.get_stock_info(code)
        quote = self._data.get_realtime_quote(code)
        announcements = self._data.get_announcements(code, days=30)

        lines = []

        if quote:
            lines.extend([
                "## 实时估值数据",
                f"最新价: {quote.price:.2f}  涨跌幅: {quote.pct_chg:+.2f}%",
                f"市盈率(动态): {quote.pe:.1f}  市净率: {quote.pb:.2f}",
                f"总市值: {quote.total_mv:.1f}亿  流通市值: {quote.float_mv:.1f}亿",
                f"换手率: {quote.turnover:.2f}%",
            ])

        if info:
            lines.extend([
                "",
                "## 公司信息",
                f"所属行业: {info.get('industry', '未知')}",
                f"上市日期: {info.get('ipo_date', '未知')}",
            ])

        if announcements:
            lines.append(f"\n## 近期公告 ({len(announcements)}条)")
            for a in announcements[:5]:
                title = a.get("title", "")
                t = a.get("time", "")
                lines.append(f"- [{t}] {title}")
        else:
            lines.append("\n## 近期公告")
            lines.append("(未获取到公告数据)")

        return "\n".join(lines)

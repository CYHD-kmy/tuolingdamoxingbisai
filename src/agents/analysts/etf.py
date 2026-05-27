"""
ETF 分析师 — 简化版四维分析，聚焦技术面+资金面。

分析维度:
- 趋势: 跟踪指数走势、均线排列
- 流动性: 成交额、折溢价
- 资金: 资金净流入方向
- 关联: 与指数的跟踪误差
"""
from __future__ import annotations

from ..base import BaseAnalyst
from ..tools import tools_for
from ...llm.schema import Tool


ETF_PROMPT = """你是一位 A 股 ETF 分析师，擅长评估 ETF 的交易机会和风险。

## 分析框架
请按照以下维度进行分析:

1. **趋势判断**
   - 跟踪指数/板块的近期走势
   - 均线排列 (多头/空头/交织)
   - 近5日涨跌幅和量价配合

2. **流动性评估**
   - 日均成交额 (确保买卖滑点可接受)
   - 折溢价水平 (大幅折价是买入信号，大幅溢价追高风险)
   - 基金规模 (规模越大流动性越好)

3. **资金面**
   - 近期资金净流入/流出方向
   - 是否有大额申赎异动

4. **市场环境**
   - 对应板块/指数的催化剂
   - 政策或事件风险

## 输出格式
请以 JSON 格式输出:
```json
{
  "analyst_type": "etf",
  "signal": "bullish",
  "confidence": 0.70,
  "reasoning": "核心分析逻辑 (200字以内)",
  "key_points": ["发现1", "发现2"],
  "risks": ["风险1"]
}
```

signal: bullish / bearish / neutral
"""


class ETFAnalyst(BaseAnalyst):
    """ETF 分析师 — 技术面+资金面为主"""

    analyst_type = "etf"

    @property
    def system_prompt(self) -> str:
        return ETF_PROMPT

    @property
    def tools(self) -> list[Tool]:
        return [t for t in tools_for("technical") if t.name in ("get_daily_data", "get_realtime_quote")]

    def build_context(self, code: str) -> str:
        """构建 ETF 分析上下文"""
        daily = self._data.get_etf_daily(code, days=30)
        quote = self._data.get_realtime_quote(code)

        lines = [f"## ETF: {code}"]

        if quote:
            lines.extend([
                f"## 实时行情",
                f"名称: {quote.name}  最新价: {quote.price:.3f}  涨跌幅: {quote.pct_chg:+.2f}%",
                f"成交额: {quote.amount:.0f}  换手率: {quote.turnover:.2f}%",
                f"量比: {quote.volume_ratio:.2f}",
                "",
            ])

        if not daily:
            lines.append("未获取到日线数据")
            return "\n".join(lines)

        latest = daily[-1]
        lines.extend([
            f"## 最新日线 ({latest.date})",
            f"开:{latest.open:.3f} 高:{latest.high:.3f} 低:{latest.low:.3f} 收:{latest.close:.3f}",
            f"涨跌幅:{latest.pct_chg:+.2f}%  成交额:{latest.amount:.0f}",
        ])

        if latest.ma5 > 0:
            lines.append(f"MA5:{latest.ma5:.3f}  MA10:{latest.ma10:.3f}  MA20:{latest.ma20:.3f}")

        lines.extend(["", "## 近5日走势"])
        for d in daily[-5:]:
            direction = "[+]" if d.pct_chg > 0 else ("[-]" if d.pct_chg < 0 else "[=]")
            lines.append(
                f"  {d.date} {direction} {d.pct_chg:+.2f}%  "
                f"C:{d.close:.3f}  量:{d.volume:.0f}"
            )

        return "\n".join(lines)

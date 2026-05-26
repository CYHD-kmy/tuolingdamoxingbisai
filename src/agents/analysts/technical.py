"""
技术面分析师 — 从走势、量价、指标角度分析。

关注:
- 趋势: 均线形态 (多头/空头排列、金叉/死叉)
- 动量: MACD 位置与方向、RSI 超买超卖
- 量价: 成交量变化、放量/缩量信号
- 形态: 支撑/压力位、突破/破位
"""

from __future__ import annotations

from typing import Any

from ..base import BaseAnalyst
from ..tools import tools_for
from ...llm.schema import Tool


TECHNICAL_PROMPT = """你是一位资深 A 股技术分析师，擅长从走势图形和指标中提炼交易信号。

## 分析框架
请按照以下维度进行分析，并给出明确的看多/看空/中性判断：

1. **趋势判断** (最重要)
   - 均线排列: MA5/MA10/MA20 是多头还是空头排列？
   - 价格与均线关系: 股价在所有均线之上是强势，被均线压制是弱势
   - 近期走势方向: 连阳还是连阴？斜率如何？

2. **动量信号**
   - MACD: DIF/DEA 在零轴上方还是下方？金叉/死叉状态？柱状线变化？
   - RSI: RSI(6) 和 RSI(14) 是否超买(>80)/超卖(<30)？

3. **量价配合**
   - 成交量的近5日变化趋势 (递增/递减/平稳)
   - 量比是否 > 1.5? 放量上涨是强势，放量下跌是危险信号
   - 换手率是否异常？

4. **关键位置**
   - 近期高点在什么价位？是否正在挑战压力位？
   - 近期低点在什么价位？是否有支撑？
   - 当前价格距离压力/支撑位的空间

## 输出格式
请务必以 JSON 格式输出你的分析结论:
```json
{
  "analyst_type": "technical",
  "signal": "bullish",
  "confidence": 0.75,
  "reasoning": "核心分析逻辑 (200字以内)",
  "key_points": ["发现点1", "发现点2", "发现点3"],
  "risks": ["技术风险1", "技术风险2"]
}
```

signal 取值: "bullish" (看多) / "bearish" (看空) / "neutral" (中性)
confidence 取值: 0.0 ~ 1.0 (置信度)
"""


class TechnicalAnalyst(BaseAnalyst):
    """技术面分析师"""

    analyst_type = "technical"

    @property
    def system_prompt(self) -> str:
        return TECHNICAL_PROMPT

    @property
    def tools(self) -> list[Tool]:
        return tools_for("technical")

    def build_context(self, code: str) -> str:
        """预加载日线和技术指标数据"""
        daily = self._data.get_daily_data(code, days=30)
        if not daily:
            return "未获取到日线数据，请使用 get_daily_data 工具尝试获取。"

        latest = daily[-1]
        lines = [
            f"## 最新行情 ({latest.date})",
            f"开盘:{latest.open:.2f} 最高:{latest.high:.2f} 最低:{latest.low:.2f} 收盘:{latest.close:.2f}",
            f"涨跌幅:{latest.pct_chg:+.2f}%  成交量:{latest.volume:.0f}  成交额:{latest.amount:.0f}  换手率:{latest.turnover:.2f}%",
            "",
            "## 技术指标",
            f"MA5:{latest.ma5:.2f}  MA10:{latest.ma10:.2f}  MA20:{latest.ma20:.2f}",
            f"MACD: DIF={latest.macd_dif:.4f}  DEA={latest.macd_dea:.4f}  BAR={latest.macd_bar:.4f}",
            f"RSI(6)={latest.rsi_6:.1f}  RSI(14)={latest.rsi_14:.1f}",
            "",
            "## 近5日走势",
        ]

        for d in daily[-5:]:
            dir_mark = "[+]" if d.pct_chg > 0 else ("[-]" if d.pct_chg < 0 else "[=]")
            lines.append(
                f"  {d.date}  {dir_mark} {d.pct_chg:+.2f}%  "
                f"O:{d.open:.2f} H:{d.high:.2f} L:{d.low:.2f} C:{d.close:.2f}  "
                f"量:{d.volume:.0f}"
            )

        return "\n".join(lines)

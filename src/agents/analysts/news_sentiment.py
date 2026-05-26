"""
消息面分析师 — 从新闻、公告、市场情绪角度分析。

关注:
- 财经新闻: 正面/负面/中性情感
- 公告解读: 业绩预告、重大合同、增减持
- 概念热度: 所属概念/行业是否处于热点
- 市场情绪: 整体风险偏好
"""

from __future__ import annotations

from typing import Any

from ..base import BaseAnalyst
from ..tools import tools_for
from ...llm.schema import Tool


NEWS_PROMPT = """你是一位 A 股消息面分析师，擅长从财经新闻和公告中捕捉市场情绪。

## 分析框架

1. **新闻情感分析**
   - 近期与该股相关的新闻是正面、负面还是中性？
   - 正面新闻: 业绩增长、订单增加、政策利好、研报推荐
   - 负面新闻: 业绩下滑、诉讼纠纷、监管问询、减持公告
   - 新闻数量多寡也反映市场关注度

2. **公告解读**
   - 业绩预告: 大幅预增/预减对股价有显著影响
   - 重大合同: 合同金额占营收比重大
   - 增减持: 大股东增持是强烈信号，减持需关注原因
   - 资产重组/并购: 注意重组失败风险

3. **热点概念匹配**
   - 该股是否涉及当前市场热点概念 (AI、新能源、半导体 等)？
   - 概念热度可持续性: 短期炒作 vs 长期趋势
   - 公司在概念中的核心地位: 龙头 vs 跟风

4. **市场情绪综合**
   - 近期市场整体风险偏好如何？
   - 同行业/概念板块整体表现

## 输出格式
请务必以 JSON 格式输出你的分析结论:
```json
{
  "analyst_type": "news",
  "signal": "bullish",
  "confidence": 0.60,
  "reasoning": "核心分析逻辑 (200字以内)",
  "key_points": ["发现点1", "发现点2"],
  "risks": ["消息面风险1", "消息面风险2"]
}
```

signal 取值: "bullish" / "bearish" / "neutral"
"""


class NewsSentimentAnalyst(BaseAnalyst):
    """消息面分析师"""

    analyst_type = "news"

    @property
    def system_prompt(self) -> str:
        return NEWS_PROMPT

    @property
    def tools(self) -> list[Tool]:
        return tools_for("news")

    def build_context(self, code: str) -> str:
        """预加载新闻、公告和行情"""
        name = self._data.get_stock_name(code)
        info = self._data.get_stock_info(code)
        announcements = self._data.get_announcements(code, days=14)
        news = self._data.get_news(code, days=3)
        quote = self._data.get_realtime_quote(code)

        lines = []

        if quote:
            lines.extend([
                "## 当前行情",
                f"最新价: {quote.price:.2f}  涨跌幅: {quote.pct_chg:+.2f}%  换手率: {quote.turnover:.2f}%",
            ])

        if info:
            lines.extend([
                "",
                f"所属行业: {info.get('industry', '未知')}",
            ])

        if announcements:
            lines.append(f"\n## 近期公告 ({len(announcements)}条)")
            for a in announcements[:8]:
                title = a.get("title", "")
                t = a.get("time", "")
                lines.append(f"- [{t}] {title}")
        else:
            lines.append("\n(公告数据暂未获取，可使用 get_announcements 工具获取)")

        if news:
            lines.append(f"\n## 相关新闻 ({len(news)}条)")
            for n in news[:10]:
                title = n.get("title", "")
                t = n.get("time", "")
                src = n.get("source", "")
                lines.append(f"- [{t}] {title} (来源: {src})")
        else:
            lines.append("\n(新闻数据暂未获取，可使用 get_news 工具搜索)")

        return "\n".join(lines)

"""
消息面分析师 — 从新闻、公告、市场情绪角度分析。

关注:
- 财经新闻: 正面/负面/中性情感
- 公告解读: 业绩预告、重大合同、增减持
- 概念热度: 所属概念/行业是否处于热点
- 市场情绪: 整体风险偏好
"""

from __future__ import annotations

from ..base import BaseAnalyst
from ..tools import tools_for
from ...llm.schema import Tool


NEWS_PROMPT = """你是一位 A 股消息面分析师，擅长从财经新闻、研报、机构调研和公告中捕捉市场情绪。

## 分析框架

1. **新闻与快讯分析**
   - 近期与该股相关的新闻是正面、负面还是中性？
   - 正面新闻: 业绩增长、订单增加、政策利好、研报推荐
   - 负面新闻: 业绩下滑、诉讼纠纷、监管问询、减持公告
   - 财联社电报是否有与该股/行业相关的突发消息？
   - 新闻数量多寡也反映市场关注度

2. **分析师研报**
   - 近期有多少机构发布了研报？是买入/增持/中性/减持？
   - 多家机构同时看好=市场共识强
   - 研报密集发布期通常意味着公司有重大变化

3. **机构调研**
   - 近期是否有机构密集调研？
   - 调研机构数量和质量 (知名基金/券商调研更有分量)
   - 调研后股价表现如何？

4. **公告解读**
   - 业绩预告: 大幅预增/预减对股价有显著影响
   - 重大合同: 合同金额占营收比重大
   - 增减持: 大股东增持是强烈信号，减持需关注原因

5. **热点概念匹配**
   - 该股是否涉及当前市场热点概念 (AI、新能源、半导体 等)？
   - 概念热度可持续性: 短期炒作 vs 长期趋势
   - 公司在概念中的核心地位: 龙头 vs 跟风

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
        """预加载新闻、公告、研报、机构调研和行情"""
        name = self._data.get_stock_name(code)
        info = self._data.get_stock_info(code)
        announcements = self._data.get_announcements(code, days=14)
        news = self._data.get_news(code, days=3)
        quote = self._data.get_realtime_quote(code)
        research_reports = self._data.get_research_reports(code, days=30)
        telegraph = self._data.get_telegraph(limit=20)
        visits = self._data.get_institutional_visits(days=30)
        market_activity = self._data.get_market_activity()

        lines = []

        if quote:
            lines.extend([
                "## 当前行情",
                f"最新价: {quote.price:.2f}  涨跌幅: {quote.pct_chg:+.2f}%  换手率: {quote.turnover:.2f}%",
            ])

        if info:
            lines.append(f"所属行业: {info.get('industry', '未知')}")

        # ── 财联社电报 ──
        if telegraph:
            lines.append(f"\n## 今日快讯 (财联社电报, 近{len(telegraph)}条)")
            for t in telegraph[:8]:
                title = t.get("title", "")
                content = t.get("content", "")[:100]
                lines.append(f"- {title}: {content}")
        else:
            lines.append("\n(电报数据暂未获取)")

        # ── 个股新闻 ──
        if news:
            lines.append(f"\n## 相关新闻 ({len(news)}条)")
            for n in news[:8]:
                title = n.get("title", "")
                t = n.get("time", "")
                src = n.get("source", "")
                lines.append(f"- [{t}] {title} (来源: {src})")
        else:
            lines.append("\n(新闻数据暂未获取，可使用 get_news 工具搜索)")

        # ── 分析师研报 ──
        if research_reports:
            lines.append(f"\n## 近期分析师研报 ({len(research_reports)}篇)")
            for r in research_reports[:5]:
                lines.append(f"- [{r.get('date', '')}] {r.get('org', '')}: {r.get('rating', '')} — {r.get('title', '')}")
        else:
            lines.append("\n(研报数据暂未获取，可使用 get_research_reports 工具获取)")

        # ── 机构调研 ──
        if visits:
            lines.append(f"\n## 近期机构调研 ({len(visits)}次)")
            for v in visits[:5]:
                lines.append(f"- [{v.date}] {v.institution} (参与{v.visitors}人): {v.summary[:120]}")
        else:
            lines.append("\n(机构调研数据暂未获取)")

        # ── 市场异动 ──
        if market_activity:
            relevant = [m for m in market_activity if m.code == code][:5]
            if relevant:
                lines.append(f"\n## 市场异动 ({len(relevant)}条)")
                for m in relevant:
                    lines.append(f"- [{m.time}] {m.activity_type}: {m.description[:120]}")

        # ── 公告 ──
        if announcements:
            lines.append(f"\n## 近期公告 ({len(announcements)}条)")
            for a in announcements[:5]:
                title = a.get("title", "")
                t = a.get("time", "")
                lines.append(f"- [{t}] {title}")
        else:
            lines.append("\n(公告数据暂未获取)")

        return "\n".join(lines)

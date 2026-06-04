"""
板块猎手分析师 — 板块轮动 + 热点追踪 + 游资跟投。

A 股具有极强的板块联动效应:
- 一个板块启动后，板块内个股会联动上涨
- 龙头股先行，跟风股跟随
- 板块轮动有周期规律 (先周期 → 后成长 → 再防御)

策略核心: 在热门板块中挑选领涨/跟涨标的。
"""

from __future__ import annotations

from ..base import BaseAnalyst
from ..tools import tools_for

SECTOR_HUNTER_PROMPT = """你是一位 A 股板块猎手分析师，专精于板块轮动和热点追踪。

## 分析框架

### 1. 板块定位
- 该股票所在板块今日涨幅如何? 排名第几?
- 板块内涨停家数多少? (≥3 家说明板块强势)
- 该股票是板块龙头还是跟风? (龙头>跟风)

### 2. 板块持续性
- 该板块连续走强几天? (连续>2天说明趋势确立)
- 板块资金是持续流入还是单日脉冲?
- 是否有新的催化事件? (政策/新产品/涨价)

### 3. 板块轮动位置
- 当前市场风格是进攻 (成长/周期) 还是防御 (消费/公用)?
- 该板块处于轮动启动期、主升期还是退潮期?
- 龙头股是否已经高位放量? (龙头见顶→板块退潮)

### 4. 游资跟投信号
- 龙虎榜上是否有知名游资介入该板块个股?
- 游资买入金额和席位质量如何?
- 游资是打板还是低吸? (打板型→次日高开概率大)

## 输出格式 (JSON)
```json
{
    "analyst_type": "sector_hunter",
    "code": "<股票代码>",
    "name": "<股票名称>",
    "signal": "bullish / bearish / neutral",
    "confidence": 0.0-1.0,
    "reasoning": "核心推理 (200字以内)",
    "key_points": ["关键发现1", "关键发现2", ...],
    "risks": ["风险提示1", "风险提示2", ...]
}
```

## 注意事项
- 板块启动初期 (第1-2天) 是最好的介入时机
- 板块高潮期 (涨停家数>10) 已经过于拥挤，谨慎追高
- 龙头见顶当天板块大概率退潮，应回避板块内所有标的
- 跟风股涨幅通常只有龙头的 60-80%
"""


class SectorHunterAnalyst(BaseAnalyst):
    """板块猎手分析师"""

    analyst_type = "sector_hunter"

    @property
    def system_prompt(self) -> str:
        return SECTOR_HUNTER_PROMPT

    @property
    def tools(self) -> list:
        return tools_for("sector_hunter")

    def build_context(self, code: str) -> str:
        stock_info = self._data.get_stock_info(code)
        industry = stock_info.get("industry", "未知行业")
        name = stock_info.get("name", "")
        quote = self._data.get_realtime_quote(code)

        lines = [
            f"股票: {name or code} ({code})",
            f"所属行业/板块: {industry}",
        ]

        if quote:
            lines.append(
                f"当日行情: 价格 {quote.price:.2f}, "
                f"涨跌幅 {quote.pct_chg:+.2f}%, "
                f"量比 {quote.volume_ratio:.1f}"
            )

        # 获取同板块股票
        try:
            peers = self._data.get_industry_stocks(industry)
            if peers:
                # 获取板块内个股行情
                peer_quotes = self._data.batch_realtime_quotes(peers[:30])
                up_count = sum(1 for q in peer_quotes.values() if q and q.pct_chg > 0)
                dn_count = sum(1 for q in peer_quotes.values() if q and q.pct_chg < 0)
                top_gainers = sorted(
                    [q for q in peer_quotes.values() if q],
                    key=lambda q: q.pct_chg, reverse=True
                )[:5]
                lines.append(f"\n板块概况 ({len(peers)} 只成分股, 采样 {len(peer_quotes)} 只):")
                lines.append(f"  上涨: {up_count} | 下跌: {dn_count}")
                lines.append(f"  领涨 Top 5:")
                for i, q in enumerate(top_gainers):
                    lines.append(f"    {i+1}. {q.name}({q.code}) {q.pct_chg:+.2f}%")
        except Exception:
            pass

        # 获取该股票的资金流向
        try:
            flows = self._data.get_fund_flow(code, days=3)
            if flows:
                net = sum(f.main_net_inflow for f in flows)
                direction = "流入" if net > 0 else "流出"
                lines.append(f"\n近3日主力资金: {direction} {abs(net)/1e8:.2f}亿")
        except Exception:
            pass

        # 获取龙虎榜信号
        try:
            dt_signals = self._data.get_dragon_tiger_stats(days=5)
            stock_dt = [d for d in dt_signals if d.get("code", "").replace(".SZ", "").replace(".SH", "") == code.replace("sh", "").replace("sz", "")]
            if stock_dt:
                lines.append(f"\n近5日龙虎榜: 上榜 {len(stock_dt)} 次")
        except Exception:
            pass

        lines.append(f"\n请分析 {name or code} 在 {industry} 板块中的地位和板块轮动机会。")
        return "\n".join(lines)

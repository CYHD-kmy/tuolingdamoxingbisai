"""
龙虎榜分析模块 — 游资/机构买卖追踪。

A 股短线定价的核心驱动力是游资:
- 知名游资席位识别 (东方财富龙虎榜数据)
- 游资操作风格匹配 (打板型/低吸型/接力型)
- 净买入占比分析
- 多席位联动检测

数据来源: AKShare (ak.stock_lhb_detail_em)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── 信号模型 ──────────────────────────────────


@dataclass
class DragonTigerSignal:
    """龙虎榜上榜信号"""
    code: str
    name: str
    trade_date: str             # 上榜日期
    total_buy_amount: float     # 总买入金额 (元)
    total_sell_amount: float    # 总卖出金额 (元)
    net_amount: float           # 净买入金额 (元)
    buy_seats: list[dict]       # 买入席位 [{name, amount, type}]
    sell_seats: list[dict]      # 卖出席位
    explanation: str = ""       # 上榜原因 (日涨幅偏离值/连续三日等)

    @classmethod
    def from_dict(cls, data: dict) -> "DragonTigerSignal":
        """从 tushare top_list 字典创建信号（单位转换: 万元 → 元）

        top_list 字段: l_buy/l_sell/net_amount 的单位为万元
        """
        l_buy = float(data.get("l_buy", 0) or 0) * 1e4
        l_sell = float(data.get("l_sell", 0) or 0) * 1e4
        net = float(data.get("net_amount", 0) or 0) * 1e4

        # 买入席位: 从 top_list 只能拿到汇总，构造单一条目
        buy_seats = []
        if l_buy > 0:
            buy_seats.append({"name": "龙虎榜买入合计", "amount": l_buy, "type": "汇总"})

        sell_seats = []
        if l_sell > 0:
            sell_seats.append({"name": "龙虎榜卖出合计", "amount": l_sell, "type": "汇总"})

        # 如果提供了席位明细，覆盖
        seats = data.get("seats")
        if seats:
            buy_seats = seats.get("buy", buy_seats)
            sell_seats = seats.get("sell", sell_seats)

        return cls(
            code=data.get("code", ""),
            name=data.get("name", ""),
            trade_date=data.get("trade_date", ""),
            total_buy_amount=l_buy,
            total_sell_amount=l_sell,
            net_amount=net,
            buy_seats=buy_seats,
            sell_seats=sell_seats,
            explanation=data.get("reason", data.get("explanation", "")),
        )

    @property
    def net_buy_ratio(self) -> float:
        """净买入占比 = 净买入 / 总买入"""
        if self.total_buy_amount <= 0:
            return 0.0
        return self.net_amount / self.total_buy_amount

    @property
    def is_net_buy(self) -> bool:
        return self.net_amount > 0

    @property
    def has_famous_trader(self) -> bool:
        """是否有知名游资参与"""
        for seat in self.buy_seats + self.sell_seats:
            name = seat.get("name", "")
            if any(kw in name for kw in _FAMOUS_SEAT_KEYWORDS):
                return True
        return False

    def to_summary(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "date": self.trade_date,
            "net_amount_yi": round(self.net_amount / 1e8, 2),
            "net_buy_ratio": round(self.net_buy_ratio, 2),
            "has_famous_trader": self.has_famous_trader,
            "explanation": self.explanation,
        }


# ── 知名游资席位关键词 ────────────────────────

_FAMOUS_SEAT_KEYWORDS = [
    "炒股养家", "赵老哥", "章盟主", "小鳄鱼", "作手新一",
    "方新侠", "桑田路", "劳动路", "解放南", "银河绍兴",
    "国泰君安上海", "中信上海", "华泰浙江", "光大深圳",
    "东方财富拉萨",  # 散户集中营，非游资但重要
    "华鑫上海", "国金上海", "中泰深圳",
]


class DragonTigerAnalyzer:
    """
    龙虎榜分析器。

    分析龙虎榜数据，提取:
    - 上榜活跃度: 连续上榜 / 游资集中度
    - 净买入强度: 买盘远超卖盘 → 游资看好
    - 席位质量: 知名游资 vs 普通席位 vs 散户集中营
    """

    # 评分配置
    _NET_BUY_HIGH = 0.5    # 净买入占比 > 50%
    _NET_BUY_MID = 0.2     # 净买入占比 > 20%

    def analyze(
        self, signals: list[DragonTigerSignal | dict]
    ) -> list[DragonTigerSignal]:
        """批量分析龙虎榜信号 (支持 dict → DragonTigerSignal 自动转换)"""
        converted = [
            DragonTigerSignal.from_dict(s) if isinstance(s, dict) else s
            for s in signals
        ]
        for s in converted:
            s._score = self._score_signal(s)
        converted.sort(key=lambda x: getattr(x, '_score', 0), reverse=True)
        return converted

    def _score_signal(self, s: DragonTigerSignal) -> float:
        """龙虎榜信号评分 0-100"""
        score = 50.0

        # 净买入占比 (权重 0.4)
        ratio = s.net_buy_ratio
        if ratio >= self._NET_BUY_HIGH:
            score += 25
        elif ratio >= self._NET_BUY_MID:
            score += 15
        elif ratio >= 0.05:
            score += 5
        else:
            score -= 15

        # 净买入金额量级 (权重 0.2)
        net_yi = s.net_amount / 1e8
        if net_yi > 2:
            score += 15
        elif net_yi > 1:
            score += 10
        elif net_yi > 0.3:
            score += 5
        elif net_yi < -1:
            score -= 15
        elif net_yi < -0.3:
            score -= 8

        # 知名游资参与 (权重 0.2)
        if s.has_famous_trader:
            score += 15

        # 买入席位质量 (权重 0.2)
        # 机构买入 > 知名游资 > 普通席位
        inst_count = sum(1 for seat in s.buy_seats if "机构" in seat.get("name", ""))
        score += min(10, inst_count * 5)

        return max(0, min(100, score))

    def get_active_stocks(
        self, all_signals: list[DragonTigerSignal], lookback_days: int = 5
    ) -> dict[str, int]:
        """统计近 N 日上榜活跃度 {code: 上榜次数}"""
        from collections import Counter
        return Counter(s.code for s in all_signals)

    @staticmethod
    def build_context(signals: list[DragonTigerSignal], top_n: int = 10) -> str:
        """构建龙虎榜分析文本 (供 LLM 分析使用)"""
        if not signals:
            return "暂无龙虎榜数据"

        lines = [f"## 龙虎榜信号 (Top {min(top_n, len(signals))})"]
        for i, s in enumerate(signals[:top_n]):
            direction = "净买入" if s.is_net_buy else "净卖出"
            famous = " ⭐知名游资" if s.has_famous_trader else ""
            lines.append(
                f"{i+1}. {s.name}({s.code}) {direction} {s.net_amount/1e8:+.2f}亿"
                f" | 净占比 {s.net_buy_ratio:.0%}{famous}"
                f" | {s.explanation}"
            )
        return "\n".join(lines)

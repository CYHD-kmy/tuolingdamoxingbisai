"""
涨停板分析模块 — 涨停强度与接力分析。

A 股涨停板是短线最核心的信号:
- 涨停强度 (封板时间、封单量、开板次数)
- 连板分析 (首板/二板/三板 → 接力追进)
- 涨停原因 (题材驱动/公告驱动/资金驱动)
- 溢价概率 (次日高开概率)

数据来源: AKShare (ak.stock_zt_pool_em)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── 信号模型 ──────────────────────────────────


@dataclass
class LimitUpSignal:
    """涨停信号"""
    code: str
    name: str
    trade_date: str
    limit_up_time: str              # 封板时间 (HH:MM:SS)
    consecutive_days: int = 0       # 连板天数
    first_limit_up_time: str = ""   # 首次封板时间
    open_count: int = 0             # 开板次数
    limit_up_amt: float = 0.0       # 封单额 (元)
    turnover_rate: float = 0.0      # 换手率
    amount: float = 0.0             # 成交额 (元)
    pct_chg: float = 0.0            # 涨幅
    reason: str = ""                # 涨停原因
    sector: str = ""                # 所属板块

    @property
    def ban_strength(self) -> str:
        """封板强度分类"""
        if self.open_count == 0 and self.consecutive_days >= 3:
            return "换手三板"  # 高标接力
        if self.open_count == 0 and self.limit_up_time <= "09:40":
            return "秒板"       # 最强
        if self.open_count == 0 and self.limit_up_time <= "10:00":
            return "早盘封板"   # 强势
        if self.open_count <= 1 and self.limit_up_time <= "10:30":
            return "上午封板"   # 一般
        if self.open_count <= 3:
            return "尾盘封板"   # 偏弱
        return "反复开板"       # 最弱

    @property
    def seal_amount_ratio(self) -> float:
        """封单成交比: 封单额 / 成交额"""
        if self.amount <= 0:
            return 0.0
        return self.limit_up_amt / self.amount

    @property
    def is_quality_ban(self) -> bool:
        """是否高质量涨停 (值得跟踪接力)"""
        return (
            self.ban_strength in ("秒板", "早盘封板") and
            self.seal_amount_ratio > 0.3 and
            self.open_count <= 1
        )


# ── 分析器 ────────────────────────────────────


class LimitUpAnalyzer:
    """
    涨停板分析器。

    分析当日涨停板数据，输出:
    - 涨停强度排序
    - 连板接力候选
    - 板块涨停热度
    """

    def analyze(self, signals: list[LimitUpSignal]) -> list[LimitUpSignal]:
        """批量分析涨停信号"""
        for s in signals:
            s._score = self._score_ban(s)
        signals.sort(key=lambda x: getattr(x, '_score', 0), reverse=True)
        return signals

    def _score_ban(self, s: LimitUpSignal) -> float:
        """涨停质量评分 0-100"""
        score = 50.0

        # 封板时间 (权重 0.25)
        time = s.limit_up_time
        if time <= "09:35":
            score += 20  # 秒板
        elif time <= "09:45":
            score += 15  # 早盘封板
        elif time <= "10:00":
            score += 10
        elif time <= "10:30":
            score += 5
        elif time <= "11:30":
            score += 2
        elif time <= "14:00":
            score -= 5
        else:
            score -= 15  # 尾盘偷袭

        # 开板次数 (权重 0.15)
        if s.open_count == 0:
            score += 10
        elif s.open_count == 1:
            score += 5
        elif s.open_count <= 3:
            score -= 5
        else:
            score -= 15

        # 封单成交比 (权重 0.2)
        ratio = s.seal_amount_ratio
        if ratio > 1.0:
            score += 15
        elif ratio > 0.5:
            score += 10
        elif ratio > 0.2:
            score += 5
        else:
            score -= 5

        # 换手率 (权重 0.15)
        turnover = s.turnover_rate
        if 5 <= turnover <= 20:
            score += 10   # 健康换手
        elif 3 <= turnover < 5:
            score += 5    # 略低
        elif turnover > 30:
            score -= 10   # 出货嫌疑

        # 连板加分 (权重 0.25) - 适度连板是强势信号
        if s.consecutive_days == 1:
            score += 8    # 首板: 安全性高
        elif s.consecutive_days == 2:
            score += 12   # 二板: 确认信号
        elif s.consecutive_days == 3:
            score += 10   # 三板: 加速期
        elif s.consecutive_days >= 4:
            score += 5    # 高标: 风险增大

        return max(0, min(100, score))

    def get_continuation_candidates(
        self, signals: list[LimitUpSignal]
    ) -> list[LimitUpSignal]:
        """筛选连板接力候选 (首板+二板且有板块支撑的)"""
        return [
            s for s in signals
            if s.consecutive_days <= 2 and
            s.ban_strength in ("秒板", "早盘封板") and
            s.open_count <= 1
        ]

    def get_sector_limit_up_heat(
        self, signals: list[LimitUpSignal]
    ) -> dict[str, int]:
        """板块涨停热度统计 {板块: 涨停家数}"""
        from collections import Counter
        return Counter(s.sector for s in signals if s.sector)

    @staticmethod
    def build_context(signals: list[LimitUpSignal], top_n: int = 15) -> str:
        """构建涨停分析文本 (供 LLM 分析使用)"""
        if not signals:
            return "暂无涨停数据"

        lines = [f"## 涨停板分析 (共 {len(signals)} 只涨停, 展示 Top {min(top_n, len(signals))})"]
        for i, s in enumerate(signals[:top_n]):
            consec = f"{s.consecutive_days}连板" if s.consecutive_days >= 2 else "首板"
            lines.append(
                f"{i+1}. {s.name}({s.code}) {consec} | "
                f"{s.ban_strength} ({s.limit_up_time}) | "
                f"开板{s.open_count}次 | 封单比{s.seal_amount_ratio:.1%} | "
                f"换手{s.turnover_rate:.1f}% | {s.reason}"
            )
        return "\n".join(lines)

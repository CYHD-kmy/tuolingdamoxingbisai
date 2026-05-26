"""
辩论与决策数据模型。

定义研究员辩论 → 研究主管 → 风控 → 组合主管 全链路的数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── 辩论 ──────────────────────────────────────

@dataclass
class DebateRound:
    """一轮辩论记录"""
    round_num: int
    bull_argument: str        # 多头论点
    bear_argument: str = ""   # 空头反驳
    bull_rebuttal: str = ""   # 多头回应 (第二轮起)
    bear_summary: str = ""    # 空头总结


@dataclass
class DebateResult:
    """某只股票的完整辩论结果"""
    code: str
    name: str
    rounds: list[DebateRound] = field(default_factory=list)
    total_rounds: int = 0


# ── 研究结论 ──────────────────────────────────

@dataclass
class ResearchVerdict:
    """研究主管给出的最终研报结论"""
    code: str
    name: str
    direction: str            # "buy" / "sell" / "hold"
    confidence: float         # 0.0 ~ 1.0
    target_price: float = 0.0
    risk_level: str = "medium"  # "low" / "medium" / "high"
    core_reasoning: str = ""    # 核心理由
    key_risks: list[str] = field(default_factory=list)


# ── 风控 ──────────────────────────────────────

@dataclass
class PositionLimit:
    """单只股票的风控约束"""
    code: str
    name: str
    max_position_pct: float   # 最大仓位比例 (0-1)
    max_shares: int           # 最大可买股数 (100的整数倍)
    max_value: float          # 最大可买金额
    volatility: float = 0.0   # 近期波动率 %
    risk_flags: list[str] = field(default_factory=list)


# ── 最终决策 ──────────────────────────────────

@dataclass
class FinalDecision:
    """组合主管输出的最终买卖决策 (赛道标准格式)"""
    symbol: str
    symbol_name: str
    volume: int
    entry_price: float = 0.0  # 入场价格，用于计算浮动盈亏

    def to_dict(self) -> dict:
        d = {"symbol": self.symbol, "symbol_name": self.symbol_name, "volume": self.volume}
        if self.entry_price > 0:
            d["entry_price"] = self.entry_price
        return d


@dataclass
class PortfolioResult:
    """完整的组合决策结果"""
    decisions: list[FinalDecision] = field(default_factory=list)
    cash_used: float = 0.0
    cash_remaining: float = 0.0
    total_positions: int = 0
    risk_summary: str = ""

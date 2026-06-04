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
    asset_type: str = "stock"   # "stock" / "etf"


# ── 风控 ──────────────────────────────────────

@dataclass
class PositionLimit:
    """单只股票/ETF 的风控约束"""
    code: str
    name: str
    max_position_pct: float   # 最大仓位比例 (0-1)
    max_shares: int           # 最大可买股数 (100的整数倍)
    max_value: float          # 最大可买金额
    volatility: float = 0.0   # 近期波动率 %
    risk_flags: list[str] = field(default_factory=list)
    asset_type: str = "stock"  # "stock" / "etf"
    tier: str = "satellite"    # "core" / "satellite" (三级仓位分层)


# ── 最终决策 ──────────────────────────────────

@dataclass
class FinalDecision:
    """组合主管输出的最终买卖决策 (赛道标准格式)"""
    symbol: str
    symbol_name: str
    volume: int
    entry_price: float = 0.0  # 入场价格，用于计算浮动盈亏
    asset_type: str = "stock"  # "stock" / "etf"
    direction: str = "buy"     # "buy" / "sell"

    def to_dict(self) -> dict:
        """返回赛道标准 JSON 格式: symbol / symbol_name / volume"""
        return {
            "symbol": self.symbol,
            "symbol_name": self.symbol_name,
            "volume": self.volume,
        }


@dataclass
class TargetAllocation:
    """LLM 输出的目标仓位分配"""
    code: str
    name: str
    target_weight: float       # 0.0 ~ 1.0, 0=清仓
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class PortfolioResult:
    """完整的组合决策结果"""
    decisions: list[FinalDecision] = field(default_factory=list)
    cash_used: float = 0.0
    cash_remaining: float = 0.0
    total_positions: int = 0
    risk_summary: str = ""
    target_allocations: list[TargetAllocation] = field(default_factory=list)
    sell_proceeds: float = 0.0  # 当日卖出回收资金

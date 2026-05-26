"""
策略竞争引擎 — 并行运行多种 Alpha 策略，合并候选，追踪表现。
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .base import BaseStrategy, StrategyResult
from .registry import StrategyRegistry, StrategyPerformance
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .quality import QualityStrategy
from .sentiment import SentimentStrategy
from .default_strategy import DefaultStrategy
from ..screening.scorer import FactorScore

logger = logging.getLogger(__name__)


@dataclass
class CompetitionResult:
    """策略竞争结果"""
    strategy_results: dict[str, StrategyResult] = field(default_factory=dict)
    merged_candidates: list[FactorScore] = field(default_factory=list)
    allocation: dict[str, float] = field(default_factory=dict)
    performance: dict[str, StrategyPerformance] = field(default_factory=dict)


class CompetitionEngine:
    """
    策略竞争引擎 — 并行运行多种策略，合并候选并追踪表现。

    使用方式:
        engine = CompetitionEngine(strategies=["momentum", "quality"])
        result = engine.run(daily_data, fund_flows)
        top = result.merged_candidates[:10]
    """

    def __init__(
        self,
        strategies: list[str] | None = None,
        performance_file: str = "",
    ) -> None:
        self._register_default_strategies()
        self._strategy_names = strategies or list(StrategyRegistry.list_names())
        self._performance_file = performance_file or os.path.join(
            os.path.dirname(__file__), "..", "..", "results", "strategy_performance.json",
        )
        self._load_performance()

    def run(
        self,
        snapshots: list,
        daily_data: dict[str, list],
        fund_flows: dict[str, list],
    ) -> CompetitionResult:
        """
        并行运行所有已注册策略。

        snapshots: 全市场快照列表
        daily_data: {code: [StockDaily, ...]}
        fund_flows: {code: [FundFlow, ...]}

        返回: CompetitionResult
        """
        strategy_results: dict[str, StrategyResult] = {}
        active_strategies = {
            name: strat
            for name, strat in StrategyRegistry.get_all().items()
            if name in self._strategy_names or "all" in self._strategy_names
        }

        if not active_strategies:
            logger.warning("无活跃策略")
            return CompetitionResult()

        # 并行执行
        with ThreadPoolExecutor(max_workers=min(len(active_strategies), 6)) as pool:
            futures = {
                pool.submit(self._run_single, name, strat, snapshots, daily_data, fund_flows): name
                for name, strat in active_strategies.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    strategy_results[name] = result
                    logger.info("策略 %s: %d 候选 (%.1fs)",
                                name, len(result.candidates),
                                result.metadata.get("elapsed", 0))
                except Exception as e:
                    logger.warning("策略 %s 执行失败: %s", name, e)
            del futures

        # 合并候选: 按多策略命中次数 + 平均得分排序
        merged = self._merge_candidates(strategy_results)

        # 计算当前资金分配 (softmax over historical Sharpe)
        allocation = self._compute_allocation()

        return CompetitionResult(
            strategy_results=strategy_results,
            merged_candidates=merged,
            allocation=allocation,
            performance=StrategyRegistry.get_all_performance(),
        )

    def rebalance(self, lookback: int = 20) -> dict[str, float]:
        """
        根据历史表现重新分配策略权重。

        使用 softmax over Sharpe: alloc_i = exp(sharpe_i/10) / sum(exp(sharpe_j/10))
        """
        perf = StrategyRegistry.get_all_performance()
        if not perf:
            return {}

        sharpes = {}
        for name, p in perf.items():
            if p.total_runs < 3:
                sharpes[name] = 0.0
            else:
                sharpes[name] = max(-5, min(5, p.sharpe_approx))

        # Softmax
        import math
        exp_sum = sum(math.exp(s / 5) for s in sharpes.values())
        if exp_sum > 0:
            allocation = {name: round(math.exp(s / 5) / exp_sum, 4) for name, s in sharpes.items()}
        else:
            n = len(sharpes)
            allocation = {name: 1.0 / n for name in sharpes}

        logger.info("策略资金分配: %s",
                     {k: f"{v:.1%}" for k, v in sorted(allocation.items(), key=lambda x: -x[1])})
        return allocation

    # ── 内部 ──────────────────────────────────

    @staticmethod
    def _register_default_strategies() -> None:
        """注册内置策略"""
        for strat_cls in [DefaultStrategy, MomentumStrategy, MeanReversionStrategy,
                          QualityStrategy, SentimentStrategy]:
            if strat_cls.name not in StrategyRegistry.list_names():
                StrategyRegistry.register(strat_cls())

    @staticmethod
    def _run_single(
        name: str,
        strategy: BaseStrategy,
        snapshots: list,
        daily_data: dict[str, list],
        fund_flows: dict[str, list],
    ) -> StrategyResult:
        import time
        t0 = time.monotonic()
        result = strategy.run(snapshots, daily_data, fund_flows)
        result.metadata["elapsed"] = round(time.monotonic() - t0, 2)
        return result

    @staticmethod
    def _merge_candidates(strategy_results: dict[str, StrategyResult]) -> list[FactorScore]:
        """
        合并多策略候选。

        策略: 统计每只股票被多少策略选中 + 平均得分 → 排序。
        """
        code_hits: dict[str, int] = {}
        code_scores: dict[str, list[float]] = {}
        code_name: dict[str, str] = {}
        code_all_scores: dict[str, dict[str, float]] = {}

        for result in strategy_results.values():
            for c in result.candidates:
                code = c.code
                code_hits[code] = code_hits.get(code, 0) + 1
                if code not in code_scores:
                    code_scores[code] = []
                code_scores[code].append(c.composite)
                if code not in code_name:
                    code_name[code] = c.name
                if code not in code_all_scores:
                    code_all_scores[code] = dict(c.scores)

        merged = []
        for code, hits in code_hits.items():
            avg_score = sum(code_scores[code]) / len(code_scores[code])
            # 多策略共识加分: 每多一个策略选中 +5%
            consensus_bonus = 1.0 + (hits - 1) * 0.05
            final_score = min(95, avg_score * consensus_bonus)

            merged.append(FactorScore(
                code=code,
                name=code_name.get(code, code),
                composite=round(final_score, 1),
                scores=code_all_scores.get(code, {}),
            ))

        merged.sort(key=lambda x: x.composite, reverse=True)
        logger.info("合并候选: %d 策略 → %d 候选 → %d 合并",
                     len(strategy_results),
                     sum(len(r.candidates) for r in strategy_results.values()),
                     len(merged))
        return merged

    def _compute_allocation(self) -> dict[str, float]:
        """计算当前资金分配"""
        perf = StrategyRegistry.get_all_performance()
        if not perf:
            return {}

        import math
        total_runs = sum(p.total_runs for p in perf.values())
        if total_runs < 3:
            n = len(perf)
            return {name: round(1.0 / n, 4) for name in perf}

        return self.rebalance()

    def _load_performance(self) -> None:
        """从文件加载历史表现"""
        if not os.path.exists(self._performance_file):
            return
        try:
            with open(self._performance_file, encoding="utf-8") as f:
                data = json.load(f)
            for name, perf_data in data.items():
                perf = StrategyRegistry.get_performance(name)
                if perf:
                    perf.total_runs = perf_data.get("total_runs", 0)
                    perf.win_count = perf_data.get("win_count", 0)
                    perf.cumulative_return = perf_data.get("cumulative_return", 0.0)
                    perf.returns = perf_data.get("returns", [])
            logger.info("策略表现已加载: %d 条记录", len(data))
        except Exception:
            logger.debug("策略表现加载失败", exc_info=True)

    def save_performance(self) -> None:
        """保存历史表现到文件"""
        try:
            os.makedirs(os.path.dirname(self._performance_file), exist_ok=True)
            data = {}
            for name, perf in StrategyRegistry.get_all_performance().items():
                data[name] = {
                    "total_runs": perf.total_runs,
                    "win_count": perf.win_count,
                    "cumulative_return": perf.cumulative_return,
                    "returns": perf.returns[-100:],  # 只保留最近100条
                }
            with open(self._performance_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.debug("策略表现保存失败", exc_info=True)

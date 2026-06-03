"""
海选流水线 — 串联过滤 → 打分 → 取 Top-N 的完整流程。

使用方式:
    from src.data.interface import UnifiedDataInterface
    from src.screening.pipeline import ScreeningPipeline

    udi = UnifiedDataInterface()
    pipeline = ScreeningPipeline(udi)
    candidates = pipeline.run(top_n=20)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from ..data.interface import UnifiedDataInterface
from ..utils.config import get_config
from .filters import (
    filter_tradable,
    filter_liquidity,
    filter_volatility,
    extract_codes,
)
from .scorer import ScreeningScorer, FactorScore

logger = logging.getLogger(__name__)


@dataclass
class ScreeningResult:
    """海选结果"""
    candidates: list[FactorScore]       # Top-N 候选，按得分降序
    total_screened: int                  # 全市场数量
    after_filters: int                   # 过滤后数量
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class ScreeningPipeline:
    """
    海选流水线。

    流程:
        全市场快照 → 过滤(行情) → 批量拉日线+资金 → 过滤(波动率) → 多因子打分 → Top-N
    """

    def __init__(self, data: UnifiedDataInterface) -> None:
        self._data = data
        self._config = get_config()
        self._scorer = ScreeningScorer()

    def run(self, top_n: int | None = None) -> ScreeningResult:
        """
        执行完整海选流水线。

        top_n: 最终候选数量，默认使用配置值
        """
        if top_n is None:
            top_n = self._config.max_candidates

        t0 = time.monotonic()
        errors: list[str] = []

        # ── 1. 全市场快照 ──────────────────────
        logger.info("===== 海选开始 =====")
        snapshots = self._data.get_market_snapshot()
        if not snapshots:
            return ScreeningResult(candidates=[], total_screened=0, after_filters=0, errors=["全市场快照获取失败"])
        total = len(snapshots)
        logger.info("1/5 全市场快照: %d 只", total)

        # ── 1b. 预热 AKShare 缓存 (避免批量查询时线程争抢) ──
        try:
            from ..data.fetchers.akshare_fetcher import AKShareFetcher
            AKShareFetcher._warm_spot_cache()
        except Exception:
            logger.debug("AKShare 缓存预热失败，跳过")

        # ── 1c. 预热 BaoStock 股票基本信息缓存 (避免 3000 只线程争抢) ──
        try:
            from ..data.fetchers.baostock_fetcher import BaoStockFetcher
            BaoStockFetcher._warm_stock_basic_cache()
        except Exception:
            logger.debug("BaoStock 缓存预热失败，跳过")

        # ── 2. 基础过滤 (ST/停牌/新股) ──────────
        # 先做一轮快速过滤 (ST/停牌), 避免为全市场 3000 只各自查询 stock_info
        pre_filtered = filter_tradable(snapshots, stock_infos=None)
        logger.info("2/5 快速过滤(名称/停牌): %d 只", len(pre_filtered))

        # 仅对通过快速过滤的股票批量获取基本信息 (新股票过滤需要 IPO 日期)
        codes_pre = extract_codes(pre_filtered)
        if codes_pre:
            stock_infos = self._data.batch_stock_info(codes_pre)
        else:
            stock_infos = {}
        logger.debug("2/5 批量获取 %d 只股票基本信息完成", len(stock_infos))

        # 第二次过滤: 用基本信息做新股票过滤
        tradable = filter_tradable(pre_filtered, stock_infos)
        logger.info("2/5 新股过滤后: %d 只", len(tradable))

        # ── 3. 流动性过滤 ───────────────────────
        liquid = filter_liquidity(tradable)
        if not liquid:
            return ScreeningResult(candidates=[], total_screened=total, after_filters=0, errors=["无股票通过流动性过滤"])
        logger.info("3/5 流动性过滤: %d 只", len(liquid))

        # ── 3b. 按成交额截断 (避免批量拉日线过多) ──
        max_daily_batch = self._config.max_candidates * 25  # 500 for default max_candidates=20
        if len(liquid) > max_daily_batch:
            liquid.sort(key=lambda s: s.amount, reverse=True)
            liquid = liquid[:max_daily_batch]
            logger.info("3b/5 成交额截断: %d 只", len(liquid))

        # ── 4. 批量拉取日线 ────────────────────
        codes = extract_codes(liquid)
        logger.info("4/5 批量拉取 %d 只股票日线数据...", len(codes))

        daily_data = self._data.batch_daily_data(codes, days=30, max_workers=4)

        # ── 5. 波动率过滤 ───────────────────────
        exclude_vol = filter_volatility(daily_data)
        if exclude_vol:
            codes = [c for c in codes if c not in exclude_vol]
            liquid = [s for s in liquid if s.code not in exclude_vol]
            logger.info("5/5 波动率过滤后: %d 只", len(codes))

        # ── 5b. 第一轮打分 (基础因子) ───────────
        # 仅用快照+日线数据快速打分, 避免对 500 只股票逐只拉取增强数据
        snap_map = {s.code: s for s in liquid}
        round1_scored = self._scorer.score_all(
            codes, snap_map, daily_data,
            fund_flows={},           # 第二轮再拉
            northbound_stocks=None,  # 第二轮再拉
            financials=None,         # 第二轮再拉
            shareholders=None,       # 第二轮再拉
        )
        round1_top = self._scorer.top_n(round1_scored, n=min(top_n * 2, len(round1_scored)))
        codes_r1 = [fs.code for fs in round1_top]
        logger.info("5b/5 第一轮基础打分完成: Top-%d → %d 只候选进入增强分析",
                    top_n * 2, len(codes_r1))

        # ── 6. 对 Top-40 候选拉取增强数据 ─────────
        # 资金流向 + 北向持仓 + 财务指标 + 筹码集中度
        fund_flows = self._data.batch_fund_flows(codes_r1, days=5, max_workers=8)
        northbound_stocks = self._data.batch_northbound_stocks(codes_r1, days=10, max_workers=6)
        financials = self._data.batch_financials(codes_r1, max_workers=4)
        shareholders = self._data.batch_shareholders(codes_r1, max_workers=6)

        # ── 7. 第二轮打分 (全因子) ───────────────
        # 用增强数据重新打分, 此时 capital_flow/northbound 等因子可用
        round2_scored = self._scorer.score_all(
            codes_r1, snap_map, daily_data, fund_flows,
            northbound_stocks=northbound_stocks,
            financials=financials,
            shareholders=shareholders,
        )
        top = self._scorer.top_n(round2_scored, n=top_n)

        elapsed = time.monotonic() - t0

        # 日志输出
        if top:
            logger.info("===== 海选完成: Top-%d (%.1fs) =====", len(top), elapsed)
            for i, fs in enumerate(top, 1):
                logger.info(
                    "  %2d. %s %s  综合:%.1f  %s",
                    i, fs.code, fs.name, fs.composite,
                    _describe_scores(fs),
                )
        else:
            logger.warning("海选完成: 无候选标的")

        return ScreeningResult(
            candidates=top,
            total_screened=total,
            after_filters=len(codes),
            elapsed_seconds=elapsed,
            errors=errors,
        )


def _describe_scores(fs: FactorScore) -> str:
    """简短描述各因子得分"""
    parts = []
    for k, v in fs.scores.items():
        short = {"trend": "趋势", "momentum": "动量", "volume_price": "量价",
                 "capital_flow": "资金", "northbound": "北向", "sentiment": "情绪",
                 "quality": "质量", "risk": "风险", "liquidity": "流动性",
                 "shareholder_conc": "筹码"}.get(k, k)
        parts.append(f"{short}:{v:.0f}")
    return " ".join(parts)

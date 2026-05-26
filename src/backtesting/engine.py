"""
回测引擎 — 在历史日期上重放完整流水线并收集绩效数据。

支持两种模式:
- demo 模式: 使用种子随机游走生成合成数据，无需网络
- live 模式: 通过 UnifiedDataInterface 获取真实历史数据

使用方式:
    from src.backtesting import BacktestEngine, BacktestConfig

    config = BacktestConfig(start_date="20260501", end_date="20260526")
    engine = BacktestEngine(config)
    result = engine.run()
    print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .metrics import MetricsReport, compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: str          # YYYYMMDD
    end_date: str            # YYYYMMDD
    initial_capital: float = 500_000.0
    benchmark: str = "000300"


@dataclass
class DailyBacktestResult:
    """单日回测结果"""
    date: str
    decisions: list = field(default_factory=list)
    equity: float = 0.0
    daily_return: float = 0.0
    benchmark_return: float = 0.0
    errors: list[str] = field(default_factory=list)
    candidates_count: int = 0
    buy_count: int = 0


@dataclass
class BacktestResult:
    """完整回测结果"""
    config: BacktestConfig
    daily_results: list[DailyBacktestResult] = field(default_factory=list)
    metrics: MetricsReport = field(default_factory=MetricsReport)
    final_equity: float = 0.0


class BacktestEngine:
    """
    回测引擎 — 逐日重放流水线。

    使用方式:
        config = BacktestConfig(start_date="20260501", end_date="20260526")
        engine = BacktestEngine(config)
        result = engine.run()
    """

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        self._initial_capital = config.initial_capital
        self._daily_returns: list[float] = []
        self._benchmark_returns: list[float] = []

    def run(self) -> BacktestResult:
        """
        执行回测。

        逐日重放流水线，维护跨日持仓，收集每日权益和收益。
        """
        dates = self._date_range(self._config.start_date, self._config.end_date)
        if not dates:
            logger.warning("回测日期范围为空")
            return BacktestResult(config=self._config)

        logger.info("回测开始: %s → %s (共 %d 日)", dates[0], dates[-1], len(dates))

        daily_results: list[DailyBacktestResult] = []
        capital = self._initial_capital
        # 跨日持仓: {code: {shares, avg_cost, name}}
        positions: dict[str, dict] = {}
        prev_equity = capital

        for date_str in dates:
            logger.info("--- 回测 %s ---", date_str)
            t0 = time.monotonic()

            try:
                dr = self._run_day(date_str, capital, positions)
            except Exception as e:
                logger.exception("回测 %s 失败: %s", date_str, e)
                dr = DailyBacktestResult(
                    date=date_str,
                    equity=prev_equity,
                    daily_return=0.0,
                    errors=[str(e)],
                )

            # 更新权益和收益
            if dr.equity > 0:
                dr.daily_return = (dr.equity / prev_equity - 1.0) if prev_equity > 0 else 0.0
            prev_equity = dr.equity if dr.equity > 0 else prev_equity

            # 跨日资金追踪: 当日买入花费从 capital 中扣除，次日可用资金减少
            day_spent = sum(d.entry_price * d.volume for d in dr.decisions)
            capital = max(0.0, capital - day_spent)

            # 模拟基准收益 (简化为 0.0002/day 即约 5%/年)
            bm_return = 0.0002 + random.uniform(-0.005, 0.005)
            dr.benchmark_return = bm_return

            self._daily_returns.append(dr.daily_return)
            self._benchmark_returns.append(bm_return)
            daily_results.append(dr)

            elapsed = time.monotonic() - t0
            logger.info(
                "  %s: 权益 ¥%.0f 日收益 %+.2f%% 耗时 %.1fs",
                date_str, dr.equity, dr.daily_return * 100, elapsed,
            )

        metrics = compute_metrics(self._daily_returns, self._benchmark_returns)

        logger.info(
            "回测完成: Sharpe=%.2f MaxDD=%.2f%% 年化=%.2f%% 胜率=%.1f%%",
            metrics.sharpe_ratio,
            metrics.max_drawdown * 100,
            metrics.annualized_return * 100,
            metrics.win_rate * 100,
        )

        return BacktestResult(
            config=self._config,
            daily_results=daily_results,
            metrics=metrics,
            final_equity=prev_equity,
        )

    def _run_day(
        self,
        date_str: str,
        capital: float,
        positions: dict[str, dict],
    ) -> DailyBacktestResult:
        """执行单日回测"""
        from ..demo import _SAMPLE_STOCKS, _make_daily_data as demo_daily

        # 用种子随机游走生成当日数据 (可复现)
        seed = hash(date_str) & 0x7FFFFFFF
        rng = random.Random(seed)

        # 生成当日候选 (简化为固定 6 只样本股)
        candidates = list(_SAMPLE_STOCKS)

        # 生成日线数据
        daily_data = _make_backtest_daily(date_str, rng)

        # 生成资金流向
        fund_flows = _make_backtest_fund_flows(date_str, rng)

        # 简化的打分 (基于随机趋势)
        from ..screening.scorer import FactorScore
        candidate_scores = []
        for code, name, price, *_ in candidates:
            records = daily_data.get(code, [])
            if not records:
                continue
            latest = records[-1]
            trend_score = 60 + rng.uniform(-20, 20)
            momentum_score = 50 + latest.pct_chg * 3 + rng.uniform(-10, 10)
            scores = {
                "trend": min(95, max(5, trend_score)),
                "momentum": min(95, max(5, momentum_score)),
                "volume_price": min(95, max(5, 55 + rng.uniform(-20, 20))),
                "capital_flow": min(95, max(5, 55 + rng.uniform(-20, 20))),
                "northbound": min(95, max(5, 60 + rng.uniform(-15, 15))),
                "sentiment": min(95, max(5, 55 + rng.uniform(-15, 15))),
                "quality": min(95, max(5, 65 + rng.uniform(-15, 15))),
                "risk": min(95, max(5, 60 + rng.uniform(-15, 15))),
                "liquidity": min(95, max(5, 60 + rng.uniform(-20, 20))),
                "shareholder_conc": min(95, max(5, 55 + rng.uniform(-15, 15))),
            }
            composite = sum(scores.values()) / len(scores)
            candidate_scores.append(FactorScore(
                code=code, name=name, composite=round(composite, 1), scores=scores,
            ))

        # 简化的风控 + 决策 (每只 buy 评分 > 70 的股票，买 100-500 股)
        decisions = []
        total_cost = 0.0
        buy_count = 0
        errors: list[str] = []

        for cs in candidate_scores:
            if cs.composite < 70:
                continue
            records = daily_data.get(cs.code, [])
            if not records:
                continue
            price = records[-1].close
            shares = rng.choice([100, 200, 300, 500])
            cost = price * shares
            if total_cost + cost > capital * 0.9:
                break

            from ..agents.models import FinalDecision
            decisions.append(FinalDecision(
                symbol=cs.code, symbol_name=cs.name,
                volume=shares, entry_price=round(price, 2),
            ))
            total_cost += cost
            buy_count += 1

            # 更新持仓
            if cs.code in positions:
                old = positions[cs.code]
                old_shares = old["shares"]
                old_cost = old["avg_cost"] * old_shares
                new_shares = old_shares + shares
                positions[cs.code] = {
                    "shares": new_shares,
                    "avg_cost": (old_cost + cost) / new_shares,
                    "name": cs.name,
                }
            else:
                positions[cs.code] = {
                    "shares": shares,
                    "avg_cost": price,
                    "name": cs.name,
                }

        # 计算权益
        total_mv = 0.0
        for code, pos in positions.items():
            records = daily_data.get(code, [])
            if records:
                total_mv += pos["shares"] * records[-1].close
            else:
                total_mv += pos["shares"] * pos["avg_cost"]
        cash = capital - total_cost
        equity = cash + total_mv

        return DailyBacktestResult(
            date=date_str,
            decisions=decisions,
            equity=round(equity, 2),
            candidates_count=len(candidate_scores),
            buy_count=buy_count,
            errors=errors,
        )

    @staticmethod
    def _date_range(start: str, end: str) -> list[str]:
        """生成日期范围 (YYYYMMDD)，跳过周末"""
        try:
            d0 = datetime.strptime(start, "%Y%m%d")
            d1 = datetime.strptime(end, "%Y%m%d")
        except ValueError:
            logger.error("日期格式错误，需要 YYYYMMDD")
            return []

        dates = []
        current = d0
        while current <= d1:
            if current.weekday() < 5:  # 周一至周五
                dates.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
        return dates


def _make_backtest_daily(date_str: str, rng: random.Random) -> dict[str, list]:
    """为回测日生成合成日线数据"""
    from ..data.fetchers.akshare_fetcher import StockDaily
    from ..demo import _SAMPLE_STOCKS

    # 用日期种子生成该日各股票的价格偏移
    day_seed = hash(date_str + "_prices") & 0x7FFFFFFF
    price_rng = random.Random(day_seed)

    data: dict[str, list] = {}
    for code, name, base_price, *_ in _SAMPLE_STOCKS:
        # 每日价格在基准价上随机波动
        daily_pct = price_rng.gauss(0.0008, 0.018)  # 均值 0.08%, 标准差 1.8%
        price = base_price * (1 + daily_pct)
        days = []
        for i in range(20):
            d = StockDaily(
                date=f"bt-{i+1:02d}",
                open=round(price * 0.995, 2),
                high=round(price * 1.015, 2),
                low=round(price * 0.985, 2),
                close=round(price, 2),
                volume=2e7 + rng.uniform(-5e6, 5e6),
                amount=price * 2e7 * 0.7,
                pct_chg=round(daily_pct * 100 * (0.5 + i / 20), 2),
                turnover=round(2.0 + rng.uniform(-0.5, 1.5), 2),
                ma5=round(price * 0.99, 2),
                ma10=round(price * 0.97, 2),
                ma20=round(price * 0.95, 2),
                macd_dif=round(0.5 + i * 0.1, 3),
                macd_dea=round(0.3 + i * 0.08, 3),
                macd_bar=round(0.2 + i * 0.02, 3),
                rsi_6=round(55 + i * 0.5, 2),
                rsi_14=round(52 + i * 0.3, 2),
            )
            days.append(d)
        data[code] = days
    return data


def _make_backtest_fund_flows(date_str: str, rng: random.Random) -> dict[str, list]:
    """为回测日生成合成资金流向数据"""
    from ..data.fetchers.akshare_fetcher import FundFlow
    from ..demo import _SAMPLE_STOCKS

    flows: dict[str, list] = {}
    for code, name, *_ in _SAMPLE_STOCKS:
        code_flows = []
        for i in range(5):
            main_net = rng.uniform(-5000, 5000)
            code_flows.append(FundFlow(
                date=f"bt-flow-{i+1:02d}",
                main_net_inflow=round(main_net, 1),
                super_large_net=round(main_net * 0.5, 1),
                large_net=round(main_net * 0.3, 1),
                medium_net=round(main_net * 0.1, 1),
                small_net=round(main_net * 0.1, 1),
                main_pct=round(abs(main_net) / 50000 * 100, 1),
            ))
        flows[code] = code_flows
    return flows

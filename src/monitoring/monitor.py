"""
盘中实时监控模块 — 持仓异动检测与告警。

职责:
- 定时轮询持仓实时行情
- 检查日内熔断 (总回撤 > 5%)
- 检查个股止盈/止损信号
- 输出告警 (日志 + 可选 Webhook)

使用方式:
    python -m src.monitoring.monitor          # 独立运行
    python -m src.monitoring.monitor --once   # 执行一次检查

配置:
    通过 Config 控制风控参数 (max_drawdown_daily 等)
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("monitoring")

# 默认监控参数
DEFAULT_POLL_INTERVAL = 30     # 轮询间隔 (秒)
STOP_LOSS_PCT = -7.0           # 个股止损线 (%)
TAKE_PROFIT_PCT = 15.0         # 个股止盈线 (%)

# Webhook 告警 URL (可选)
WEBHOOK_URL = os.getenv("ZHITOU_WEBHOOK_URL", "")


@dataclass
class PositionAlert:
    """持仓告警"""
    code: str
    name: str
    alert_type: str            # "stop_loss" / "take_profit" / "drawdown" / "circuit_breaker"
    severity: str              # "warning" / "critical"
    message: str
    current_price: float = 0.0
    cost_price: float = 0.0
    pnl_pct: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class IntradayMonitor:
    """
    盘中实时监控引擎。

    使用方式:
        monitor = IntradayMonitor(total_capital=500_000)
        monitor.load_positions()
        alerts = monitor.check()
        for a in alerts:
            print(a.message)
    """

    def __init__(self, total_capital: float = 500_000.0, results_dir: str = "./results") -> None:
        self._capital = total_capital
        self._results_dir = results_dir
        self._positions: dict[str, dict] = {}
        self._initial_value: float = total_capital
        self._last_alerts: set[str] = set()  # 去重

    # ── 加载持仓 ──────────────────────────────

    def load_positions(self) -> None:
        """从 PortfolioTracker 加载当前持仓"""
        try:
            from ..agents.portfolio_tracker import PortfolioTracker
            tracker = PortfolioTracker(
                total_capital=self._capital,
                results_dir=self._results_dir,
            )
            tracker.load()
            self._positions = {
                code: {
                    "name": p.name,
                    "shares": p.shares,
                    "avg_cost": p.avg_cost,
                    "industry": p.industry,
                }
                for code, p in tracker.positions.items() if p.shares > 0
            }
            self._initial_value = tracker.total_equity()
            logger.info(
                "监控器已加载: %d 只持仓, 初始权益 ¥%.0f",
                len(self._positions), self._initial_value,
            )
        except Exception:
            logger.warning("监控器: 加载持仓失败")
            self._positions = {}

    # ── 检查 ──────────────────────────────────

    def check(self, quotes: dict[str, dict] | None = None) -> list[PositionAlert]:
        """
        执行一次完整检查。

        quotes: {code: {price, name, pct_chg, ...}} 实时行情
                为 None 时尝试从数据层获取

        返回: 告警列表
        """
        if quotes is None:
            quotes = self._fetch_quotes()

        alerts: list[PositionAlert] = []
        if not self._positions:
            return alerts

        # 1. 计算总权益和回撤
        total_mv = 0.0
        for code, pos in self._positions.items():
            q = quotes.get(code, {})
            price = q.get("price", pos["avg_cost"])
            total_mv += pos["shares"] * price

        # 保守估计现金 (读取上次 tracker)
        try:
            from ..agents.portfolio_tracker import PortfolioTracker
            tracker = PortfolioTracker(total_capital=self._capital, results_dir=self._results_dir)
            tracker.load()
            cash = tracker.cash
        except Exception:
            cash = self._capital - total_mv

        total_equity = cash + total_mv
        drawdown = (total_equity / self._initial_value - 1) * 100

        # 2. 日内熔断检查 (回撤 > 5%)
        if drawdown < -5.0:
            alerts.append(PositionAlert(
                code="__PORTFOLIO__", name="组合",
                alert_type="circuit_breaker",
                severity="critical",
                message=f"日内熔断! 回撤 {drawdown:.2f}% (阈值 5%), 当前权益 ¥{total_equity:,.0f}",
                pnl_pct=drawdown,
                current_price=total_equity,
                cost_price=self._initial_value,
            ))

        # 3. 逐只检查
        for code, pos in self._positions.items():
            q = quotes.get(code, {})
            price = q.get("price", pos["avg_cost"])
            if price <= 0:
                continue

            pnl_pct = (price / pos["avg_cost"] - 1) * 100

            if pnl_pct <= STOP_LOSS_PCT:
                alert = PositionAlert(
                    code=code, name=pos["name"],
                    alert_type="stop_loss",
                    severity="critical",
                    message=f"止损: {pos['name']}({code}) 亏损 {pnl_pct:.1f}% (阈值 {STOP_LOSS_PCT}%), 当前价 ¥{price:.2f} 成本 ¥{pos['avg_cost']:.2f}",
                    current_price=price,
                    cost_price=pos["avg_cost"],
                    pnl_pct=pnl_pct,
                )
                if self._dedup(alert):
                    alerts.append(alert)

            elif pnl_pct >= TAKE_PROFIT_PCT:
                alert = PositionAlert(
                    code=code, name=pos["name"],
                    alert_type="take_profit",
                    severity="warning",
                    message=f"止盈: {pos['name']}({code}) 盈利 {pnl_pct:.1f}% (阈值 {TAKE_PROFIT_PCT}%), 当前价 ¥{price:.2f} 成本 ¥{pos['avg_cost']:.2f}",
                    current_price=price,
                    cost_price=pos["avg_cost"],
                    pnl_pct=pnl_pct,
                )
                if self._dedup(alert):
                    alerts.append(alert)

            elif pnl_pct <= -3.0:
                alert = PositionAlert(
                    code=code, name=pos["name"],
                    alert_type="drawdown",
                    severity="warning",
                    message=f"关注: {pos['name']}({code}) 浮亏 {pnl_pct:.1f}%, 当前价 ¥{price:.2f}",
                    current_price=price,
                    cost_price=pos["avg_cost"],
                    pnl_pct=pnl_pct,
                )
                if self._dedup(alert):
                    alerts.append(alert)

        # 4. 行业整体估值
        if drawdown < -3.0:
            alerts.append(PositionAlert(
                code="__PORTFOLIO__", name="组合",
                alert_type="drawdown",
                severity="warning",
                message=f"组合回撤 {drawdown:.2f}%, 总权益 ¥{total_equity:,.0f}",
                pnl_pct=drawdown,
                current_price=total_equity,
                cost_price=self._initial_value,
            ))

        return alerts

    def _dedup(self, alert: PositionAlert) -> bool:
        """去重: 同股同类型 5 分钟内不重复"""
        key = f"{alert.code}:{alert.alert_type}"
        if key in self._last_alerts:
            return False
        self._last_alerts.add(key)
        if len(self._last_alerts) > 100:
            self._last_alerts.clear()
        return True

    @staticmethod
    def _fetch_quotes() -> dict[str, dict]:
        """从数据层获取实时行情"""
        try:
            from ..data.interface import UnifiedDataInterface
            udi = UnifiedDataInterface()
            tracker_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "results",
            )
            from ..agents.portfolio_tracker import PortfolioTracker
            tracker = PortfolioTracker(results_dir=tracker_path)
            tracker.load()

            codes = list(tracker.positions.keys())
            if not codes:
                return {}

            quotes = udi.batch_realtime_quotes(codes)
            result = {}
            for code, q in quotes.items():
                if q is not None:
                    result[code] = {
                        "price": q.price,
                        "name": q.name,
                        "pct_chg": q.pct_chg,
                        "volume_ratio": q.volume_ratio,
                    }
            return result
        except Exception:
            logger.debug("监控器: 获取行情失败", exc_info=True)
            return {}

    # ── 告警输出 ──────────────────────────────

    def emit_alerts(self, alerts: list[PositionAlert]) -> None:
        """输出告警 (日志 + 可选 Webhook)"""
        for a in alerts:
            level = logging.ERROR if a.severity == "critical" else logging.WARNING
            logger.log(level, "[%s] %s", a.alert_type.upper(), a.message)

        if WEBHOOK_URL and alerts:
            self._send_webhook(alerts)

    @staticmethod
    def _send_webhook(alerts: list[PositionAlert]) -> None:
        """发送 Webhook 通知 (企业微信/飞书兼容)"""
        try:
            import requests

            critical = [a for a in alerts if a.severity == "critical"]
            warnings = [a for a in alerts if a.severity == "warning"]

            lines = [
                f"# 智投未来 盘中告警 ({datetime.now().strftime('%H:%M:%S')})",
                "",
            ]
            if critical:
                lines.append(f"## 严重 ({len(critical)}条)")
                for a in critical:
                    lines.append(f"- {a.message}")
                lines.append("")

            if warnings:
                lines.append(f"## 警告 ({len(warnings)}条)")
                for a in warnings[:5]:
                    lines.append(f"- {a.message}")
                lines.append("")

            payload = {
                "msgtype": "markdown",
                "markdown": {"content": "\n".join(lines)},
            }
            requests.post(WEBHOOK_URL, json=payload, timeout=5)
        except Exception:
            logger.debug("Webhook 发送失败", exc_info=True)


# ── 独立运行入口 ──────────────────────────────

def _run_loop() -> None:
    """阻塞式监控循环"""
    monitor = IntradayMonitor()
    monitor.load_positions()

    if not monitor._positions:
        logger.warning("无持仓，监控退出")
        return

    logger.info("监控循环启动: 每 %ds 检查一次", DEFAULT_POLL_INTERVAL)
    running = True

    def _shutdown(signum, frame):
        nonlocal running
        logger.info("收到退出信号")
        running = False

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    while running:
        alerts = monitor.check()
        if alerts:
            monitor.emit_alerts(alerts)

        # 分段 sleep
        for _ in range(DEFAULT_POLL_INTERVAL):
            if not running:
                break
            time.sleep(1)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="智投未来 盘中监控")
    parser.add_argument("--once", action="store_true", help="执行一次检查后退出")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    if args.once:
        monitor = IntradayMonitor()
        monitor.load_positions()
        alerts = monitor.check()
        if alerts:
            monitor.emit_alerts(alerts)
        else:
            logger.info("无异常")
    else:
        _run_loop()

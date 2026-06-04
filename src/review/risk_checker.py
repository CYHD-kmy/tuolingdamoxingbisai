"""
风控红线检查 (P0) — 借鉴金策智算门下省设计 + 三级仓位分层框架。

纯确定性计算，不消耗 LLM Token。对每个持仓逐条检查风控红线。

红线体系 (10 条):
    个股级:
        R1: 分层单票仓位 — 核心 ≤ 40% / 卫星 ≤ 14%
        R5: 长期持仓止损 — 持有超 holding_clear_days 天且浮亏 > 3%
        R6: 移动止损触发 — 现价跌破 ATR 移动止损价
        R7: 超长持有预警 — 持有 > 20 天
    组合级:
        R2: 环境自适应总仓位 — 牛市 ≤ 75% / 震荡 ≤ 34% / 熊市 = 0%
        R3: 行业集中度 — 单行业 ≤ 40%
        R4: 现金安全线 — 现金 ≥ 25%
        R8: 开盘大跌过滤 — 单日开盘跌幅 > 5% 则剔除 (默认不启用，需传入开盘数据)
        R9: 全面下跌熔断 — 全市场 > 3000 只下跌 → 强制空仓
        R10: 仓位数量合规 — 核心 ≤ 2 只 / 卫星 ≤ 4 只

使用方式:
    from src.review.risk_checker import RiskChecker, classify_tier
    checker = RiskChecker(tracker, market_regime="neutral")
    # 可选: 传入评分用于分层
    result = checker.check(scores={code: composite_score}, market_data=snapshots)
"""

from __future__ import annotations

import logging
from typing import Any

from ..utils.config import get_config

logger = logging.getLogger(__name__)

# 规则配置
_RULES = [
    {"id": "R1", "name": "分层单票仓位"},
    {"id": "R2", "name": "环境自适应总仓位"},
    {"id": "R3", "name": "行业集中度"},
    {"id": "R4", "name": "现金安全线"},
    {"id": "R5", "name": "长期持仓止损"},
    {"id": "R6", "name": "移动止损触发"},
    {"id": "R7", "name": "超长持有预警"},
    {"id": "R8", "name": "开盘大跌过滤"},
    {"id": "R9", "name": "全面下跌熔断"},
    {"id": "R10", "name": "仓位数量合规"},
]


def classify_tier(
    code: str,
    composite_score: float = 50.0,
    pe: float = 0.0,
    market_cap: float = 0.0,
) -> str:
    """
    将股票分类为核心仓(core) 或 卫星仓(satellite)。

    核心条件 (全部满足):
        - 综合得分 ≥ core_score_threshold (默认 75)
        - PE ≤ core_pe_max (默认 30), PE <= 0 视为不可分类
        - 市值 ≥ core_market_cap_min (默认 100 亿)

    不满足任一条件 → 卫星仓
    """
    cfg = get_config()
    if (
        composite_score >= cfg.core_score_threshold
        and 0 < pe <= cfg.core_pe_max
        and market_cap >= cfg.core_market_cap_min
    ):
        return "core"
    return "satellite"


class RiskChecker:
    """
    风控红线检查器 — 三级仓位分层 + 市场环境自适应。

    门下省核心设计理念: 策略信号和风控审核彻底分离。
    本检查器在每日复盘时运行，对现有持仓做合规审计。
    """

    def __init__(self, tracker, market_regime: str = "neutral"):
        """
        tracker: PortfolioTracker 实例
        market_regime: 市场环境 "bull" / "neutral" / "bear"
        """
        self._tracker = tracker
        self._regime = market_regime
        self._cfg = get_config()

    def check(
        self,
        scores: dict[str, float] | None = None,
        market_data: list | None = None,
        stock_infos: dict[str, dict] | None = None,
    ) -> dict:
        """
        对全部持仓执行风控红线检查。

        参数:
            scores: {code: 综合得分} 用于分层分类 (可选, 无则按持仓大小推测)
            market_data: MarketSnapshot 列表 (可选, 用于 R8/R9)
            stock_infos: {code: {pe, total_mv, ...}} 用于分层分类 (可选)

        返回:
            {
                "checks": {code: [RiskLine, ...]},
                "summary": "整体风控状态描述",
                "violation_count": int,
                "warning_count": int,
            }
        """
        from .engine import RiskLine

        all_checks: dict[str, list[RiskLine]] = {}
        violation_count = 0
        warning_count = 0
        equity = self._tracker.total_equity()

        if equity <= 0:
            return {
                "checks": {}, "summary": "无法计算: 总权益为 0",
                "violation_count": 0, "warning_count": 0,
            }

        positions = self._tracker.positions
        market_value = self._tracker.total_market_value()
        cash_pct = self._tracker.cash / equity if equity > 0 else 0

        # ── 分层分类 ──
        tiers = self._classify_positions(positions, scores or {}, stock_infos or {})
        core_stocks = [c for c, t in tiers.items() if t == "core"]
        satellite_stocks = [c for c, t in tiers.items() if t == "satellite"]

        # ── 逐只持仓检查 ──
        for code, pos in positions.items():
            lines: list[RiskLine] = []
            pos_pct = pos.ratio_of_equity(equity)
            tier = tiers.get(code, "satellite")
            single_limit = (
                self._cfg.core_single_pct if tier == "core"
                else self._cfg.satellite_single_pct
            )

            # R1: 分层单票仓位
            if pos_pct > single_limit:
                status_r1 = "violation" if pos_pct > single_limit * 1.1 else "warn"
                if status_r1 == "violation":
                    violation_count += 1
                else:
                    warning_count += 1
                msg_r1 = f"[{tier}] 仓位 {pos_pct:.1%} 超过分层上限 {single_limit:.0%}"
            elif pos_pct > single_limit * 0.8:
                status_r1 = "warn"
                warning_count += 1
                msg_r1 = f"[{tier}] 仓位 {pos_pct:.1%} 接近分层上限 {single_limit:.0%}"
            else:
                status_r1 = "pass"
                msg_r1 = f"[{tier}] 仓位 {pos_pct:.1%} 正常"
            lines.append(RiskLine(
                rule_id="R1", rule_name="分层单票仓位",
                value=round(pos_pct * 100, 1), threshold=single_limit * 100,
                status=status_r1, message=msg_r1,
            ))

            # R5: 长期持仓止损
            clear_days = self._cfg.holding_clear_days
            days = pos.holding_days
            pnl = pos.pnl_pct
            if days > clear_days and pnl < -3.0:
                status_r5 = "violation"
                violation_count += 1
                msg_r5 = f"持有 {days} 天，浮亏 {pnl:+.1f}%，超过止损阈值"
            elif days > clear_days and pnl < 0:
                status_r5 = "warn"
                warning_count += 1
                msg_r5 = f"持有 {days} 天，浮亏 {pnl:+.1f}%，注意止损"
            else:
                status_r5 = "pass"
                msg_r5 = f"持有 {days} 天，盈亏 {pnl:+.1f}%"
            lines.append(RiskLine(
                rule_id="R5", rule_name="长期持仓止损",
                value=round(pnl, 1), threshold=-3.0,
                status=status_r5, message=msg_r5,
            ))

            # R6: 移动止损
            trailing = pos.trailing_stop
            if trailing > 0 and pos.last_price > 0:
                if pos.last_price <= trailing:
                    status_r6 = "violation"
                    violation_count += 1
                    msg_r6 = f"现价 ¥{pos.last_price:.2f} ≤ 止损价 ¥{trailing:.2f}"
                elif pos.last_price <= trailing * 1.03:
                    status_r6 = "warn"
                    warning_count += 1
                    msg_r6 = f"现价 ¥{pos.last_price:.2f} 接近止损价 ¥{trailing:.2f}"
                else:
                    status_r6 = "pass"
                    msg_r6 = f"现价 ¥{pos.last_price:.2f} 安全 (止损 ¥{trailing:.2f})"
            else:
                status_r6 = "pass"
                msg_r6 = "未设置止损价" if trailing == 0 else ""
            lines.append(RiskLine(
                rule_id="R6", rule_name="移动止损触发",
                value=pos.last_price, threshold=trailing,
                status=status_r6, message=msg_r6,
            ))

            # R7: 超长持有
            max_days = 20
            if days > max_days * 2:
                status_r7 = "violation"
                violation_count += 1
                msg_r7 = f"持有 {days} 天，远超 {max_days} 天预警线"
            elif days > max_days:
                status_r7 = "warn"
                warning_count += 1
                msg_r7 = f"持有 {days} 天，超过 {max_days} 天预警线"
            else:
                status_r7 = "pass"
                msg_r7 = f"持有 {days} 天"
            lines.append(RiskLine(
                rule_id="R7", rule_name="超长持有预警",
                value=days, threshold=max_days,
                status=status_r7, message=msg_r7,
            ))

            all_checks[code] = lines

        # ── 组合级红线 ──
        combo_checks: list[RiskLine] = []

        # R2: 环境自适应总仓位
        max_total = {
            "bull": self._cfg.max_total_position_bull,
            "neutral": self._cfg.max_total_position_neutral,
            "bear": self._cfg.max_total_position_bear,
        }.get(self._regime, self._cfg.max_total_position_neutral)

        total_pct = market_value / equity if equity > 0 else 0
        if self._regime == "bear" and total_pct > 0:
            combo_checks.append(RiskLine(
                rule_id="R2", rule_name="环境自适应总仓位",
                value=round(total_pct * 100, 1), threshold=0,
                status="violation",
                message=f"熊市应空仓，当前仓位 {total_pct:.1%}",
            ))
            violation_count += 1
        elif total_pct > max_total:
            combo_checks.append(RiskLine(
                rule_id="R2", rule_name="环境自适应总仓位",
                value=round(total_pct * 100, 1), threshold=max_total * 100,
                status="violation",
                message=f"[{self._regime}] 总仓位 {total_pct:.1%} 超过上限 {max_total:.0%}",
            ))
            violation_count += 1
        elif total_pct > max_total * 0.85:
            combo_checks.append(RiskLine(
                rule_id="R2", rule_name="环境自适应总仓位",
                value=round(total_pct * 100, 1), threshold=max_total * 100,
                status="warn",
                message=f"[{self._regime}] 总仓位 {total_pct:.1%} 接近上限 {max_total:.0%}",
            ))
            warning_count += 1
        else:
            combo_checks.append(RiskLine(
                rule_id="R2", rule_name="环境自适应总仓位",
                value=round(total_pct * 100, 1), threshold=max_total * 100,
                status="pass",
                message=f"[{self._regime}] 总仓位 {total_pct:.1%}",
            ))

        # R3: 行业集中度
        max_industry = self._cfg.max_industry_exposure
        industry_exposure = self._tracker.industry_exposure()
        max_ind_pct = 0.0
        max_ind_name = ""
        for ind, pct in industry_exposure.items():
            if pct > max_ind_pct:
                max_ind_pct = pct
                max_ind_name = ind
        industry_decimal = max_ind_pct / 100.0
        if industry_decimal > max_industry:
            combo_checks.append(RiskLine(
                rule_id="R3", rule_name="行业集中度",
                value=round(max_ind_pct, 1), threshold=max_industry * 100,
                status="violation",
                message=f"行业'{max_ind_name}'占比 {max_ind_pct:.1f}% 超过上限 {max_industry:.0%}",
            ))
            violation_count += 1
        else:
            combo_checks.append(RiskLine(
                rule_id="R3", rule_name="行业集中度",
                value=round(max_ind_pct, 1), threshold=max_industry * 100,
                status="pass",
                message=f"最大行业'{max_ind_name}'占比 {max_ind_pct:.1f}%",
            ))

        # R4: 现金安全线 (≥ 25%)
        min_cash = self._cfg.min_cash_reserve
        if cash_pct < min_cash:
            status_r4 = "violation"
            violation_count += 1
            msg_r4 = f"现金仅 {cash_pct:.1%}，低于安全线 {min_cash:.0%}"
        elif cash_pct < min_cash * 1.2:
            status_r4 = "warn"
            warning_count += 1
            msg_r4 = f"现金 {cash_pct:.1%}，接近安全线 {min_cash:.0%}"
        else:
            status_r4 = "pass"
            msg_r4 = f"现金 {cash_pct:.1%} 充足"
        combo_checks.append(RiskLine(
            rule_id="R4", rule_name="现金安全线",
            value=round(cash_pct * 100, 1), threshold=min_cash * 100,
            status=status_r4, message=msg_r4,
        ))

        # R8: 开盘大跌过滤 (需要 market_data)
        if market_data:
            open_drop_count = 0
            open_drop_threshold = self._cfg.open_loss_filter_pct
            for snap in market_data:
                pct = getattr(snap, "pct_chg", 0.0)
                if pct <= open_drop_threshold:
                    open_drop_count += 1
                    # 检查持仓中是否有该股票
                    code = getattr(snap, "code", "")
                    if code in positions:
                        combo_checks.append(RiskLine(
                            rule_id="R8", rule_name="开盘大跌过滤",
                            value=pct, threshold=open_drop_threshold,
                            status="violation",
                            message=f"{code} 跌幅 {pct:+.1f}% 触发开盘大跌过滤 (≤ {open_drop_threshold}%)",
                        ))
                        violation_count += 1
            if open_drop_count == 0:
                combo_checks.append(RiskLine(
                    rule_id="R8", rule_name="开盘大跌过滤",
                    value=0, threshold=open_drop_threshold,
                    status="pass",
                    message="无触发个股",
                ))

        # R9: 全面下跌熔断
        if market_data:
            falling_count = sum(
                1 for s in market_data if getattr(s, "pct_chg", 0) < 0
            )
            decline_threshold = self._cfg.broad_decline_threshold
            if falling_count >= decline_threshold:
                combo_checks.append(RiskLine(
                    rule_id="R9", rule_name="全面下跌熔断",
                    value=falling_count, threshold=decline_threshold,
                    status="violation",
                    message=f"全市场 {falling_count} 只下跌 (≥ {decline_threshold})，应强制空仓",
                ))
                violation_count += 1
            else:
                combo_checks.append(RiskLine(
                    rule_id="R9", rule_name="全面下跌熔断",
                    value=falling_count, threshold=decline_threshold,
                    status="pass",
                    message=f"下跌 {falling_count} 只 (阈值 {decline_threshold})",
                ))

        # R10: 仓位数量合规
        if len(core_stocks) > 2:
            combo_checks.append(RiskLine(
                rule_id="R10", rule_name="仓位数量合规",
                value=len(core_stocks), threshold=2,
                status="violation",
                message=f"核心仓 {len(core_stocks)} 只 (上限 2 只): {core_stocks}",
            ))
            violation_count += 1
        else:
            combo_checks.append(RiskLine(
                rule_id="R10", rule_name="仓位数量合规",
                value=len(core_stocks), threshold=2,
                status="pass",
                message=f"核心 {len(core_stocks)} 只 / 卫星 {len(satellite_stocks)} 只",
            ))
        if len(satellite_stocks) > 4:
            combo_checks.append(RiskLine(
                rule_id="R10", rule_name="仓位数量合规",
                value=len(satellite_stocks), threshold=4,
                status="warn",
                message=f"卫星仓 {len(satellite_stocks)} 只 (建议 ≤ 4 只)",
            ))
            warning_count += 1

        if combo_checks:
            all_checks["_portfolio"] = combo_checks

        # ── 汇总 ──
        if violation_count == 0 and warning_count == 0:
            summary = f"风控状态 [{self._regime}]: 全部通过"
        elif violation_count > 0:
            summary = f"风控状态 [{self._regime}]: {violation_count} 条违规 — 建议优先处理"
        else:
            summary = f"风控状态 [{self._regime}]: {warning_count} 条预警 — 需关注"

        return {
            "checks": all_checks,
            "summary": summary,
            "violation_count": violation_count,
            "warning_count": warning_count,
        }

    def _classify_positions(
        self,
        positions: dict,
        scores: dict[str, float],
        stock_infos: dict[str, dict],
    ) -> dict[str, str]:
        """对全部持仓做分层分类。"""
        tiers: dict[str, str] = {}
        for code in positions:
            score = scores.get(code, 50.0)
            info = stock_infos.get(code, {})
            pe = info.get("pe", 0.0)
            mv = info.get("total_mv", 0.0)
            tiers[code] = classify_tier(code, score, pe, mv)
        return tiers

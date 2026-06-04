"""
持仓合理性评分 (P1) — 复用 ScreeningScorer 重新打分 + LLM 辅助研判。

对当前持仓的每只股票:
    1. 获取最新行情/资金流向/财务数据
    2. 用 ScoringScorer 重新计算因子得分
    3. 对比历史得分 (从 trace 读取建仓时得分)
    4. LLM 综合研判: 持有/减仓/清仓 + 理由

使用 quick LLM (低成本), 仅做判断不做决策。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class PositionScorer:
    """
    持仓合理性评分器 — 定期重新验证持仓是否仍具备持有价值。

    使用方式:
        scorer = PositionScorer(data_interface, tracker, llm)
        result = scorer.score_all()
    """

    def __init__(self, data_interface, tracker, llm=None):
        self._data = data_interface
        self._tracker = tracker
        self._llm = llm

    def score_all(self) -> dict:
        """
        对全部持仓执行重新评分。

        返回:
            {
                "positions": {code: PositionReview, ...},
                "summary": "评分概述",
            }
        """
        from .engine import PositionReview
        from ..screening.scorer import ScreeningScorer

        positions = self._tracker.positions
        if not positions:
            return {"positions": {}, "summary": "无持仓"}

        codes = list(positions.keys())
        scorer = ScreeningScorer()

        # 1. 获取最新数据
        daily_data = {}
        fund_flows = {}
        financials = {}
        try:
            daily_data = self._data.batch_daily_data(codes, days=30, max_workers=4)
        except Exception:
            logger.warning("批量获取日线数据失败")

        try:
            fund_flows = self._data.batch_fund_flows(codes, days=5, max_workers=6)
        except Exception:
            logger.debug("批量获取资金流向失败")

        try:
            financials = self._data.batch_financials(codes, max_workers=4)
        except Exception:
            logger.debug("批量获取财务数据失败")

        # 2. 构建快照 (从持仓信息)
        snapshots: dict = {}
        for code, pos in positions.items():
            snap = type("Snapshot", (), {
                "code": code,
                "name": pos.name,
                "price": pos.last_price,
                "pct_chg": 0.0,
                "volume_ratio": 1.0,
                "turnover": 0.0,
                "amount": pos.market_value,
                "pe": 0.0,
                "total_mv": 0.0,
            })()
            snapshots[code] = snap

        # 3. 重新打分 (使用所有可用数据)
        try:
            scores = scorer.score_all(
                codes=codes,
                snapshots=snapshots,
                daily_data=daily_data,
                fund_flows=fund_flows,
                financials=financials,
            )
            score_map = {s.code: s for s in scores}
        except Exception:
            logger.exception("重新评分失败")
            score_map = {}

        # 4. 读取历史得分 (从最近的 trace 文件中提取首次出现时的得分)
        hist_scores = self._load_historical_scores(codes)

        # 5. 构建结果
        results: dict[str, PositionReview] = {}
        equity = self._tracker.total_equity()

        recommend_total = 0
        reduce_count = 0
        clear_count = 0

        for code, pos in positions.items():
            fs = score_map.get(code)
            cur_score = round(fs.composite, 1) if fs else 50.0
            factor_detail = dict(fs.scores) if fs else {}

            prev_score = hist_scores.get(code, cur_score)
            score_change = round(cur_score - prev_score, 1)

            pr = PositionReview(
                code=code,
                name=pos.name,
                current_score=cur_score,
                score_change=score_change,
                factor_detail=factor_detail,
                pnl_pct=round(pos.pnl_pct, 1),
                holding_days=pos.holding_days,
                position_pct=round(pos.ratio_of_equity(equity) * 100, 1),
            )

            # 确定性研判 (LLM 不可用时的降级)
            if cur_score >= 70:
                pr.recommendation = "hold"
                pr.reasoning = f"综合得分 {cur_score} 分 (优秀)，因子强度维持良好"
                recommend_total += 1
            elif cur_score >= 50:
                pr.recommendation = "hold"
                pr.reasoning = f"综合得分 {cur_score} 分 (中性)，暂无明显恶化"
                recommend_total += 1
            elif cur_score >= 30:
                pr.recommendation = "reduce"
                pr.reasoning = f"综合得分 {cur_score} 分 (偏低)，因子弱化，建议减仓观察"
                reduce_count += 1
            else:
                pr.recommendation = "clear"
                pr.reasoning = f"综合得分 {cur_score} 分 (差)，多因子全面恶化，建议清仓"
                clear_count += 1

            if score_change < -10:
                pr.reasoning += f"；得分较建仓时下降 {score_change:.0f} 分"
            elif score_change > 10:
                pr.reasoning += f"；得分较建仓时上升 +{score_change:.0f} 分"

            # 浮亏加成
            if pos.pnl_pct < -5:
                pr.reasoning += f"；当前浮亏 {pos.pnl_pct:+.1f}%"

            results[code] = pr

        # 6. LLM 增强研判 (如果 LLM 可用且有需要关注的持仓)
        if self._llm:
            try:
                self._llm_enhance(results, positions, daily_data)
            except Exception:
                logger.debug("LLM 增强研判失败", exc_info=True)

        # 7. 汇总
        parts = [f"共 {len(results)} 只持仓"]
        if recommend_total > 0:
            parts.append(f"{recommend_total} 只建议持有")
        if reduce_count > 0:
            parts.append(f"{reduce_count} 只建议减仓")
        if clear_count > 0:
            parts.append(f"{clear_count} 只建议清仓")
        summary = "，".join(parts)

        return {"positions": results, "summary": summary}

    def _load_historical_scores(self, codes: list[str]) -> dict[str, float]:
        """从历史 trace 文件中提取每只股票首次出现时的得分。"""
        hist: dict[str, float] = {}
        results_dir = getattr(self._tracker, "_results_dir", "./results")

        # 收集所有可用的 trace 文件
        try:
            files = sorted([
                f for f in os.listdir(results_dir)
                if f.startswith("trace_") and f.endswith(".json")
            ])
        except OSError:
            return hist

        for code in codes:
            for fname in files:
                try:
                    fpath = os.path.join(results_dir, fname)
                    with open(fpath, "r", encoding="utf-8") as f:
                        trace = json.load(f)
                    candidates = trace.get("screening", {}).get("candidates", [])
                    for c in candidates:
                        if c.get("code") == code:
                            hist[code] = c.get("score", c.get("composite", 50))
                            break
                    if code in hist:
                        break
                except (json.JSONDecodeError, OSError, KeyError):
                    continue

        return hist

    def _llm_enhance(
        self,
        results: dict,
        positions,
        daily_data: dict,
    ) -> None:
        """用 LLM 增强持仓研判 (对得分偏低或有争议的持仓做深度分析)。"""
        # 只对需要关注的持仓做 LLM 分析
        needs_attention = {
            code: pr for code, pr in results.items()
            if pr.recommendation in ("reduce", "clear") or pr.score_change < -5
        }
        if not needs_attention:
            return

        # 构建 prompt
        prompt = _build_position_review_prompt(needs_attention, positions, daily_data)

        try:
            resp = self._llm.chat([
                {"role": "system", "content": _POSITION_REVIEW_SYSTEM},
                {"role": "user", "content": prompt},
            ])
            content = resp.content if hasattr(resp, "content") else str(resp)
            # 解析 LLM 输出
            parsed = _parse_llm_recommendations(content, results)
            for code, (rec, conf, reason) in parsed.items():
                if code in results:
                    results[code].recommendation = rec
                    results[code].confidence = conf
                    results[code].reasoning = reason
        except Exception:
            logger.debug("LLM 研判解析失败", exc_info=True)


_POSITION_REVIEW_SYSTEM = """你是一个 A 股持仓复盘专家。你的任务是审查现有持仓，判断每只股票是否仍然值得持有。

审查要点:
1. 得分变化趋势: 如果得分持续下降，即使绝对值不低，也需警惕
2. 浮亏程度: 浮亏超过 5% 且得分低于 50 应重点考虑减仓
3. 持有时间: 持有超过 10 天仍未盈利，需评估是否继续等待
4. 因子是否消失: 原来支撑买入的逻辑是否仍然成立

输出格式 (严格 JSON):
{
    "reviews": {
        "CODE": {
            "recommendation": "hold" | "reduce" | "clear",
            "confidence": 0.0-1.0,
            "reasoning": "一句话理由"
        }
    }
}"""


def _build_position_review_prompt(
    needs_attention: dict,
    positions,
    daily_data: dict,
) -> str:
    """构建持仓复盘 prompt"""
    parts = ["## 需要关注的持仓\n"]
    for code, pr in needs_attention.items():
        pos = positions.get(code)
        parts.append(
            f"### {pr.name} ({code})\n"
            f"- 综合得分: {pr.current_score} (变化: {pr.score_change:+.0f})\n"
            f"- 浮动盈亏: {pr.pnl_pct:+.1f}%\n"
            f"- 持有天数: {pr.holding_days}\n"
            f"- 仓位占比: {pr.position_pct:.1f}%\n"
        )
        if pr.factor_detail:
            parts.append(f"- 因子得分: {pr.factor_detail}\n")
        # 价格信息
        if pos and pos.last_price > 0:
            parts.append(f"- 成本价: ¥{pos.avg_cost:.2f} / 现价: ¥{pos.last_price:.2f}\n")
    parts.append("\n请对以上每只持仓给出: hold/reduce/clear 建议")
    return "\n".join(parts)


def _parse_llm_recommendations(content: str, results: dict) -> dict[str, tuple[str, float, str]]:
    """解析 LLM 输出的持仓建议"""
    parsed: dict = {}
    try:
        # 提取 JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        data = json.loads(content)
        reviews = data.get("reviews", data) if isinstance(data, dict) else {}

        for code, review in reviews.items():
            if not isinstance(review, dict):
                continue
            rec = review.get("recommendation", "hold")
            if rec not in ("hold", "reduce", "clear"):
                rec = "hold"
            conf = float(review.get("confidence", 0.5))
            reason = review.get("reasoning", "")
            parsed[code] = (rec, conf, reason)
    except (json.JSONDecodeError, ValueError, AttributeError):
        logger.debug("LLM 输出解析失败")
    return parsed

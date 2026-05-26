"""
辩论引擎 — 编排多头/空头研究员的多轮辩论。

流程 (每只股票):
  多头第一轮论点
    → 空头第一轮反驳
    → 多头第二轮回应 (可选)
    → 空头第二轮总结 (可选)
    → 输出完整辩论记录
"""

from __future__ import annotations

import logging

from .bull import BullResearcher
from .bear import BearResearcher
from ..base import AnalystReport
from ..models import DebateRound, DebateResult
from ...llm.client import LLMClient

logger = logging.getLogger(__name__)


class DebateEngine:
    """
    多轮辩论引擎。

    使用方式:
        engine = DebateEngine(quick_llm)
        result = engine.debate("600519", "贵州茅台", analyst_reports)
    """

    def __init__(self, quick_llm: LLMClient) -> None:
        self._bull = BullResearcher(quick_llm)
        self._bear = BearResearcher(quick_llm)

    def debate(
        self,
        code: str,
        name: str,
        reports: list[AnalystReport],
        max_rounds: int = 3,
    ) -> DebateResult:
        """
        执行多轮辩论。

        reports: 四份分析师报告
        max_rounds: 最大辩论轮数 (默认3轮)
        当检测到论点收敛时提前终止 (连续两轮相似度 > 阈值)。

        返回: 完整辩论记录
        """
        result = DebateResult(code=code, name=name)

        bull_argument = ""
        bear_argument = ""
        bull_rebuttal = ""
        prev_bear_response = ""
        prev_bull_rebuttal = ""

        for rnd in range(1, max_rounds + 1):
            logger.info("辩论: %s %s 第%d轮", code, name, rnd)

            # 多头论点
            if rnd == 1:
                bull_argument = self._bull.research(code, name, reports, "", rnd)
            else:
                bull_rebuttal = self._bull.research(code, name, reports, bear_argument, rnd)

            # 空头反驳 (优先针对最新回应)
            target = bull_rebuttal if bull_rebuttal else bull_argument
            bear_response = self._bear.research(code, name, reports, target, rnd)

            # 记录本轮
            if rnd == 1:
                bear_argument = bear_response
                result.rounds.append(DebateRound(
                    round_num=rnd,
                    bull_argument=bull_argument,
                    bear_argument=bear_argument,
                ))
                prev_bear_response = bear_response
            else:
                result.rounds.append(DebateRound(
                    round_num=rnd,
                    bull_argument=bull_rebuttal,
                    bear_argument=bear_response,
                ))

                # 收敛检测: 连续两轮论点高度重复 → 提前终止
                if rnd >= 2:
                    bull_sim = _text_similarity(prev_bull_rebuttal, bull_rebuttal)
                    bear_sim = _text_similarity(prev_bear_response, bear_response)
                    if bull_sim > 0.75 and bear_sim > 0.75:
                        logger.info(
                            "辩论收敛: %s %s 第%d轮后终止 (bull_sim=%.2f bear_sim=%.2f)",
                            code, name, rnd, bull_sim, bear_sim,
                        )
                        break

                # 推进论点: 当前轮的新观点成为下一轮的起始基线
                bull_argument = bull_rebuttal
                bear_argument = bear_response
                prev_bull_rebuttal = bull_rebuttal
                prev_bear_response = bear_response

        result.total_rounds = len(result.rounds)
        logger.info("辩论完成: %s %s 共%d轮", code, name, result.total_rounds)
        return result


def _text_similarity(a: str, b: str) -> float:
    """计算两段中文文本的字符级 bigram 重叠相似度 (0.0-1.0)。"""
    if not a or not b:
        return 0.0

    def bigrams(s: str) -> set[str]:
        s = s.replace("\n", "").replace(" ", "")
        if len(s) < 2:
            return {s}
        return {s[i:i + 2] for i in range(len(s) - 1)}

    bg_a = bigrams(a)
    bg_b = bigrams(b)
    if not bg_a or not bg_b:
        return 0.0
    intersection = bg_a & bg_b
    union = bg_a | bg_b
    return len(intersection) / len(union)


def debate_result_to_text(result: DebateResult) -> str:
    """将辩论结果转为文本，供研究主管阅读"""
    lines = [f"# 辩论记录: {result.name} ({result.code})", f"共 {result.total_rounds} 轮辩论", ""]

    for r in result.rounds:
        lines.append(f"## 第 {r.round_num} 轮")
        lines.append(f"### 多头论点:\n{r.bull_argument}\n")
        if r.bull_rebuttal:
            lines.append(f"### 多头回应:\n{r.bull_rebuttal}\n")
        if r.bear_argument:
            lines.append(f"### 空头反驳:\n{r.bear_argument}\n")
        if r.bear_summary:
            lines.append(f"### 空头总结:\n{r.bear_summary}\n")
        lines.append("---")

    return "\n".join(lines)

"""
风险平价优化器 — 基于协方差矩阵的权重分配。

支持三种方法:
- ERC (Equal Risk Contribution): 每个资产对组合风险的贡献相等
- Min Variance: 最小化组合方差
- Max Diversification: 最大化分散化比率

全部手动计算，无 numpy 依赖。

使用方式:
    optimizer = RiskParityOptimizer(method=OptimizationMethod.ERC)
    result = optimizer.optimize(codes, daily_data, position_limits)
    print(result.weights)  # {code: weight}
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_ITER = 100
TOLERANCE = 1e-6


class OptimizationMethod(enum.Enum):
    EQUAL_WEIGHT = "equal"
    ERC = "erc"
    MIN_VARIANCE = "min_var"
    MAX_DIVERSIFICATION = "max_div"


@dataclass
class OptimizationResult:
    """优化结果"""
    weights: dict[str, float] = field(default_factory=dict)
    method: str = "equal"
    expected_volatility: float = 0.0
    diversification_ratio: float = 0.0
    converged: bool = False


class RiskParityOptimizer:
    """
    风险平价 / 最小方差 / 最大分散化 优化器。

    使用方式:
        optimizer = RiskParityOptimizer(method=OptimizationMethod.ERC)
        result = optimizer.optimize(codes, daily_data, position_limits)
    """

    def __init__(
        self,
        method: OptimizationMethod = OptimizationMethod.ERC,
        max_iter: int = MAX_ITER,
        tolerance: float = TOLERANCE,
    ) -> None:
        self._method = method
        self._max_iter = max_iter
        self._tolerance = tolerance

    def optimize(
        self,
        codes: list[str],
        daily_data: dict[str, list],
        position_limits: dict | None = None,
    ) -> OptimizationResult:
        """
        优化权重分配。

        codes: 股票代码列表
        daily_data: {code: [StockDaily, ...]} 用于计算协方差
        position_limits: {code: PositionLimit} 风控上限

        返回: OptimizationResult
        """
        if len(codes) < 2:
            w = {codes[0]: 1.0} if codes else {}
            return OptimizationResult(weights=w, method=self._method.value, converged=True)

        # 构建收益率矩阵
        returns = self._build_return_matrix(codes, daily_data)
        if not returns or len(returns[0]) < 5:
            w = self._equal_weights(codes)
            return OptimizationResult(weights=w, method="equal_fallback", converged=False)

        # 协方差矩阵
        cov = self._covariance_matrix(returns)
        n = len(codes)

        # 根据方法计算权重
        if self._method == OptimizationMethod.EQUAL_WEIGHT:
            weights = self._equal_weights(codes)
        elif self._method == OptimizationMethod.ERC:
            weights = self._erc_optimize(cov, codes)
        elif self._method == OptimizationMethod.MIN_VARIANCE:
            weights = self._min_variance_optimize(cov, codes)
        elif self._method == OptimizationMethod.MAX_DIVERSIFICATION:
            weights = self._max_div_optimize(cov, codes)
        else:
            weights = self._equal_weights(codes)

        # 权重裁剪到风控上限并重归一化
        if position_limits:
            weights = self._clip_to_limits(weights, position_limits)

        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {c: w / total for c, w in weights.items()}

        # 计算期望波动率和分散化比率
        exp_vol = self._portfolio_volatility(weights, cov, codes)
        div_ratio = self._diversification_ratio(weights, cov, codes)

        return OptimizationResult(
            weights={c: round(weights.get(c, 0.0), 6) for c in codes},
            method=self._method.value,
            expected_volatility=round(exp_vol, 6),
            diversification_ratio=round(div_ratio, 4),
            converged=True,
        )

    # ── 收益率矩阵 ─────────────────────────────

    @staticmethod
    def _build_return_matrix(codes: list[str], daily_data: dict[str, list]) -> list[list[float]]:
        """从日线数据提取收益率矩阵 (按日期对齐)，shape: [n_assets][n_periods]"""
        # 收集所有公共日期
        date_sets = []
        for code in codes:
            records = daily_data.get(code, [])
            dates = {r.date: r.pct_chg for r in records if hasattr(r, "pct_chg")}
            date_sets.append(dates)

        if not date_sets:
            return []

        common = sorted(set.intersection(*map(set, date_sets)) if date_sets else set())
        if len(common) < 5:
            return []

        returns = []
        for i, code in enumerate(codes):
            dates = date_sets[i]
            returns.append([dates[d] for d in common])
        return returns

    # ── 协方差矩阵 ─────────────────────────────

    @staticmethod
    def _covariance_matrix(returns: list[list[float]]) -> list[list[float]]:
        """手动计算协方差矩阵"""
        n_assets = len(returns)
        n_periods = len(returns[0])
        means = [sum(r) / n_periods for r in returns]
        cov = [[0.0] * n_assets for _ in range(n_assets)]
        for i in range(n_assets):
            for j in range(n_assets):
                cov[i][j] = sum(
                    (returns[i][t] - means[i]) * (returns[j][t] - means[j])
                    for t in range(n_periods)
                ) / (n_periods - 1)
        return cov

    # ── ERC 优化 ───────────────────────────────

    def _erc_optimize(self, cov: list[list[float]], codes: list[str]) -> dict[str, float]:
        """
        等风险贡献优化。
        使用迭代算法: w_i = 1/sigma_i / sum(1/sigma_j), 逐步调整。
        """
        n = len(codes)
        # 初始权重: 逆波动率
        sigmas = [(cov[i][i] ** 0.5) if cov[i][i] > 0 else 1.0 for i in range(n)]
        inv_sigmas = [1.0 / s if s > 1e-10 else 1.0 for s in sigmas]
        total_inv = sum(inv_sigmas)
        w = [inv / total_inv for inv in inv_sigmas]

        for iteration in range(self._max_iter):
            # 计算边际风险贡献
            portfolio_var = sum(
                w[i] * w[j] * cov[i][j] for i in range(n) for j in range(n)
            )
            portfolio_vol = portfolio_var ** 0.5

            if portfolio_vol < 1e-10:
                break

            mrc = [0.0] * n
            for i in range(n):
                mrc[i] = sum(w[j] * cov[i][j] for j in range(n)) / portfolio_vol

            rc = [w[i] * mrc[i] for i in range(n)]
            avg_rc = sum(rc) / n

            # 检查收敛
            max_dev = max(abs(rc[i] - avg_rc) for i in range(n))
            if max_dev < self._tolerance:
                break

            # 调整权重: 风险贡献低的增权，高的减权
            for i in range(n):
                if rc[i] > 0:
                    w[i] *= avg_rc / rc[i]

            # 归一化
            total = sum(w)
            if total > 0:
                w = [wi / total for wi in w]

        return {codes[i]: max(0.0, w[i]) for i in range(n)}

    # ── 最小方差优化 ───────────────────────────

    def _min_variance_optimize(self, cov: list[list[float]], codes: list[str]) -> dict[str, float]:
        """
        最小方差优化。
        使用梯度下降: minimize w'Σw s.t. sum(w)=1, w_i >= 0.
        """
        n = len(codes)
        w = [1.0 / n] * n
        lr = 0.01

        for _ in range(self._max_iter):
            # 梯度: 2 * Σw (忽略因子2)
            grad = [0.0] * n
            for i in range(n):
                grad[i] = 2 * sum(w[j] * cov[i][j] for j in range(n))

            # 投影梯度 (保持 sum(w)=1)
            grad_mean = sum(grad) / n
            for i in range(n):
                w[i] -= lr * (grad[i] - grad_mean)
                w[i] = max(0.0, w[i])

            # 归一化
            total = sum(w)
            if total > 1e-10:
                w = [wi / total for wi in w]

            lr *= 0.99

        return {codes[i]: max(0.0, w[i]) for i in range(n)}

    # ── 最大分散化优化 ─────────────────────────

    def _max_div_optimize(self, cov: list[list[float]], codes: list[str]) -> dict[str, float]:
        """
        最大分散化优化。
        maximize (w'σ) / sqrt(w'Σw), σ = 各资产波动率对角向量.
        """
        n = len(codes)
        sigma = [(cov[i][i] ** 0.5) if cov[i][i] > 0 else 0.01 for i in range(n)]
        w = [1.0 / n] * n
        lr = 0.01

        for _ in range(self._max_iter):
            portfolio_var = sum(w[i] * w[j] * cov[i][j] for i in range(n) for j in range(n))
            portfolio_vol = portfolio_var ** 0.5
            if portfolio_vol < 1e-10:
                break

            weighted_sigma = sum(w[i] * sigma[i] for i in range(n))

            for i in range(n):
                mrc_i = sum(w[j] * cov[i][j] for j in range(n))
                grad = sigma[i] / portfolio_vol - weighted_sigma * mrc_i / (portfolio_vol ** 3)
                w[i] += lr * grad
                w[i] = max(0.0, w[i])

            total = sum(w)
            if total > 1e-10:
                w = [wi / total for wi in w]
            lr *= 0.99

        return {codes[i]: max(0.0, w[i]) for i in range(n)}

    # ── 辅助 ──────────────────────────────────

    @staticmethod
    def _equal_weights(codes: list[str]) -> dict[str, float]:
        n = len(codes)
        return {c: 1.0 / n for c in codes} if n > 0 else {}

    @staticmethod
    def _clip_to_limits(
        weights: dict[str, float],
        position_limits: dict,
    ) -> dict[str, float]:
        """将权重裁剪到风控上限"""
        result = {}
        for code, w in weights.items():
            limit = position_limits.get(code)
            if limit and hasattr(limit, "max_position_pct"):
                result[code] = min(w, limit.max_position_pct)
            else:
                result[code] = w
        return result

    @staticmethod
    def _portfolio_volatility(
        weights: dict[str, float],
        cov: list[list[float]],
        codes: list[str],
    ) -> float:
        """计算组合波动率"""
        n = len(codes)
        w = [weights.get(c, 0.0) for c in codes]
        var = sum(w[i] * w[j] * cov[i][j] for i in range(n) for j in range(n))
        return var ** 0.5

    @staticmethod
    def _diversification_ratio(
        weights: dict[str, float],
        cov: list[list[float]],
        codes: list[str],
    ) -> float:
        """计算分散化比率 = (Σ w_i*σ_i) / σ_portfolio"""
        n = len(codes)
        w = [weights.get(c, 0.0) for c in codes]
        sigma = [(cov[i][i] ** 0.5) if cov[i][i] > 0 else 0.0 for i in range(n)]
        weighted_vol = sum(w[i] * sigma[i] for i in range(n))
        portfolio_vol = sum(w[i] * w[j] * cov[i][j] for i in range(n) for j in range(n)) ** 0.5
        if portfolio_vol < 1e-10:
            return 1.0
        return round(weighted_vol / portfolio_vol, 4)

"""TransformerScorer — 将 StockTransformer 集成到 FactorScore 接口。"""

from ..screening.scorer import FactorScore
from .features import extract_features, pad_or_truncate
from .model import StockTransformer


class TransformerScorer:
    """基于 Transformer 的股票评分器，输出 list[FactorScore]。

    用法:
        model = StockTransformer.load("results/transformer_model.json")
        tscorer = TransformerScorer(model)
        scored = tscorer.score_all(codes, daily_data)
    """

    def __init__(self, model: StockTransformer, score_weight: float = 1.0):
        self._model = model
        self._score_weight = score_weight

    @property
    def model(self) -> StockTransformer:
        return self._model

    def score_single(
        self,
        code: str,
        daily_records: list,
        name: str = "",
    ) -> FactorScore | None:
        """对单只股票打分。

        Args:
            code: 股票代码
            daily_records: StockDaily 列表 (至少 20 条)
            name: 股票名称

        Returns:
            FactorScore 或 None (数据不足时)
        """
        if len(daily_records) < 20:
            return None

        features = extract_features(daily_records)
        padded, mask = pad_or_truncate(features, self._model.max_seq_len)

        raw_score = self._model.forward(padded, mask)
        composite = self._raw_to_score(raw_score)

        return FactorScore(
            code=code,
            name=name or code,
            scores={"transformer": round(composite, 1)},
            composite=round(composite, 1),
        )

    def score_all(
        self,
        codes: list[str],
        daily_data: dict[str, list],
    ) -> list[FactorScore]:
        """批量评分。

        Args:
            codes: 股票代码列表
            daily_data: {code: [StockDaily]}

        Returns:
            list[FactorScore]: 按 composite 降序排列
        """
        results: list[FactorScore] = []
        for code in codes:
            records = daily_data.get(code, [])
            if not records:
                continue
            fs = self.score_single(code, records)
            if fs is not None:
                results.append(fs)

        results.sort(key=lambda x: x.composite, reverse=True)
        return results

    @staticmethod
    def _raw_to_score(raw: float) -> float:
        """将预测收益率 [-0.3, 0.3] 映射到 [5, 95] 评分区间。

        score = 50 + raw * 150  (0.3 → 95, -0.3 → 5)
        """
        return max(5.0, min(95.0, 50.0 + raw * 150.0))

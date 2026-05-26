"""
ChromaDB 记忆存储 — 向量化历史决策轨迹。

设计:
- 每份 trace JSON 生成一段描述文本 → 向量化 → 存入 ChromaDB
- 支持按相似行情检索历史决策
- ChromaDB 不可用时优雅降级为空搜索

使用方式:
    store = MemoryStore()
    store.index_trace(trace_dict)
    results = store.search_similar(current_desc, k=5)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# 尝试导入 ChromaDB，不可用时降级
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    logger.info("chromadb 未安装，记忆系统降级为空搜索")


class MemoryStore:
    """
    ChromaDB 记忆存储。

    索引内容: trace JSON → 自然语言摘要 → 向量嵌入
    metadata: 日期、收益率、市场方向、主要决策等
    """

    COLLECTION_NAME = "trading_memories"

    def __init__(self, persist_dir: str | None = None) -> None:
        if not _CHROMA_AVAILABLE:
            self._client = None
            self._collection = None
            return

        if persist_dir is None:
            persist_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "chroma_db")

        os.makedirs(persist_dir, exist_ok=True)

        try:
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("MemoryStore: ChromaDB 已连接 (%d 条记录)", self._collection.count())
        except Exception:
            logger.warning("MemoryStore: ChromaDB 连接失败，降级为空搜索", exc_info=True)
            self._client = None
            self._collection = None

    @property
    def available(self) -> bool:
        return self._collection is not None

    def index_trace(self, trace: dict[str, Any]) -> bool:
        """
        将一份 trace JSON 索引到记忆库。

        trace: build_trace() 输出的完整字典
        返回: 是否成功
        """
        if not self.available:
            return False

        date_str = trace.get("date", "")
        if not date_str:
            return False

        # 检查是否已索引
        existing = self._collection.get(ids=[date_str])
        if existing and existing.get("ids") and len(existing["ids"]) > 0:
            logger.debug("MemoryStore: %s 已索引，跳过", date_str)
            return True

        doc = _trace_to_document(trace)
        metadata = _trace_to_metadata(trace)

        try:
            self._collection.add(
                ids=[date_str],
                documents=[doc],
                metadatas=[metadata],
            )
            logger.info("MemoryStore: 已索引 %s", date_str)
            return True
        except Exception:
            logger.warning("MemoryStore: 索引失败 %s", date_str, exc_info=True)
            return False

    def search_similar(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        搜索与当前市场状况最相似的历史交易日。

        query: 当前市场状况的自然语言描述
        k: 返回条数

        返回: [{"date": ..., "score": ..., "metadata": {...}}, ...]
        """
        if not self.available:
            return []

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(k, self._collection.count()),
                include=["metadatas", "distances"],
            )
        except Exception:
            logger.warning("MemoryStore: 搜索失败", exc_info=True)
            return []

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results.get("distances", [[0]] * k)[0][i] if results.get("distances") else 0
            similarity = round(1 - distance, 4)  # cosine distance → similarity
            meta = (results.get("metadatas", [{}])[0][i:i + 1] or [{}])[0]

            output.append({
                "date": doc_id,
                "similarity": similarity,
                "direction": meta.get("direction", ""),
                "return_pct": meta.get("return_pct", 0),
                "position_count": meta.get("position_count", 0),
                "errors_count": meta.get("errors_count", 0),
            })

        output.sort(key=lambda x: x["similarity"], reverse=True)
        return output

    def count(self) -> int:
        if not self.available:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0


def _trace_to_document(trace: dict[str, Any]) -> str:
    """将 trace JSON 转为自然语言文档，用于向量嵌入"""
    parts = []

    screening = trace.get("screening", {})
    candidates = screening.get("candidates", [])
    parts.append(f"候选池 {len(candidates)} 只股票")

    # 候选股板块分布
    if candidates:
        top = candidates[:5]
        parts.append("得分最高的股票: " + ", ".join(
            f"{c.get('code','')} {c.get('name','')} 综合{c.get('score',0):.0f}分"
            for c in top
        ))

    verdicts = trace.get("verdicts", {})
    buy_count = sum(1 for v in verdicts.values() if v.get("direction") == "buy")
    sell_count = sum(1 for v in verdicts.values() if v.get("direction") == "sell")
    parts.append(f"研判结果: {buy_count}只买入 {sell_count}只卖出")

    decisions = trace.get("decisions", [])
    if decisions:
        parts.append("最终买入: " + ", ".join(
            f"{d.get('symbol','')} {d.get('symbol_name','')} {d.get('volume',0)}股"
            for d in decisions
        ))
    else:
        parts.append("最终决策: 空仓")

    portfolio = trace.get("portfolio", {})
    parts.append(f"使用资金 {portfolio.get('cash_used',0):.0f} 剩余 {portfolio.get('cash_remaining',0):.0f}")

    errors = trace.get("errors", [])
    if errors:
        parts.append(f"发生 {len(errors)} 个错误: {'; '.join(errors[:3])}")

    return "。".join(parts)


def _trace_to_metadata(trace: dict[str, Any]) -> dict[str, Any]:
    """提取 trace 的结构化元数据"""
    verdicts = trace.get("verdicts", {})
    buy_count = sum(1 for v in verdicts.values() if v.get("direction") == "buy")
    decisions = trace.get("decisions", [])
    portfolio = trace.get("portfolio", {})

    # 判断市场方向
    if buy_count >= 3 and decisions:
        direction = "strong_bull"
    elif decisions:
        direction = "moderate_bull"
    elif buy_count >= 2:
        direction = "cautious"
    else:
        direction = "bearish"

    total_cap = trace.get("total_capital", 500000)
    cash_remaining = portfolio.get("cash_remaining", total_cap)
    cash_ratio = cash_remaining / total_cap if total_cap > 0 else 1.0

    return {
        "direction": direction,
        "position_count": len(decisions),
        "cash_ratio": round(cash_ratio, 2),
        "candidates_count": len(trace.get("screening", {}).get("candidates", [])),
        "errors_count": len(trace.get("errors", [])),
        "return_pct": 0.0,  # 需后续更新
    }

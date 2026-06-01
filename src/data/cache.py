"""
数据缓存 — 基于字典的 TTL 缓存 + JSON 文件持久化。

设计原则:
- 优先复用已有项目模式，保持简洁
- TTL 自动过期，手动刷新
- 磁盘持久化保证重启后可恢复最近缓存
"""

import dataclasses
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Optional

from ..utils.config import get_config

logger = logging.getLogger(__name__)

# 延迟注册的数据类类型映射 (用于缓存反序列化时重建对象)
_TYPE_MAP: dict[str, type] = {}
_TYPE_REGISTERED = False


def _register_types() -> None:
    """延迟注册已知的 dataclass 类型 (避免循环导入)."""
    global _TYPE_REGISTERED
    if _TYPE_REGISTERED:
        return
    try:
        from .fetchers.akshare_fetcher import (  # noqa: F811
            StockDaily, RealtimeQuote, FundFlow, MarketSnapshot,
            NorthboundFlow, MarginData, FinancialIndicator, ETFSpot,
            UnlockShares, ShareholderCount, InstitutionalVisit, MarketActivity,
        )
        for cls in [
            StockDaily, RealtimeQuote, FundFlow, MarketSnapshot,
            NorthboundFlow, MarginData, FinancialIndicator, ETFSpot,
            UnlockShares, ShareholderCount, InstitutionalVisit, MarketActivity,
        ]:
            _TYPE_MAP[cls.__name__] = cls
    except ImportError:
        pass
    _TYPE_REGISTERED = True


@dataclass
class CacheEntry:
    data: Any
    created_at: float
    ttl: int  # 秒


class DataCache:
    """线程安全的 TTL 缓存，支持磁盘持久化。"""

    def __init__(self) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._lock = RLock()
        self._config = get_config()
        self._persist_path = Path(self._config.results_dir) / ".cache.json"
        self._load_from_disk()

    # ── 磁盘持久化 ──────────────────────────

    @staticmethod
    def _to_serializable(data: Any) -> Any:
        """将数据转为可 JSON 序列化的格式 (dataclass -> dict)."""
        if dataclasses.is_dataclass(data) and not isinstance(data, type):
            return {
                "__type__": type(data).__name__,
                "fields": dataclasses.asdict(data),
            }
        if isinstance(data, list):
            return [DataCache._to_serializable(item) for item in data]
        return data

    @staticmethod
    def _from_serializable(data: Any) -> Any:
        """从序列化格式恢复 dataclass 对象."""
        _register_types()
        if isinstance(data, dict) and "__type__" in data:
            cls = _TYPE_MAP.get(data["__type__"])
            if cls is not None:
                return cls(**data["fields"])
            return data  # 类型未知，保持 dict
        if isinstance(data, list):
            return [DataCache._from_serializable(item) for item in data]
        return data

    def _load_from_disk(self) -> None:
        """从磁盘恢复缓存"""
        try:
            if self._persist_path.exists():
                with open(self._persist_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                loaded = 0
                now = time.time()
                for key, entry in raw.items():
                    # 跳过已过期的条目
                    if now - entry["created_at"] > entry["ttl"]:
                        continue
                    self._store[key] = CacheEntry(
                        data=self._from_serializable(entry["data"]),
                        created_at=entry["created_at"],
                        ttl=entry["ttl"],
                    )
                    loaded += 1
                if loaded:
                    logger.info("缓存: 从磁盘恢复 %d 条记录", loaded)
        except Exception:
            logger.debug("缓存: 磁盘加载失败，使用空缓存")
            self._store.clear()

    def _save_to_disk(self) -> None:
        """持久化到磁盘 (全程持锁，防止并发写入). dataclass 对象自动转为 dict."""
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = {}
                for key, entry in self._store.items():
                    payload[key] = {
                        "data": self._to_serializable(entry.data),
                        "created_at": entry.created_at,
                        "ttl": entry.ttl,
                    }
                with open(self._persist_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
        except Exception:
            logger.debug("缓存: 磁盘保存失败", exc_info=True)

    # ── 基本操作 ─────────────────────────────

    def _make_key(self, prefix: str, *parts: str) -> str:
        return f"{prefix}:{':'.join(parts)}"

    def get(self, prefix: str, *parts: str) -> Optional[Any]:
        key = self._make_key(prefix, *parts)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.time() - entry.created_at > entry.ttl:
                del self._store[key]
                return None
            return entry.data

    def set(self, prefix: str, data: Any, ttl: int, *parts: str) -> None:
        key = self._make_key(prefix, *parts)
        with self._lock:
            self._store[key] = CacheEntry(data=data, created_at=time.time(), ttl=ttl)

    def clear(self, prefix: Optional[str] = None) -> None:
        with self._lock:
            if prefix is None:
                self._store.clear()
            else:
                keys = [k for k in self._store if k.startswith(f"{prefix}:")]
                for k in keys:
                    del self._store[k]

    # ── 便捷方法 ─────────────────────────────

    def daily_data(self, code: str) -> Optional[Any]:
        return self.get("daily", code)

    def set_daily_data(self, code: str, data: Any) -> None:
        self.set("daily", data, self._config.cache_ttl_daily, code)

    def realtime_quote(self, code: str) -> Optional[Any]:
        return self.get("realtime", code)

    def set_realtime_quote(self, code: str, data: Any) -> None:
        self.set("realtime", data, self._config.cache_ttl_realtime, code)

    def fundamentals(self, code: str) -> Optional[Any]:
        return self.get("fundamental", code)

    def set_fundamentals(self, code: str, data: Any) -> None:
        self.set("fundamental", data, self._config.cache_ttl_fundamental, code)

    # ── 生命周期 ─────────────────────────────

    def persist(self) -> None:
        """显式触发磁盘持久化"""
        self._save_to_disk()

    def __del__(self) -> None:
        """析构时自动持久化 (最佳努力)"""
        try:
            self._save_to_disk()
        except Exception:
            pass


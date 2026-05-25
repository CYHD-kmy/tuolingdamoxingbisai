"""
简单数据缓存 - 基于字典的 TTL 缓存。

设计原则:
- 优先复用已有项目模式，保持简洁
- TTL 自动过期，手动刷新
"""

import time
from dataclasses import dataclass
from threading import RLock
from typing import Any, Optional

from ..utils.config import get_config


@dataclass
class CacheEntry:
    data: Any
    created_at: float
    ttl: int  # 秒


class DataCache:
    """线程安全的 TTL 缓存。"""

    def __init__(self) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._lock = RLock()
        self._config = get_config()

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

"""
配置管理 - 从环境变量和 .env 文件加载配置。

设计原则：
- 新配置做到"不配置也可运行，配置后增强能力"
- 所有密钥通过环境变量注入，不写死
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

def _project_root() -> Path:
    """惰性计算项目根目录 (避免 import-time 副作用)."""
    return Path(__file__).resolve().parent.parent.parent


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("1", "true", "yes", "on")


@dataclass
class Config:
    # ── 数据源 ──────────────────────────────────
    tushare_token: Optional[str] = field(
        default_factory=lambda: os.getenv("TUSHARE_TOKEN")
    )

    # 数据源优先级: 0=优先, 值越大越靠后
    # 当 TUSHARE_TOKEN 可用时，Tushare 自动升到最高优先级
    akshare_priority: int = 1
    tushare_priority: int = 0
    baostock_priority: int = 3

    # 数据缓存 TTL (秒)
    cache_ttl_daily: int = 300        # 日线数据 5min
    cache_ttl_realtime: int = 60      # 实时行情 1min
    cache_ttl_fundamental: int = 3600 # 基本面 1hour

    # 请求控制
    request_timeout: int = 15
    max_retries: int = 3
    retry_backoff_base: float = 2.0

    # ── 市场过滤 ──────────────────────────────
    min_daily_amount: float = 50_000_000    # 最小日均成交额 (5000万)
    max_candidates: int = 20                # 海选后进入深度分析的股票数
    min_listing_days: int = 60              # 排除上市不足60天的新股
    max_volatility_pct: float = 9.8         # 单日涨跌幅异常阈值 (A股±10%涨跌停内)

    # ── 风控参数 ──────────────────────────────
    initial_capital: float = 500_000.0      # 初始虚拟资金 50万
    max_single_position: float = 0.20       # 单票 ≤ 20%
    max_industry_exposure: float = 0.40     # 同行业 ≤ 40%
    max_daily_turnover: float = 0.50        # 日换手率 ≤ 50%
    max_drawdown_daily: float = 0.05        # 日内熔断线 5%
    min_cash_reserve: float = 0.10          # 保留 ≥ 10% 现金

    # ── LLM ────────────────────────────────────
    llm_quick: str = field(
        default_factory=lambda: os.getenv("LLM_QUICK_MODEL", "deepseek-chat")
    )
    llm_deep: str = field(
        default_factory=lambda: os.getenv("LLM_DEEP_MODEL", "deepseek-reasoner")
    )
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "")
    )
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.3"))
    )
    llm_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "4096"))
    )

    # ── 辩论 ────────────────────────────────────
    max_debate_rounds: int = 3
    max_analyst_tool_calls: int = 3

    # ── 输出 ────────────────────────────────────
    results_dir: str = field(
        default_factory=lambda: str(_project_root() / "results")
    )
    save_trace: bool = True

    @property
    def tushare_available(self) -> bool:
        return bool(self.tushare_token)

    def fetcher_priority(self, name: str) -> int:
        """返回 fetcher 的有效优先级 (考虑 Token 是否可用)"""
        if name == "tushare" and not self.tushare_available:
            return 999  # 不可用
        if name == "tushare" and self.tushare_available:
            return 0    # Token 可用时提到最高
        return getattr(self, f"{name}_priority", 99)


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config

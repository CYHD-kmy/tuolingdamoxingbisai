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

from dotenv import load_dotenv

# 自动加载项目根目录 .env 文件
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# 代理绕过: macOS 系统代理可能拦截金融数据源响应
for _proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                    "ALL_PROXY", "all_proxy"):
    os.environ.pop(_proxy_var, None)
os.environ["NO_PROXY"] = "*"
import urllib.request as _urllib_request
_urllib_request.getproxies = lambda: {}

import requests as _requests
_requests.Session.trust_env = False

def _project_root() -> Path:
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

    akshare_priority: int = 999  # 永久降级为兜底
    tushare_priority: int = 0
    baostock_priority: int = 3

    # AKShare 代理 (国内 IP 访问东方财富/新浪等 CDN)
    akshare_proxy: Optional[str] = field(
        default_factory=lambda: os.getenv("AKSHARE_PROXY",
                                          "http://127.0.0.1:7897")
        if os.getenv("AKSHARE_PROXY") or os.path.exists(
            "/Applications/Clash Verge.app") else ""
    )

    cache_ttl_daily: int = 300
    cache_ttl_realtime: int = 60
    cache_ttl_fundamental: int = 3600

    request_timeout: int = 15
    max_retries: int = 3
    retry_backoff_base: float = 2.0

    # ── 市场过滤 ──────────────────────────────
    min_daily_amount: float = 50_000_000
    max_candidates: int = 20
    min_listing_days: int = 60
    max_volatility_pct: float = 9.8

    # ── 风控参数 ──────────────────────────────
    initial_capital: float = 500_000.0

    # 三级仓位分层
    core_total_pct: float = 0.40
    core_single_pct: float = 0.40
    satellite_total_pct: float = 0.35
    satellite_single_pct: float = 0.14
    min_cash_reserve: float = 0.25

    max_industry_exposure: float = 0.40
    max_daily_turnover: float = 0.50
    max_drawdown_daily: float = 0.05

    open_loss_filter_pct: float = -5.0
    broad_decline_threshold: int = 3000

    # ATR 止损
    atr_period: int = 14
    atr_stop_multiplier: float = 3.0

    @property
    def max_single_position(self) -> float:
        return self.core_single_pct

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

    # ── 增强分析模块 ───────────────────────────
    # 新增分析师 (政策面 + 板块猎手)
    enable_policy_analyst: bool = True
    enable_sector_hunter: bool = True

    # 市场情绪分析
    enable_market_sentiment: bool = True
    broad_decline_threshold: int = 3000
    broad_advance_threshold: int = 3000

    # 集合竞价
    enable_auction_analysis: bool = True

    # 龙虎榜
    enable_dragon_tiger: bool = True

    # 涨停分析
    enable_limit_up_analysis: bool = True

    # 量价关系
    enable_volume_price: bool = True

    # 内部竞赛机制 (ContestTrade 风格)
    enable_competition_scoring: bool = True
    competition_consensus_threshold: int = 3  # 至少 N 个分析师看好才进入辩论

    # ── ETF ──────────────────────────────────────
    etf_enabled: bool = True
    etf_max_allocation: float = 0.20
    etf_max_candidates: int = 5
    etf_min_daily_amount: float = 50_000_000
    etf_min_fund_size: float = 100_000_000
    etf_max_single_position: float = 0.10

    # ── 组合管理 ────────────────────────────────
    use_target_allocation: bool = False  # True=LLM分配权重, False=确定性规则

    # ── 输出 ────────────────────────────────────
    results_dir: str = field(
        default_factory=lambda: str(_project_root() / "results")
    )
    save_trace: bool = True

    def __post_init__(self) -> None:
        import logging
        _log = logging.getLogger(__name__)

        checks: list[tuple[float, float, float, str]] = [
            (self.core_single_pct, 0.05, 0.50, "core_single_pct"),
            (self.satellite_single_pct, 0.01, 0.25, "satellite_single_pct"),
            (self.max_industry_exposure, 0.05, 1.00, "max_industry_exposure"),
            (self.max_drawdown_daily, 0.01, 0.20, "max_drawdown_daily"),
            (self.min_cash_reserve, 0.05, 0.50, "min_cash_reserve"),
            (self.request_timeout, 3, 120, "request_timeout"),
            (self.max_retries, 0, 10, "max_retries"),
        ]
        for value, lo, hi, name in checks:
            if not (lo <= value <= hi):
                _log.warning("Config.%s=%.2f 超出合理范围 [%.2f, %.2f]，请确认配置",
                             name, value, lo, hi)

    @property
    def tushare_available(self) -> bool:
        return bool(self.tushare_token)

    def fetcher_priority(self, name: str) -> int:
        if name == "tushare" and not self.tushare_available:
            return 999
        if name == "tushare" and self.tushare_available:
            return 0
        return getattr(self, f"{name}_priority", 99)


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config

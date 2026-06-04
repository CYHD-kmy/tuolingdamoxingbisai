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

# 代理绕过: macOS 系统代理 (Clash/V2Ray/Surge) 可能拦截/破坏金融数据源响应,
# 通过清除代理环境变量 + monkey-patch 双重确保直连
for _proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                    "ALL_PROXY", "all_proxy"):
    os.environ.pop(_proxy_var, None)
os.environ["NO_PROXY"] = "*"  # requests/urllib3 读取此变量决定是否走代理
import urllib.request as _urllib_request
_urllib_request.getproxies = lambda: {}

# requests 库在 macOS 上通过 _scproxy 直接读系统代理，不依赖环境变量,
# 必须显式禁用 trust_env 才能绕过
import requests as _requests
_requests.Session.trust_env = False

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
    akshare_priority: int = 999  # 永久降级为兜底
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

    # 三级仓位分层 (核心重仓 / 卫星分散 / 现金备用)
    core_total_pct: float = 0.40            # 核心仓总占比 ≤ 40% (20万)
    core_single_pct: float = 0.40           # 核心单票上限 ≤ 40% (20万, 最多1-2只)
    satellite_total_pct: float = 0.35       # 卫星仓总占比 ≤ 35% (17.5万)
    satellite_single_pct: float = 0.14      # 卫星单票上限 ≤ 14% (7万, 3-4只)
    min_cash_reserve: float = 0.25          # 现金备⽤ ≥ 25% (12.5万)

    # 核心/卫星分类阈值
    core_score_threshold: float = 75.0      # 综合得分 ≥ 75 可进入核心仓
    core_pe_max: float = 30.0               # 核心仓 PE ≤ 30 (低估值龙头)
    core_market_cap_min: float = 100e8      # 核心仓 市值 ≥ 100亿

    # 市场环境自适应 — 总仓位上限
    max_total_position_bull: float = 0.75   # 牛市: 核心40% + 卫星35%
    max_total_position_neutral: float = 0.34  # 震荡: 核心20% + 卫星14%
    max_total_position_bear: float = 0.0    # 熊市: 强制空仓

    max_industry_exposure: float = 0.40     # 同行业 ≤ 40%
    max_daily_turnover: float = 0.50        # 日换手率 ≤ 50%
    max_drawdown_daily: float = 0.05        # 日内熔断线 5%

    # 单日风控
    open_loss_filter_pct: float = -5.0      # 开盘跌幅超过此值则剔除 (默认-5%)
    broad_decline_threshold: int = 3000     # 全市场下跌超N只 → 强制空仓

    # 兼容旧参数 (deprecated, 使用 core_single_pct 替代)
    @property
    def max_single_position(self) -> float:
        """向后兼容: 返回核心单票上限 (新代码请使用 core_single_pct / satellite_single_pct)"""
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

    # ── 回测 ────────────────────────────────────
    backtest_start_date: str = ""
    backtest_end_date: str = ""
    backtest_benchmark: str = "000300"
    backtest_output_dir: str = field(
        default_factory=lambda: str(_project_root() / "results" / "backtests")
    )

    # ── 组合优化 ──────────────────────────────
    risk_parity_method: str = "equal"  # equal / erc / min_var / max_div
    risk_parity_lookback: int = 20

    # ── 多策略 ──────────────────────────────────
    active_strategies: str = "default"  # 逗号分隔: default,momentum,mean_reversion,quality,sentiment
    strategy_rebalance_period: int = 10

    # ── 强化学习 ──────────────────────────────
    rl_enabled: bool = False
    rl_model_path: str = ""
    rl_episodes: int = 200
    rl_signal_weight: float = 0.15

    # ── Transformer ───────────────────────────
    transformer_enabled: bool = field(
        default_factory=lambda: _env_bool("TRANSFORMER_ENABLED", False)
    )
    transformer_model_path: str = field(
        default_factory=lambda: os.getenv("TRANSFORMER_MODEL_PATH", "")
    )
    transformer_train_epochs: int = 50
    transformer_seq_len: int = 30
    transformer_forward_days: int = 5
    transformer_lr: float = 0.001
    transformer_scorer_weight: float = 0.30  # 与手工因子融合时的权重
    transformer_rl_features: bool = field(
        default_factory=lambda: _env_bool("TRANSFORMER_RL_FEATURES", False)
    )

    # ── 持仓管理 ──────────────────────────────
    # 持仓天数管理
    holding_reduce_days: int = 5            # 持有超过此天数且收益低于阈值 → 建议减仓
    holding_reduce_return: float = 0.03     # 减仓收益阈值 (3%)
    holding_clear_days: int = 8             # 持有超过此天数且收益低于阈值 → 建议清仓
    holding_clear_return: float = 0.0       # 清仓收益阈值 (0%)

    # ATR 动态止损
    atr_period: int = 14                    # ATR 计算周期
    atr_stop_multiplier: float = 2.0        # 止损 ATR 倍数 (stop = cost - N * ATR)

    # 目标仓位模型
    use_target_allocation: bool = True      # 启用目标仓位模式 (而非直接买卖信号)
    max_total_alloc_weight: float = 0.90    # 总仓位权重上限 (保留 10% 现金)

    # ── 市场环境 ──────────────────────────────
    regime_lookback_days: int = 20          # 市场环境判定回看天数
    regime_bull_mult: float = 1.00          # 牛市仓位倍率 (1.0=全量, 核心40%+卫星35%)
    regime_bear_mult: float = 0.0           # 熊市仓位倍率 (0.0=强制空仓)
    regime_index_code: str = "000300"       # 市场环境判定基准指数 (沪深300)

    # ── ETF ──────────────────────────────────────
    etf_enabled: bool = True
    etf_max_allocation: float = 0.20       # ETF 最大资金占比
    etf_max_candidates: int = 5            # ETF 候选数量
    etf_min_daily_amount: float = 50_000_000   # ETF 最小日均成交额
    etf_min_fund_size: float = 100_000_000     # ETF 最小基金规模
    etf_max_single_position: float = 0.10  # 单只 ETF 最大仓位

    # ── 持仓复盘 ──────────────────────────────
    review_enabled: bool = field(
        default_factory=lambda: _env_bool("REVIEW_ENABLED", True)
    )
    review_llm_model: str = field(
        default_factory=lambda: os.getenv("REVIEW_LLM_MODEL", "deepseek-chat")
    )
    review_post_mortem_days: int = 5     # 事后验证回溯天数

    # ── 输出 ────────────────────────────────────
    results_dir: str = field(
        default_factory=lambda: str(_project_root() / "results")
    )
    save_trace: bool = True

    def __post_init__(self) -> None:
        """校验关键参数范围，超出范围仅告警 (不覆写用户配置)。"""
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

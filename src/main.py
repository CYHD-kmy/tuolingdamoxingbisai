"""
智投未来 — A股日内投资智能体 主入口。

使用方式:
    python -m src.main              # 正常模式 (需要数据源和 LLM API Key)
    python -m src.main --demo       # 演示模式 (使用样本数据，无需网络和 API)

环境变量:
    LLM_API_KEY      - LLM API Key (必填，--demo 模式下不需要)
    TUSHARE_TOKEN    - Tushare Token (可选，有则自动启用)
    LLM_QUICK_MODEL  - quick 模型 (默认 deepseek-chat)
    LLM_DEEP_MODEL   - deep 模型 (默认 deepseek-reasoner)
    LLM_BASE_URL     - API 地址 (默认 https://api.deepseek.com)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.config import get_config
from src.graph.workflow import run_pipeline
from src.output.json_formatter import format_decisions, validate_decisions
from src.output.trace_logger import save_trace


def setup_logging(level: int = logging.INFO) -> None:
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def main(demo: bool = False) -> None:
    """主函数 — 输出符合比赛标准的 JSON 决策（只输出到 stdout）"""
    setup_logging()
    logger = logging.getLogger("main")

    config = get_config()
    t0 = time.monotonic()

    if demo:
        from src.demo import generate_demo_state

        logger.info("===== 智投未来 启动 (演示模式) =====")
        logger.info("日期: %s", datetime.now().strftime("%Y-%m-%d"))
        state = generate_demo_state()
    else:
        if not config.llm_api_key:
            logger.error("LLM_API_KEY 未设置。")
            logger.error("请设置环境变量: export LLM_API_KEY=sk-xxx")
            logger.error("或使用演示模式: python -m src.main --demo")
            sys.exit(1)

        logger.info("===== 智投未来 启动 =====")
        logger.info("日期: %s", datetime.now().strftime("%Y-%m-%d"))
        logger.info("Quick LLM: %s", config.llm_quick)
        logger.info("Deep LLM:  %s", config.llm_deep)
        logger.info("Tushare: %s", "可用" if config.tushare_available else "未配置")

        try:
            state = run_pipeline(
                total_capital=config.initial_capital,
                available_cash=config.initial_capital,
            )
        except KeyboardInterrupt:
            logger.warning("用户中断")
            sys.exit(0)
        except Exception:
            logger.exception("流水线异常终止")
            sys.exit(1)

    total_elapsed = time.monotonic() - t0

    # ── 输出 (仅 JSON 到 stdout, 日志到 stderr) ──
    final_result = getattr(state, "final_result", None)
    position_limits = getattr(state, "position_limits", {})
    daily_data = getattr(state, "daily_data", {})
    verdicts = getattr(state, "verdicts", {})

    if final_result and final_result.decisions:
        validated = validate_decisions(
            final_result.decisions,
            position_limits,
            daily_data,
            cash_available=config.initial_capital,
            min_cash_reserve=config.min_cash_reserve,
            total_capital=config.initial_capital,
            verdicts=verdicts,
        )
        decisions_json = format_decisions(validated)
        # 仅输出 JSON 到 stdout，符合比赛"直接只返回JSON"的要求
        print(json.dumps(decisions_json, ensure_ascii=False))
    else:
        print("[]")

    logger.info("耗时: %.1fs, 决策: %d 笔", total_elapsed,
                len(final_result.decisions) if final_result else 0)

    # ── 保存 trace (可选) ──
    if config.save_trace:
        try:
            trace_path = save_trace(state, total_elapsed, config.results_dir)
            logger.info("结果已保存: %s", trace_path)
        except Exception:
            logger.debug("保存 trace 失败", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智投未来 — A股日内投资智能体")
    parser.add_argument("--demo", action="store_true", help="使用演示数据运行")
    parser.add_argument("--capital", type=float, default=500_000.0,
                        help="可用资金 (默认 500000)")
    args = parser.parse_args()
    main(demo=args.demo)

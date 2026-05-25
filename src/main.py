"""
智投未来 — A股日内投资智能体 主入口。

使用方式:
    python -m src.main

环境变量:
    LLM_API_KEY      - LLM API Key (必填)
    TUSHARE_TOKEN    - Tushare Token (可选，有则自动启用)
    LLM_QUICK_MODEL  - quick 模型 (默认 deepseek-chat)
    LLM_DEEP_MODEL   - deep 模型 (默认 deepseek-reasoner)
    LLM_BASE_URL     - API 地址 (默认 https://api.deepseek.com)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime

# 确保项目根在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.config import get_config
from src.graph.workflow import run_pipeline


def setup_logging(level: int = logging.INFO) -> None:
    """配置日志"""
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # 降低第三方库日志级别
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def main() -> None:
    """主函数"""
    setup_logging()
    logger = logging.getLogger("main")

    config = get_config()

    # 前置检查
    if not config.llm_api_key:
        logger.error("LLM_API_KEY 未设置。请设置环境变量: export LLM_API_KEY=sk-xxx")
        sys.exit(1)

    logger.info("===== 智投未来 启动 =====")
    logger.info("日期: %s", datetime.now().strftime("%Y-%m-%d"))
    logger.info("总资金: ¥%.0f", config.initial_capital)
    logger.info("Quick LLM: %s", config.llm_quick)
    logger.info("Deep LLM:  %s", config.llm_deep)
    logger.info("Tushare: %s", "可用" if config.tushare_available else "未配置")
    logger.info("")

    t0 = time.monotonic()

    try:
        state = run_pipeline(total_capital=config.initial_capital)
    except KeyboardInterrupt:
        logger.warning("用户中断")
        sys.exit(0)
    except Exception:
        logger.exception("流水线异常终止")
        sys.exit(1)

    total_elapsed = time.monotonic() - t0

    # ── 输出 ──────────────────────────────────

    print("\n" + "=" * 50)
    print("  最终决策 (赛道 JSON 格式)")
    print("=" * 50)

    if state.final_result and state.final_result.decisions:
        decisions_json = [d.to_dict() for d in state.final_result.decisions]
        print(json.dumps(decisions_json, ensure_ascii=False, indent=2))
        print(f"\n使用资金: ¥{state.final_result.cash_used:,.0f}")
        print(f"剩余资金: ¥{state.final_result.cash_remaining:,.0f}")
    else:
        print("[]  (空仓)")

    print(f"\n总耗时: {total_elapsed:.1f}s")
    for stage, elapsed in state.elapsed.items():
        print(f"  {stage}: {elapsed:.1f}s")

    if state.errors:
        print(f"\n警告/错误 ({len(state.errors)}条):")
        for e in state.errors[:5]:
            print(f"  - {e}")

    # 保存结果
    if config.save_trace:
        os.makedirs(config.results_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        result_path = os.path.join(config.results_dir, f"result_{date_str}.json")
        trace = {
            "date": date_str,
            "elapsed": total_elapsed,
            "stage_elapsed": state.elapsed,
            "candidates": [
                {"code": c.code, "name": c.name, "score": c.composite}
                for c in state.candidates
            ],
            "verdicts": {
                code: {
                    "direction": v.direction,
                    "confidence": v.confidence,
                    "risk_level": v.risk_level,
                    "core_reasoning": v.core_reasoning,
                }
                for code, v in state.verdicts.items()
            },
            "decisions": state.final_result.decisions if state.final_result else [],
            "errors": state.errors,
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
        logger.info("结果已保存: %s", result_path)


if __name__ == "__main__":
    main()

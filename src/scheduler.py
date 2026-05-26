"""
定时调度器 — 支持每日自动运行流水线。

调度方式:
    python -m src.scheduler              # 启动调度器 (阻塞运行)
    python -m src.scheduler --once       # 立即执行一次后退出

配置:
    通过环境变量控制:
    - ZHITOU_SCHEDULE_HOUR: 每日触发时刻 (默认 9)
    - ZHITOU_SCHEDULE_MINUTE: 每日触发分钟 (默认 0)
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from datetime import datetime

logger = logging.getLogger("scheduler")


def _get_schedule_time() -> tuple[int, int]:
    hour = int(os.getenv("ZHITOU_SCHEDULE_HOUR", "9"))
    minute = int(os.getenv("ZHITOU_SCHEDULE_MINUTE", "0"))
    return hour, minute


def run_once() -> None:
    """立即执行一次完整流水线"""
    try:
        from src.main import main
    except ImportError as e:
        logger.error("无法导入 src.main: %s。请确保在项目根目录下运行。", e)
        return

    logger.info("调度器: 立即执行一次")
    main(demo=False)


def run_scheduled() -> None:
    """
    阻塞运行调度循环。

    每日在配置的时间触发一次流水线。
    优雅退出: 收到 SIGINT/SIGTERM 后完成当前任务再退出。
    """
    hour, minute = _get_schedule_time()
    logger.info("调度器已启动: 每日 %02d:%02d 触发", hour, minute)

    running = True

    def _shutdown(signum, frame):
        nonlocal running
        logger.info("收到信号 %s，将在当前任务完成后退出", signum)
        running = False

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    while running:
        now = datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        if now >= target:
            # 今天的目标时间已过，等明天
            from datetime import timedelta
            target = target + timedelta(days=1)

        wait_seconds = (target - now).total_seconds()
        logger.info("下次触发: %s (%.0f 分钟后)", target.strftime("%Y-%m-%d %H:%M"), wait_seconds / 60)

        # 分段等待，每 60 秒检查一次退出信号
        while wait_seconds > 0 and running:
            sleep_chunk = min(60, wait_seconds)
            time.sleep(sleep_chunk)
            wait_seconds -= sleep_chunk

        if not running:
            break

        # 检查是否为交易日
        try:
            from src.utils.trading_calendar import is_trading_day
            if not is_trading_day(datetime.now()):
                logger.info("今日非交易日，跳过")
                continue
        except Exception:
            logger.warning("交易日历不可用，默认每日执行", exc_info=True)

        logger.info("===== 定时触发开始 =====")
        try:
            run_once()
        except Exception:
            logger.exception("定时任务执行失败")
        logger.info("===== 定时触发完成 =====")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="智投未来 定时调度器")
    parser.add_argument("--once", action="store_true", help="立即执行一次后退出")
    args = parser.parse_args()

    # 确保项目根在 sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    if args.once:
        run_once()
    else:
        run_scheduled()

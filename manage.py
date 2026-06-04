#!/usr/bin/env python3
"""
智投未来 — 跨平台管理脚本 (Windows / macOS / Linux)

Usage: python manage.py {run|run-demo|setup|schedule|schedule-stop|schedule-demo|run-scheduled}
"""

from __future__ import annotations

import argparse
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

PLATFORM = platform.system()


def _load_env():
    env_file = PROJECT_DIR / ".env"
    if not env_file.exists():
        return
    with open(env_file, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if key in os.environ:
                continue
            val = val.strip()
            if val and val[0] in ('"', "'"):
                quote = val[0]
                end_idx = val.find(quote, 1)
                if end_idx != -1:
                    val = val[1:end_idx]
                else:
                    val = val[1:]
            elif "#" in val:
                val = val.split("#")[0].strip()
            os.environ[key] = val


def _detect_python() -> str:
    for candidate in [
        PROJECT_DIR / ".venv" / "bin" / "python3",
        PROJECT_DIR / ".venv" / "bin" / "python",
        PROJECT_DIR / "venv" / "bin" / "python3",
        PROJECT_DIR / "venv" / "bin" / "python",
        PROJECT_DIR / ".venv" / "Scripts" / "python.exe",
        PROJECT_DIR / "venv" / "Scripts" / "python.exe",
    ]:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _pid_alive(pid: int) -> bool:
    try:
        if PLATFORM == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x0400, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError, PermissionError):
        return False


def _kill_process(pid: int):
    try:
        if PLATFORM == "Windows":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, signal.SIGTERM)
            for _ in range(10):
                if not _pid_alive(pid):
                    return
                time.sleep(0.5)
            os.kill(pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


def _check_env_configured() -> bool:
    env_file = PROJECT_DIR / ".env"
    if not env_file.exists():
        return False
    content = env_file.read_text(encoding="utf-8")
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("LLM_API_KEY="):
            val = line.split("=", 1)[1].strip().strip('"').strip("'")
            if val and val != "sk-xxx":
                return True
    return False


# ═══════════════════════════════════════════
#  命令
# ═══════════════════════════════════════════

def cmd_intro():
    print()
    print("=" * 62)
    print("  智投未来 — A股日内投资AI系统")
    print("=" * 62)
    print()
    print("  基于多AI智能体协作的A股投资决策系统。")
    print("  每天自动扫描全市场 5000+ 只股票，经AI分析后")
    print("  选出最优投资标的，输出标准JSON格式建议。")
    print()
    print("  核心流程:")
    print("    海选筛选 → AI深度分析 → 多空辩论 → 风控决策")
    print()
    print("  第一次使用?")
    print("    1. python manage.py setup          配置API Key")
    print("    2. python manage.py run-demo       演示模式体验")
    print("    3. python manage.py run            正式运行")
    print()
    print("  每日自动化?")
    print("    python manage.py schedule          后台定时运行")
    print()
    print("=" * 62)
    print()


def cmd_setup():
    env_file = PROJECT_DIR / ".env"

    print()
    print("=" * 60)
    print("  智投未来 — 首次配置向导")
    print("=" * 60)
    print()
    print("接下来需要你提供以下信息:")
    print("  - DeepSeek API Key (必填)")
    print("  - Tushare Token (可选，增强数据质量)")
    print()

    if env_file.exists():
        if _check_env_configured():
            print("[!] .env 已配置。重新运行将覆盖现有配置。")
            answer = input("是否继续? (y/N): ").strip().lower()
            if answer != "y":
                print("已取消。")
                return
        else:
            print("[!] .env 已存在但 API Key 未配置，现在进入配置。")
            print()

    print("-" * 60)
    print("1. DeepSeek API Key (必填)")
    print("   获取地址: https://platform.deepseek.com/api_keys")
    print()
    while True:
        api_key = input("   请输入 API Key: ").strip()
        if api_key:
            break
        print("   [!] API Key 不能为空，请重新输入。")

    print()
    print("-" * 60)
    print("2. 模型配置 (默认即可，直接回车跳过)")
    print()
    quick = input(f"   快速模型 (分析师用) [deepseek-chat]: ").strip()
    deep = input(f"   深度模型 (决策主管用) [deepseek-reasoner]: ").strip()
    quick = quick or "deepseek-chat"
    deep = deep or "deepseek-reasoner"

    print()
    print("-" * 60)
    print("3. Tushare Token (可选)")
    print("   获取地址: https://tushare.pro/register")
    print("   不填将仅使用 AKShare + BaoStock，数据质量略低")
    print()
    tushare = input("   请输入 Token (跳过直接回车): ").strip()

    env_content = f"""# ── LLM API ──────────────────────────────
LLM_API_KEY={api_key}
LLM_QUICK_MODEL={quick}
LLM_DEEP_MODEL={deep}
LLM_BASE_URL=https://api.deepseek.com
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=4096

# ── 数据源 ──────────────────────────────
TUSHARE_TOKEN={tushare}
"""
    env_file.write_text(env_content, encoding="utf-8")
    os.environ["LLM_API_KEY"] = api_key
    if tushare:
        os.environ["TUSHARE_TOKEN"] = tushare

    print()
    print("=" * 60)
    print("  配置完成！")
    print(f"  API Key:     {api_key[:8]}...{api_key[-4:]}")
    print(f"  Quick 模型:  {quick}")
    print(f"  Deep 模型:   {deep}")
    print(f"  Tushare:     {'已配置' if tushare else '未配置 (可选)'}")
    print()
    print("  下一步: python manage.py run")
    print("=" * 60)
    print()


def cmd_run():
    if not _check_env_configured():
        print()
        print("[!] 首次运行需要先配置 API Key。")
        answer = input("是否现在进入配置向导? (Y/n): ").strip().lower()
        if answer != "n":
            cmd_setup()
            if not _check_env_configured():
                print("\n[!] 配置未完成，无法运行。请执行 python manage.py setup")
                return
        else:
            print("\n可以稍后执行 python manage.py setup 来配置。")
            print("或使用演示模式: python manage.py run-demo\n")
            return

    _load_env()
    python = _detect_python()

    use_demo = "--demo" in sys.argv or os.getenv("ZHITOU_DEMO", "").lower() in ("1", "true")
    args = [python, "-m", "src.main"]
    if use_demo:
        args.append("--demo")

    mode = "演示模式" if use_demo else "正常模式"
    print()
    print("=" * 62)
    print(f"  智投未来 — {mode}")
    print("=" * 62)
    print(f"  结果目录:  {PROJECT_DIR / 'results'}")
    print("=" * 62)
    print()

    result = subprocess.run(args, cwd=str(PROJECT_DIR))
    if result.returncode != 0:
        print(f"\n[X] 流水线执行失败 (exit code: {result.returncode})")
        sys.exit(result.returncode)
    else:
        print("\n运行完成！")


def cmd_run_demo():
    os.environ["ZHITOU_DEMO"] = "true"
    cmd_run()


def cmd_schedule():
    if not _check_env_configured():
        print()
        print("[!] 首次运行需要先配置 API Key。")
        answer = input("是否现在进入配置向导? (Y/n): ").strip().lower()
        if answer != "n":
            cmd_setup()
            if not _check_env_configured():
                return
        else:
            print("\n可以稍后执行 python manage.py setup 来配置。")
            return

    _load_env()
    python = _detect_python()

    schedule_pid_file = PROJECT_DIR / ".schedule.pid"
    if schedule_pid_file.exists():
        try:
            pid = int(schedule_pid_file.read_text().strip())
            if _pid_alive(pid):
                print(f"调度器已在运行 (PID: {pid})")
                return
        except ValueError:
            pass
        schedule_pid_file.unlink(missing_ok=True)

    print("启动定时调度器 (后台运行)...")
    log_file = LOG_DIR / "schedule.log"
    f_out = open(log_file, "a", encoding="utf-8")

    if PLATFORM == "Windows":
        proc = subprocess.Popen(
            [python, "-m", "src.scheduler"],
            stdout=f_out, stderr=f_out,
            cwd=str(PROJECT_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        proc = subprocess.Popen(
            [python, "-m", "src.scheduler"],
            stdout=f_out, stderr=f_out,
            cwd=str(PROJECT_DIR),
            start_new_session=True,
        )

    schedule_pid_file.write_text(str(proc.pid))
    time.sleep(2)

    if _pid_alive(proc.pid):
        print(f"调度器已启动 (PID: {proc.pid})")
        print(f"日志: {log_file}")
        print("  停止: python manage.py schedule-stop")
    else:
        print("启动失败，查看日志: tail -20 " + str(log_file))
        schedule_pid_file.unlink(missing_ok=True)


def cmd_schedule_stop():
    schedule_pid_file = PROJECT_DIR / ".schedule.pid"
    if not schedule_pid_file.exists():
        print("调度器未运行")
        return
    try:
        pid = int(schedule_pid_file.read_text().strip())
    except ValueError:
        schedule_pid_file.unlink(missing_ok=True)
        return
    if _pid_alive(pid):
        print(f"停止调度器 (PID: {pid})...")
        _kill_process(pid)
        print("调度器已停止")
    else:
        print("PID 文件存在但进程不存在，清理中...")
    schedule_pid_file.unlink(missing_ok=True)


def cmd_run_scheduled():
    _load_env()
    try:
        from src.utils.trading_calendar import is_trading_day
        if not is_trading_day(datetime.now()):
            print(f"今日 ({datetime.now().strftime('%Y-%m-%d')}) 非交易日，跳过。")
            return
    except Exception:
        pass
    cmd_run()


def cmd_schedule_demo():
    _load_env()
    python = _detect_python()

    print("以演示模式启动定时调度器...")
    log_file = LOG_DIR / "schedule.log"
    f_out = open(log_file, "a", encoding="utf-8")

    if PLATFORM == "Windows":
        proc = subprocess.Popen(
            [python, "-m", "src.scheduler"],
            stdout=f_out, stderr=f_out,
            cwd=str(PROJECT_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            env={**os.environ, "ZHITOU_DEMO": "true"},
        )
    else:
        proc = subprocess.Popen(
            [python, "-m", "src.scheduler"],
            stdout=f_out, stderr=f_out,
            cwd=str(PROJECT_DIR),
            start_new_session=True,
            env={**os.environ, "ZHITOU_DEMO": "true"},
        )

    schedule_pid_file = PROJECT_DIR / ".schedule.pid"
    schedule_pid_file.write_text(str(proc.pid))
    time.sleep(2)

    print(f"演示调度器已启动 (PID: {proc.pid})")
    print(f"  日志: {log_file}")
    print(f"  停止: python manage.py schedule-stop")


COMMANDS = {
    "intro": cmd_intro,
    "setup": cmd_setup,
    "run": cmd_run,
    "run-demo": cmd_run_demo,
    "run-scheduled": cmd_run_scheduled,
    "schedule": cmd_schedule,
    "schedule-stop": cmd_schedule_stop,
    "schedule-demo": cmd_schedule_demo,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智投未来 管理脚本")
    parser.add_argument("command", nargs="?", choices=list(COMMANDS), help="操作命令")
    try:
        args = parser.parse_args()
    except SystemExit as e:
        if e.code != 0:
            args = None
        else:
            raise
    if args is None or args.command is None:
        print("请指定一个命令，例如: python manage.py intro")
        print()
        parser.print_help()
    else:
        COMMANDS[args.command]()

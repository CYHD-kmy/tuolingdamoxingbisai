#!/usr/bin/env python3
"""
智投未来 — 跨平台服务管理脚本 (Windows / macOS / Linux)

Usage: python manage.py {start|stop|restart|status|install|uninstall}

兼容性:
  - Windows: 不支持 install/uninstall (需手动配置)
  - macOS: install → launchd 开机自启
  - Linux: install → systemd 服务
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
sys.path.insert(0, str(PROJECT_DIR))  # 确保 src 模块可导入
PID_FILE = PROJECT_DIR / ".server.pid"
TUNNEL_PID_FILE = PROJECT_DIR / ".tunnel.pid"
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

HOST = os.getenv("ZHITOU_HOST", "0.0.0.0")
PORT = os.getenv("ZHITOU_PORT", "8000")
LOG_LEVEL = os.getenv("ZHITOU_LOG_LEVEL", "info")

PLATFORM = platform.system()  # "Darwin" / "Linux" / "Windows"


def _load_env():
    """加载 .env 文件 (处理引号、转义和内联注释)"""
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
            # 移除内联注释 (仅在不处于引号内时)
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
    """自动检测 Python 解释器"""
    for candidate in [
        PROJECT_DIR / ".venv" / "bin" / "python3",
        PROJECT_DIR / ".venv" / "bin" / "python",
        PROJECT_DIR / "venv" / "bin" / "python3",
        PROJECT_DIR / "venv" / "bin" / "python",
        PROJECT_DIR / ".venv" / "Scripts" / "python.exe",   # Windows venv
        PROJECT_DIR / "venv" / "Scripts" / "python.exe",     # Windows venv
    ]:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _pid_alive(pid: int) -> bool:
    """跨平台检查进程是否存活"""
    try:
        if PLATFORM == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x0400, False, pid)  # PROCESS_QUERY_INFORMATION
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
    """跨平台终止进程"""
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


# ═══════════════════════════════════════════
#  Commands
# ═══════════════════════════════════════════

def cmd_start():
    """后台启动服务"""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            if _pid_alive(pid):
                print(f"服务已在运行 (PID: {pid})")
                print(f"访问: http://{HOST}:{PORT}")
                return
        except ValueError:
            pass

    _load_env()
    python = _detect_python()

    log_file = LOG_DIR / "server.log"
    f_out = open(log_file, "a", encoding="utf-8")

    print(f"启动智投未来网站服务... (http://{HOST}:{PORT})")

    if PLATFORM == "Windows":
        # Windows: 使用 CREATE_NEW_PROCESS_GROUP 实现后台运行
        proc = subprocess.Popen(
            [python, "-m", "uvicorn", "src.api.server:app",
             "--host", HOST, "--port", PORT, "--log-level", LOG_LEVEL],
            stdout=f_out, stderr=f_out,
            cwd=str(PROJECT_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        proc = subprocess.Popen(
            [python, "-m", "uvicorn", "src.api.server:app",
             "--host", HOST, "--port", PORT, "--log-level", LOG_LEVEL],
            stdout=f_out, stderr=f_out,
            cwd=str(PROJECT_DIR),
            start_new_session=True,
        )

    PID_FILE.write_text(str(proc.pid))
    time.sleep(2)

    if _pid_alive(proc.pid):
        print(f"服务已启动 (PID: {proc.pid})")
        print(f"访问: http://{HOST}:{PORT}")
        print(f"日志: {log_file}")
    else:
        print("启动失败，查看日志: tail -50 " + str(log_file))
        PID_FILE.unlink(missing_ok=True)


def cmd_stop():
    """停止服务"""
    if not PID_FILE.exists():
        print("服务未运行 (无 PID 文件)")
        return

    try:
        pid = int(PID_FILE.read_text().strip())
    except ValueError:
        print("PID 文件损坏，清理中...")
        PID_FILE.unlink(missing_ok=True)
        return

    if _pid_alive(pid):
        print(f"停止服务 (PID: {pid})...")
        _kill_process(pid)
        print("服务已停止")
    else:
        print("PID 文件存在但进程不存在，清理中...")

    PID_FILE.unlink(missing_ok=True)


def cmd_restart():
    """重启服务"""
    cmd_stop()
    time.sleep(1)
    cmd_start()


def cmd_status():
    """查看服务状态"""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            if _pid_alive(pid):
                print(f"服务运行中 (PID: {pid})")
                print(f"访问: http://{HOST}:{PORT}")
                return
        except ValueError:
            pass
        PID_FILE.unlink(missing_ok=True)
    print("服务未运行")


def cmd_install():
    """安装为系统服务 (开机自启)"""
    if PLATFORM == "Darwin":
        _install_macos()
    elif PLATFORM == "Linux":
        _install_linux()
    else:
        _install_windows_guide()


def cmd_uninstall():
    """卸载系统服务"""
    if PLATFORM == "Darwin":
        _uninstall_macos()
    elif PLATFORM == "Linux":
        _uninstall_linux()
    else:
        _uninstall_windows_guide()


# ═══════════════════════════════════════════
#  macOS launchd
# ═══════════════════════════════════════════

def _install_macos():
    import plistlib

    python = _detect_python()
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.zhitoufuture.dashboard.plist"

    wrapper = PROJECT_DIR / ".launchd_wrapper.sh"
    wrapper.write_text(f'''#!/bin/bash
cd "{PROJECT_DIR}"
[ -f "{PROJECT_DIR}/.env" ] && set -a && source "{PROJECT_DIR}/.env" && set +a
exec {python} -m uvicorn src.api.server:app --host "{HOST}" --port "{PORT}" --log-level "{LOG_LEVEL}"
''', encoding="utf-8")
    wrapper.chmod(0o755)

    plist = {
        "Label": "com.zhitoufuture.dashboard",
        "ProgramArguments": ["/bin/bash", str(wrapper)],
        "WorkingDirectory": str(PROJECT_DIR),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(LOG_DIR / "launchd-stdout.log"),
        "StandardErrorPath": str(LOG_DIR / "launchd-stderr.log"),
        "ProcessType": "Background",
        "Nice": 5,
    }

    plist_path.write_bytes(plistlib.dumps(plist))
    # 先卸载再加载，避免重复加载报错
    subprocess.run(["launchctl", "unload", str(plist_path)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["launchctl", "load", str(plist_path)], check=True)

    print("已安装 launchd 开机自启服务！")
    print(f"  服务名: com.zhitoufuture.dashboard")
    print(f"  管理:   launchctl start/stop com.zhitoufuture.dashboard")
    print(f"  卸载:   python manage.py uninstall")


def _uninstall_macos():
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.zhitoufuture.dashboard.plist"
    wrapper = PROJECT_DIR / ".launchd_wrapper.sh"

    subprocess.run(["launchctl", "unload", str(plist_path)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    plist_path.unlink(missing_ok=True)
    wrapper.unlink(missing_ok=True)
    print("已卸载 launchd 服务")


# ═══════════════════════════════════════════
#  Linux systemd
# ═══════════════════════════════════════════

_SYSTEMD_TEMPLATE = """[Unit]
Description=智投未来 网站看板
After=network.target

[Service]
Type=simple
WorkingDirectory={project_dir}
ExecStart={python} -m uvicorn src.api.server:app --host {host} --port {port} --log-level {log_level}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""


def _install_linux():
    python = _detect_python()
    unit_content = _SYSTEMD_TEMPLATE.format(
        project_dir=PROJECT_DIR,
        python=python,
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL,
    )

    unit_path = Path.home() / ".config" / "systemd" / "user" / "zhitou-future.service"
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(unit_content, encoding="utf-8")

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", "zhitou-future.service"], check=True)

    print("已安装 systemd 用户服务！")
    print(f"  管理: systemctl --user start/stop/restart zhitou-future")
    print(f"  日志: journalctl --user -u zhitou-future -f")
    print(f"  卸载: python manage.py uninstall")


def _uninstall_linux():
    subprocess.run(["systemctl", "--user", "stop", "zhitou-future.service"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["systemctl", "--user", "disable", "zhitou-future.service"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    unit_path = Path.home() / ".config" / "systemd" / "user" / "zhitou-future.service"
    unit_path.unlink(missing_ok=True)
    subprocess.run(["systemctl", "--user", "daemon-reload"])
    print("已卸载 systemd 服务")


# ═══════════════════════════════════════════
#  Windows 提示
# ═══════════════════════════════════════════

def _install_windows_guide():
    print("Windows 不支持自动安装系统服务。建议使用以下方式:")
    print()
    print("方式一: 手动启动 (开机后执行)")
    print(f"  python manage.py start")
    print()
    print("方式二: Windows 任务计划程序")
    print("  1. 打开 '任务计划程序' (taskschd.msc)")
    print("  2. 创建基本任务 → 触发器: 登录时")
    print("  3. 操作: 启动程序")
    print(f"     程序: {_detect_python()}")
    print(f"     参数: -m uvicorn src.api.server:app --host {HOST} --port {PORT}")
    print(f"     起始于: {PROJECT_DIR}")
    print()
    print("方式三: Docker 部署 (推荐)")
    print("  见 DEPLOY.md")


def _uninstall_windows_guide():
    print("Windows: 如使用了任务计划程序，请在 taskschd.msc 中删除对应任务。")


# ═══════════════════════════════════════════
#  公网隧道 (serveo.net)
# ═══════════════════════════════════════════

def cmd_tunnel():
    """启动公网隧道 (通过 serveo.net 免费服务)"""
    if TUNNEL_PID_FILE.exists():
        try:
            pid = int(TUNNEL_PID_FILE.read_text().strip())
            if _pid_alive(pid):
                _show_tunnel_url()
                return
        except ValueError:
            pass

    print("正在创建公网隧道...")
    log_file = LOG_DIR / "tunnel.log"
    f_out = open(log_file, "a", encoding="utf-8")

    if PLATFORM == "Windows":
        proc = subprocess.Popen(
            ["ssh", "-o", "StrictHostKeyChecking=no",
             "-o", "ServerAliveInterval=60", "-o", "ServerAliveCountMax=3",
             "-R", f"80:localhost:{PORT}", "serveo.net"],
            stdout=f_out, stderr=f_out,
            cwd=str(PROJECT_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        proc = subprocess.Popen(
            ["ssh", "-o", "StrictHostKeyChecking=no",
             "-o", "ServerAliveInterval=60", "-o", "ServerAliveCountMax=3",
             "-R", f"80:localhost:{PORT}", "serveo.net"],
            stdout=f_out, stderr=f_out,
            cwd=str(PROJECT_DIR),
            start_new_session=True,
        )

    TUNNEL_PID_FILE.write_text(str(proc.pid))
    time.sleep(4)

    if _pid_alive(proc.pid):
        _show_tunnel_url()
    else:
        print("隧道启动失败，查看日志: tail -20 " + str(log_file))
        TUNNEL_PID_FILE.unlink(missing_ok=True)


def cmd_tunnel_stop():
    """关闭公网隧道"""
    if not TUNNEL_PID_FILE.exists():
        print("隧道未运行")
        return
    try:
        pid = int(TUNNEL_PID_FILE.read_text().strip())
    except ValueError:
        TUNNEL_PID_FILE.unlink(missing_ok=True)
        return

    if _pid_alive(pid):
        print(f"关闭隧道 (PID: {pid})...")
        _kill_process(pid)
    TUNNEL_PID_FILE.unlink(missing_ok=True)
    print("隧道已关闭")


def _show_tunnel_url():
    """从日志解析并显示公网 URL"""
    log_file = LOG_DIR / "tunnel.log"
    if log_file.exists():
        content = log_file.read_text(encoding="utf-8")
        for line in content.split("\n"):
            if "Forwarding HTTP traffic from" in line:
                url = line.split("from ")[-1].strip()
                print(f"公网地址: {url}")
                return
    print("公网隧道已启动，查看日志获取地址: tail -5 " + str(log_file))


# ═══════════════════════════════════════════
#  流水线运行 & 定时调度 (跨平台)
# ═══════════════════════════════════════════

def _check_env_configured() -> bool:
    """检查 .env 是否已配置 (有有效的 API Key)"""
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


def cmd_intro():
    """展示项目概览"""
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
    print("  查看结果?")
    print("    python manage.py start            启动Web看板")
    print("    → 浏览器打开 http://localhost:8000")
    print()
    print("  详细说明: 运行指南.md")
    print("=" * 62)
    print()


def cmd_setup():
    """交互式配置向导: 引导用户填写 API Key 和 Token"""
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

    # ── 1. DeepSeek API Key ──
    print("─" * 60)
    print("1. DeepSeek API Key (必填)")
    print("   获取地址: https://platform.deepseek.com/api_keys")
    print()
    while True:
        api_key = input("   请输入 API Key: ").strip()
        if api_key:
            break
        print("   [!] API Key 不能为空，请重新输入。")

    # ── 2. 模型选择 ──
    print()
    print("─" * 60)
    print("2. 模型配置 (默认即可，直接回车跳过)")
    print()
    quick = input(f"   快速模型 (分析师用) [deepseek-chat]: ").strip()
    deep = input(f"   深度模型 (决策主管用) [deepseek-reasoner]: ").strip()

    quick = quick or "deepseek-chat"
    deep = deep or "deepseek-reasoner"

    # ── 3. Tushare Token ──
    print()
    print("─" * 60)
    print("3. Tushare Token (可选)")
    print("   获取地址: https://tushare.pro/register")
    print("   不填将仅使用 AKShare + BaoStock，数据质量略低")
    print()
    tushare = input("   请输入 Token (跳过直接回车): ").strip()

    # ── 写入 .env ──
    env_content = f"""# ── LLM API ──────────────────────────────
LLM_API_KEY={api_key}
LLM_QUICK_MODEL={quick}
LLM_DEEP_MODEL={deep}
LLM_BASE_URL=https://api.deepseek.com
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=4096

# ── 数据源 ──────────────────────────────
TUSHARE_TOKEN={tushare}

# ── 网站服务 ────────────────────────────
ZHITOU_HOST=0.0.0.0
ZHITOU_PORT=8000
ZHITOU_LOG_LEVEL=info
ZHITOU_ACCESS_LOG=true
"""
    env_file.write_text(env_content, encoding="utf-8")
    os.environ["LLM_API_KEY"] = api_key
    if tushare:
        os.environ["TUSHARE_TOKEN"] = tushare

    print()
    print("=" * 60)
    print("  配置完成！")
    print()
    print(f"  API Key:     {api_key[:8]}...{api_key[-4:]}")
    print(f"  Quick 模型:  {quick}")
    print(f"  Deep 模型:   {deep}")
    print(f"  Tushare:     {'已配置' if tushare else '未配置 (可选)'}")
    print()
    print("  下一步:")
    print("    python manage.py run          # 立即执行一次")
    print("    python manage.py schedule     # 后台每日自动运行")
    print("=" * 60)
    print()


def cmd_run():
    """运行一次完整流水线 (前台执行)"""
    # 预检查: .env 未配置时引导用户进入 setup
    if not _check_env_configured():
        print()
        print("[!] 首次运行需要先配置 API Key。")
        print()
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
    api_key = os.getenv("LLM_API_KEY", "")
    tushare = os.getenv("TUSHARE_TOKEN", "")

    print()
    print("=" * 62)
    print("  智投未来 — A股投资决策流水线")
    print("=" * 62)
    print(f"  模式:      {mode}")
    if not use_demo:
        print(f"  API Key:   {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '****'}")
        print(f"  Tushare:   {'已配置' if tushare else '未配置 (使用AKShare)'}")
    print(f"  结果目录:  {PROJECT_DIR / 'results'}")
    print("=" * 62)
    print()
    print("流水线将按以下顺序执行:")
    print()
    print("  ① 海选筛选 ── 5000+ 只股票 → Top 20 候选")
    print("  ② AI 分析  ── 4 位AI分析师并行研判")
    print("  ③ 多空辩论 ── 正反双方辩论对抗")
    print("  ④ 风控决策 ── 仓位约束 + 最终买卖建议")
    print()
    print("-" * 62)
    print()
    sys.stdout.flush()

    result = subprocess.run(args, cwd=str(PROJECT_DIR))
    if result.returncode != 0:
        print(f"\n[X] 流水线执行失败 (exit code: {result.returncode})")
        print(f"    查看日志: tail -50 {LOG_DIR / 'schedule.log'}")
        sys.exit(result.returncode)
    else:
        print()
        print("-" * 62)
        print("  运行完成！")
        print()
        print("  查看结果:")
        print(f"    python manage.py start        → Web 看板")
        print(f"    ls results/trace_*.json       → 决策轨迹文件")
        print("=" * 62)
        print()


def cmd_run_demo():
    """运行一次完整流水线 (演示模式)"""
    os.environ["ZHITOU_DEMO"] = "true"
    cmd_run()


def cmd_schedule():
    """启动定时调度器 (后台运行，每日自动触发)"""
    if not _check_env_configured():
        print()
        print("[!] 首次运行需要先配置 API Key。")
        print()
        answer = input("是否现在进入配置向导? (Y/n): ").strip().lower()
        if answer != "n":
            cmd_setup()
            if not _check_env_configured():
                print("\n[!] 配置未完成，无法启动调度器。")
                return
        else:
            print("\n可以稍后执行 python manage.py setup 来配置。")
            print("或使用演示调度器: python manage.py schedule-demo\n")
            return

    _load_env()
    python = _detect_python()

    # 检查是否已在运行
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
        from src.utils.trading_calendar import is_trading_day
        today = datetime.now().strftime("%Y-%m-%d")
        is_td = "交易日" if is_trading_day(datetime.now()) else "非交易日"
        print(f"调度器已启动 (PID: {proc.pid})")
        print(f"今日 ({today}): {is_td}  |  日志: {log_file}")
        print("  停止: python manage.py schedule-stop")
    else:
        print("启动失败，查看日志: tail -20 " + str(log_file))
        schedule_pid_file.unlink(missing_ok=True)


def cmd_schedule_stop():
    """停止定时调度器"""
    schedule_pid_file = PROJECT_DIR / ".schedule.pid"
    if not schedule_pid_file.exists():
        print("调度器未运行")
        return

    try:
        pid = int(schedule_pid_file.read_text().strip())
    except ValueError:
        print("PID 文件损坏，清理中...")
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
    """一键启动: 若今日为交易日则执行，否则跳过 (前台执行)"""
    _load_env()
    python = _detect_python()

    try:
        from src.utils.trading_calendar import is_trading_day
        from datetime import datetime
        if not is_trading_day(datetime.now()):
            print(f"今日 ({datetime.now().strftime('%Y-%m-%d')}) 非交易日，跳过。")
            return
    except Exception:
        pass  # 交易日历不可用时默认执行

    cmd_run()


def cmd_schedule_demo():
    """启动调度器 + 演示模式 (确保无网络也能跑)"""
    _load_env()
    python = _detect_python()

    print("以演示模式启动定时调度器...")
    log_file = LOG_DIR / "schedule.log"
    f_out = open(log_file, "a", encoding="utf-8")

    # 直接用 --demo 模式运行 main 一次，然后启动调度器
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


# ═══════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════

COMMANDS = {
    "start": cmd_start,
    "stop": cmd_stop,
    "restart": cmd_restart,
    "status": cmd_status,
    "install": cmd_install,
    "uninstall": cmd_uninstall,
    "tunnel": cmd_tunnel,
    "tunnel-stop": cmd_tunnel_stop,
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
    parser = argparse.ArgumentParser(description="智投未来 服务管理")
    parser.add_argument("command", choices=list(COMMANDS), help="操作命令")
    args = parser.parse_args()

    COMMANDS[args.command]()

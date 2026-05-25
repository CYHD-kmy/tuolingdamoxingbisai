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
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PID_FILE = PROJECT_DIR / ".server.pid"
TUNNEL_PID_FILE = PROJECT_DIR / ".tunnel.pid"
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

HOST = os.getenv("ZHITOU_HOST", "0.0.0.0")
PORT = os.getenv("ZHITOU_PORT", "8000")
LOG_LEVEL = os.getenv("ZHITOU_LOG_LEVEL", "info")

PLATFORM = platform.system()  # "Darwin" / "Linux" / "Windows"


def _load_env():
    """加载 .env 文件"""
    env_file = PROJECT_DIR / ".env"
    if not env_file.exists():
        return
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
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
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智投未来 服务管理")
    parser.add_argument("command", choices=list(COMMANDS), help="操作命令")
    args = parser.parse_args()

    COMMANDS[args.command]()

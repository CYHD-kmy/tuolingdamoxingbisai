#!/bin/bash
# 智投未来 — 网站服务管理脚本
# Usage: ./manage.sh {start|stop|restart|status|install|uninstall}

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$PROJECT_DIR/.server.pid"
LOG_DIR="$PROJECT_DIR/logs"
PLIST_NAME="com.zhitoufuture.dashboard.plist"
PLIST_SRC="$PROJECT_DIR/$PLIST_NAME"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"

mkdir -p "$LOG_DIR"

# 加载 .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# 自动检测 Python (优先使用有 uvicorn 的)
_detect_python() {
    for candidate in \
        "$PROJECT_DIR/.venv/bin/python3" \
        "$PROJECT_DIR/.venv/bin/python" \
        "$PROJECT_DIR/venv/bin/python3" \
        "$PROJECT_DIR/venv/bin/python" \
        "/opt/anaconda3/bin/python3" \
        "/usr/local/bin/python3" \
        "$(which python3 2>/dev/null)" \
        "$(which python 2>/dev/null)"; do
        if [ -x "$candidate" ] && "$candidate" -c "import uvicorn" 2>/dev/null; then
            echo "$candidate"
            return 0
        fi
    done
    # 兜底: 返回系统 python3
    which python3 2>/dev/null || which python
}
PYTHON="$(_detect_python)"

HOST="${ZHITOU_HOST:-0.0.0.0}"
PORT="${ZHITOU_PORT:-8000}"
LOG_LEVEL="${ZHITOU_LOG_LEVEL:-info}"
ACCESS_LOG="${ZHITOU_ACCESS_LOG:-true}"

UVICORN_ARGS="--host $HOST --port $PORT --log-level $LOG_LEVEL"
if [ "$ACCESS_LOG" = "true" ]; then
    UVICORN_ARGS="$UVICORN_ARGS --access-log"
fi

start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "服务已在运行 (PID: $(cat "$PID_FILE"))"
        echo "访问: http://${HOST}:${PORT}"
        return 0
    fi

    echo "启动智投未来网站服务..."
    cd "$PROJECT_DIR"

    nohup "$PYTHON" -m uvicorn src.api.server:app $UVICORN_ARGS \
        >> "$LOG_DIR/server.log" 2>&1 &
    echo $! > "$PID_FILE"

    sleep 2
    if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "服务已启动 (PID: $(cat "$PID_FILE"))"
        echo "访问: http://${HOST}:${PORT}"
        echo "日志: $LOG_DIR/server.log"
    else
        echo "启动失败，查看日志: tail -50 $LOG_DIR/server.log"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "服务未运行 (无 PID 文件)"
        return 0
    fi
    PID="$(cat "$PID_FILE")"
    if kill -0 "$PID" 2>/dev/null; then
        echo "停止服务 (PID: $PID)..."
        kill "$PID"
        for i in {1..10}; do
            if ! kill -0 "$PID" 2>/dev/null; then break; fi
            sleep 0.5
        done
        if kill -0 "$PID" 2>/dev/null; then
            echo "强制终止..."
            kill -9 "$PID"
        fi
        echo "服务已停止"
    else
        echo "PID 文件存在但进程不存在，清理中..."
    fi
    rm -f "$PID_FILE"
}

restart() {
    stop
    sleep 1
    start
}

status() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        PID="$(cat "$PID_FILE")"
        echo "服务运行中 (PID: $PID)"
        echo "访问: http://${HOST}:${PORT}"
    else
        echo "服务未运行"
        [ -f "$PID_FILE" ] && rm -f "$PID_FILE"
    fi
}

install() {
    echo "安装 launchd 开机自启服务..."

    # 获取 Python 绝对路径
    PYTHON_ABS="$("$PYTHON" -c 'import sys; print(sys.executable)')"

    # 生成 plist 文件
    cat > "$PLIST_DST" << PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.zhitoufuture.dashboard</string>

    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON_ABS</string>
        <string>-m</string>
        <string>uvicorn</string>
        <string>src.api.server:app</string>
        <string>--host</string>
        <string>$HOST</string>
        <string>--port</string>
        <string>$PORT</string>
        <string>--log-level</string>
        <string>$LOG_LEVEL</string>
    </array>

    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>$LOG_DIR/launchd-stdout.log</string>

    <key>StandardErrorPath</key>
    <string>$LOG_DIR/launchd-stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin</string>
    </dict>

    <key>ProcessType</key>
    <string>Background</string>

    <key>Nice</key>
    <integer>5</integer>
</dict>
</plist>
PLIST_EOF

    # 如果存在 .env，注入环境变量到 plist
    if [ -f "$PROJECT_DIR/.env" ]; then
        echo "检测到 .env 文件，将环境变量注入 plist..."
        # 这是一个简化实现：在 WorkingDirectory 下执行前 source .env
        # 更稳健的做法是用 shell 包装启动命令
        _install_with_env
    else
        launchctl load "$PLIST_DST"
    fi

    echo ""
    echo "已安装 launchd 服务！"
    echo "  - 服务名: com.zhitoufuture.dashboard"
    echo "  - 开机自启: 已启用"
    echo "  - 崩溃重启: 已启用"
    echo ""
    echo "管理命令:"
    echo "  launchctl start com.zhitoufuture.dashboard   # 启动"
    echo "  launchctl stop com.zhitoufuture.dashboard    # 停止"
    echo "  launchctl list | grep zhito                  # 查看状态"
    echo "  ./manage.sh uninstall                        # 卸载"
}

_install_with_env() {
    # 使用 shell 包装来加载 .env 环境变量
    local WRAPPER="$PROJECT_DIR/.launchd_wrapper.sh"
    cat > "$WRAPPER" << 'WRAPPER_EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi
exec python3 -m uvicorn src.api.server:app --host "${ZHITOU_HOST:-0.0.0.0}" --port "${ZHITOU_PORT:-8000}" --log-level "${ZHITOU_LOG_LEVEL:-info}"
WRAPPER_EOF
    chmod +x "$WRAPPER"

    # 重写 plist 使用 wrapper
    cat > "$PLIST_DST" << PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.zhitoufuture.dashboard</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$WRAPPER</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/launchd-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/launchd-stderr.log</string>
    <key>ProcessType</key>
    <string>Background</string>
    <key>Nice</key>
    <integer>5</integer>
</dict>
</plist>
PLIST_EOF

    launchctl load "$PLIST_DST"
}

uninstall() {
    echo "卸载 launchd 服务..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    rm -f "$PLIST_DST"
    rm -f "$PROJECT_DIR/.launchd_wrapper.sh"
    echo "已卸载"
}

case "${1:-}" in
    start)     start ;;
    stop)      stop ;;
    restart)   restart ;;
    status)    status ;;
    install)   install ;;
    uninstall) uninstall ;;
    *)
        echo "智投未来 — 服务管理脚本"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|install|uninstall}"
        echo ""
        echo "  start      启动服务 (后台运行)"
        echo "  stop       停止服务"
        echo "  restart    重启服务"
        echo "  status     查看服务状态"
        echo "  install    安装为 launchd 服务 (开机自启)"
        echo "  uninstall  卸载 launchd 服务"
        echo ""
        exit 1
        ;;
esac

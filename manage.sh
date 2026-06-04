#!/bin/bash
# 智投未来 — 快速启动脚本
# Usage: ./manage.sh [run|run-demo|schedule|schedule-stop]

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# 加载 .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

_detect_python() {
    for candidate in \
        "$PROJECT_DIR/.venv/bin/python3" \
        "$PROJECT_DIR/.venv/bin/python" \
        "$PROJECT_DIR/venv/bin/python3" \
        "$PROJECT_DIR/venv/bin/python" \
        "$(command -v python3 2>/dev/null)" \
        "$(command -v python 2>/dev/null)"; do
        if [ -x "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    which python3 2>/dev/null || which python
}
PYTHON="$(_detect_python)"

case "${1:-}" in
    run)
        exec "$PYTHON" -m src.main
        ;;
    run-demo)
        exec "$PYTHON" -m src.main --demo
        ;;
    schedule)
        exec "$PYTHON" -m src.scheduler
        ;;
    *)
        echo "智投未来 — 管理脚本"
        echo ""
        echo "Usage: $0 {run|run-demo|schedule}"
        echo ""
        echo "  run        运行流水线"
        echo "  run-demo   演示模式运行"
        echo "  schedule   启动定时调度器"
        echo ""
        exit 1
        ;;
esac

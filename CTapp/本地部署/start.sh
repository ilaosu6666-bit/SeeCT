#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "========================================"
echo "  智影溯源 - AI肺结节教学平台"
echo "  离线版本"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 Python3，请先安装 Python 3.10+"
    exit 1
fi

# Setup venv if not exists
if [ ! -d ".venv" ]; then
    echo "[首次运行] 正在创建虚拟环境..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install dependencies if needed
python -c "import streamlit" >/dev/null 2>&1 || {
    echo "[首次运行] 正在安装依赖（可能需要几分钟）..."
    pip install -r requirements.txt -q
}

echo ""
echo "正在启动服务..."
echo "浏览器将自动打开，或手动访问：http://localhost:8501"
echo "按 Ctrl+C 停止服务"
echo ""

streamlit run app.py --server.port 8501

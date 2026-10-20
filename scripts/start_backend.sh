#!/usr/bin/env bash
# 启动后端：在 backend/ 目录下运行 uvicorn
set -e
cd "$(dirname "$0")/../backend"
python -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"

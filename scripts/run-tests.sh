#!/bin/sh
# 容器内测试入口：仅在能连通 Postgres 时运行端到端 DB 测试
set -e
export PYTHONUNBUFFERED=1

if [ -n "$DATABASE_URL" ]; then
  export RUN_DB_TESTS=1
fi
exec python -m pytest -q "$@"

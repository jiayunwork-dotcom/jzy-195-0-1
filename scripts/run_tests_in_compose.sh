#!/bin/sh
# 在 compose 网络中准备测试库并运行全部 pytest（引擎 + API/持久化）。
# 用法：
#   docker compose up -d db
#   ./scripts/run_tests_in_compose.sh
set -eu

docker compose exec -T db psql -U panel -d panelapp -tc \
  "SELECT 1 FROM pg_database WHERE datname='panelapp_test'" | grep -q 1 \
  || docker compose exec -T db createdb -U panel panelapp_test

docker compose build api
docker compose run --rm \
  -e TEST_DATABASE_URL=postgresql://panel:panel@db:5432/panelapp_test \
  api sh -c "pip install --no-cache-dir -r requirements-dev.txt && pytest -v"

"""测试夹具：

* 引擎测试（tests/test_did_engine.py）完全不依赖数据库/HTTP，直接调用 app.did；
* API 集成测试（tests/test_api.py）需要 PostgreSQL，连接串由 TEST_DATABASE_URL
  指定；未提供或连不上时自动跳过。

在 compose 内运行：
    docker compose run --rm \
      -e TEST_DATABASE_URL=postgresql://panel:panel@db:5432/panelapp_test \
      api sh -c "pip install --no-cache-dir -r requirements-dev.txt && pytest"
（需要先在 db 上创建测试库：
    docker compose exec db createdb -U panel panelapp_test ）
"""
from __future__ import annotations

import os

import pytest

from app import storage


@pytest.fixture(scope="session")
def pool_initialized():
    dsn = os.environ.get(
        "TEST_DATABASE_URL", "postgresql://panel:panel@localhost:5432/panelapp_test"
    )
    try:
        storage.init_pool(dsn)
        storage.init_db()
    except Exception:  # noqa: BLE001
        pytest.skip("PostgreSQL 不可用（设置 TEST_DATABASE_URL 后运行 API 集成测试）")
    yield True


@pytest.fixture()
def client(pool_initialized):
    # 每个测试前清空业务表，保证相互独立、可重复运行。
    from fastapi.testclient import TestClient

    from app.main import app

    with storage.get_pool().conn() as conn:
        conn.execute(
            "TRUNCATE estimates, version_changes, panel_rows, "
            "dataset_versions, datasets RESTART IDENTITY CASCADE"
        )
        conn.commit()

    with TestClient(app) as c:
        yield c

import os
import sys

# 容器内依赖装在 site-packages；本地沙箱里装在 /tmp/pylibs
_EXTRA = "/tmp/pylibs"
if os.path.isdir(_EXTRA) and _EXTRA not in sys.path:
    sys.path.insert(0, _EXTRA)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_collection_modifyitems(config, items):
    """无 Postgres 时把 DB 测试标记为 skip（而非 collection 失败）。"""
    db_available = os.environ.get("RUN_DB_TESTS") == "1"
    if db_available:
        return
    skip_db = __import__("pytest").mark.skip(
        reason="需要 Postgres：设置 RUN_DB_TESTS=1 与 DATABASE_URL"
    )
    # test_api 整体依赖数据库
    for item in items:
        if "test_api" in str(item.fspath):
            item.add_marker(skip_db)

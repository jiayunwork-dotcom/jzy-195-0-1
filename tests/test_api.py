"""端到端 API 测试：需要 Postgres（DATABASE_URL 指向测试库）。

本地无数据库时自动跳过；在 compose 环境中通过 docker compose 运行。
"""

from __future__ import annotations

import json
import os
import random

import pytest

psycopg = pytest.importorskip("psycopg")
from fastapi.testclient import TestClient  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DB_TESTS") != "1",
    reason="需要 Postgres：设置 RUN_DB_TESTS=1 与 DATABASE_URL",
)


@pytest.fixture(scope="session")
def client():
    from app import db
    from app.main import app

    # 建表并清空（每个会话使用干净库）
    db.init_db()
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE results, analyses, observations, version_changes, "
                "dataset_versions, datasets RESTART IDENTITY CASCADE"
            )
    with TestClient(app) as c:
        yield c


def panel_payload(rows):
    return [
        {
            "unit": u,
            "period": int(t),
            "outcome": float(y),
            "treated": bool(d),
            "covariates": {"x": float(i % 3)},
        }
        for (u, t, y, d, i) in rows
    ]


def make_2x2_rows(prefix="T", shift=0.0, k=1.0):
    rows = []
    for i in range(25):
        rows.append((f"{prefix}A{i}", 0, (10.0 + shift) * k, False, i))
        rows.append((f"{prefix}A{i}", 1, (15.0 + shift) * k, True, i))
        rows.append((f"{prefix}B{i}", 0, (8.0 + shift) * k, False, i))
        rows.append((f"{prefix}B{i}", 1, (10.0 + shift) * k, False, i))
    return panel_payload(rows)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_full_flow_two_by_two_and_caching_and_immutability(client):
    # 1. 上传 v1
    r = client.post("/api/datasets", json={
        "name": "教科书2x2",
        "description": "处理效应应为3",
        "rows": make_2x2_rows(),
    })
    assert r.status_code == 201, r.text
    v1 = r.json()
    ds_id = v1["dataset_id"]
    assert v1["version"] == 1 and v1["n_rows"] == 100

    # 2. 提交分析
    r = client.post(f"/api/versions/{v1['version_id']}/analyses",
                    json={"name": "main", "control_strategy": "never_treated"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["cached"] is False
    assert body["result"]["att"] == pytest.approx(3.0, abs=1e-8)
    result_id = body["result_id"]

    # 3. 同设定重复提交 -> 缓存（200，cached=True）
    r2 = client.post(f"/api/versions/{v1['version_id']}/analyses",
                     json={"name": "main", "control_strategy": "never_treated"})
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["cached"] is True
    assert body2["result_id"] == result_id

    # 4. 结果里同时给了 TWFE（2x2 下同样是 3）
    assert body["result"]["twfe"]["coef_static"] == pytest.approx(3.0, abs=1e-8)

    # 5. 取回完整内容
    r = client.get(f"/api/datasets/{ds_id}/versions/1")
    assert r.status_code == 200
    assert len(r.json()["rows"]) == 100

    # 6. 差异修订生成 v2（改 2 行），v1 结果不受影响
    diff = {
        "upserts": [
            {"unit": "TA0", "period": 1, "outcome": 99.0, "treated": True, "covariates": {"x": 0.0}},
            {"unit": "TA1", "period": 1, "outcome": 88.0, "treated": True, "covariates": {"x": 1.0}},
        ],
        "deletes": [],
        "change_note": "改两个处理后值",
    }
    r = client.post(f"/api/datasets/{ds_id}/versions", json=diff)
    assert r.status_code == 201, r.text
    v2 = r.json()
    assert v2["version"] == 2 and v2["parent_version"] == 1

    # 旧版本旧结果仍在，且结果不变
    r = client.get(f"/api/results/{result_id}")
    assert r.status_code == 200
    assert r.json()["result"]["att"] == pytest.approx(3.0, abs=1e-8)

    # 7. v2 上同设定是新结果（且因数据被改，效应不同）
    r = client.post(f"/api/versions/{v2['version_id']}/analyses",
                    json={"name": "main", "control_strategy": "never_treated"})
    assert r.json()["cached"] is False
    assert abs(r.json()["result"]["att"] - 3.0) > 1.0


def test_diff_versions_match_full_upload(client):
    """差异版本与直接上传相同完整内容，估计一致。"""
    base = make_2x2_rows("D")
    r = client.post("/api/datasets", json={"name": "diff-base", "rows": base})
    v1 = r.json()
    ds_id = v1["dataset_id"]

    # 完整目标内容：改一行、删一行、加一个单位
    full = [row for row in base]
    full[1]["outcome"] = 20.0
    full = [row for row in full if not (row["unit"] == "DB2" and row["period"] == 1)]
    full.append({"unit": "DNEW", "period": 0, "outcome": 5.0, "treated": False, "covariates": {"x": 0.0}})
    full.append({"unit": "DNEW", "period": 1, "outcome": 7.0, "treated": False, "covariates": {"x": 0.0}})

    upserts = [full[1], full[-2], full[-1]]
    deletes = [{"unit": "DB2", "period": 1}]

    r = client.post(f"/api/datasets/{ds_id}/versions",
                    json={"upserts": upserts, "deletes": deletes, "change_note": "v2"})
    assert r.status_code == 201, r.text
    v2_diff_id = r.json()["version_id"]

    r = client.post("/api/datasets", json={"name": "diff-full", "rows": full})
    v2_full_id = r.json()["version_id"]

    a = {"name": "m", "control_strategy": "never_treated"}
    r1 = client.post(f"/api/versions/{v2_diff_id}/analyses", json=a).json()["result"]
    r2 = client.post(f"/api/versions/{v2_full_id}/analyses", json=a).json()["result"]
    assert r1["att"] == pytest.approx(r2["att"], abs=1e-10)
    assert r1["std_error"] == pytest.approx(r2["std_error"], abs=1e-10)


def test_invariants_via_api(client):
    rows = make_2x2_rows("I")
    r = client.post("/api/datasets", json={"name": "inv", "rows": rows})
    vid = r.json()["version_id"]
    spec = {"name": "s", "control_strategy": "never_treated"}
    base = client.post(f"/api/versions/{vid}/analyses", json=spec).json()["result"]

    # 加常数
    shifted = [{**row, "outcome": row["outcome"] + 42.0} for row in rows]
    r = client.post("/api/datasets", json={"name": "inv-c", "rows": shifted})
    r2 = client.post(f"/api/versions/{r.json()['version_id']}/analyses", json=spec).json()["result"]
    assert r2["att"] == pytest.approx(base["att"], abs=1e-10)
    assert r2["std_error"] == pytest.approx(base["std_error"], abs=1e-10)

    # 乘 k
    scaled = [{**row, "outcome": row["outcome"] * 3.5} for row in rows]
    r = client.post("/api/datasets", json={"name": "inv-k", "rows": scaled})
    r3 = client.post(f"/api/versions/{r.json()['version_id']}/analyses", json=spec).json()["result"]
    assert r3["att"] == pytest.approx(3.5 * base["att"], rel=1e-9)
    assert r3["std_error"] == pytest.approx(3.5 * base["std_error"], rel=1e-9)

    # 打乱行序
    shuffled = rows[:]
    random.Random(1).shuffle(shuffled)
    r = client.post("/api/datasets", json={"name": "inv-o", "rows": shuffled})
    r4 = client.post(f"/api/versions/{r.json()['version_id']}/analyses", json=spec).json()["result"]
    assert r4["att"] == pytest.approx(base["att"], abs=1e-10)
    assert r4["std_error"] == pytest.approx(base["std_error"], abs=1e-10)


# ---------------------------------------------------------------------------
# 校验错误：都应是 422 且指出字段/原因
# ---------------------------------------------------------------------------


def _post_dataset(client, rows):
    return client.post("/api/datasets", json={"name": "bad", "rows": rows})


def test_api_duplicate_rows(client):
    rows = make_2x2_rows("X")
    rows.append(dict(rows[0]))
    r = _post_dataset(client, rows)
    assert r.status_code == 422
    body = json.dumps(r.json(), ensure_ascii=False)
    assert "重复" in body or "多行" in body


def test_api_treatment_reversal(client):
    rows = make_2x2_rows("R")
    # RA0 在 t=0 已处理、t=1 未处理 = 撤回（违反单调）
    for row in rows:
        if row["unit"] == "RA0" and row["period"] == 0:
            row["treated"] = True
        if row["unit"] == "RA0" and row["period"] == 1:
            row["treated"] = False
    r = _post_dataset(client, rows)
    assert r.status_code == 422
    assert "撤回" in json.dumps(r.json(), ensure_ascii=False)


def test_api_bad_period(client):
    rows = make_2x2_rows("P")
    rows[0]["period"] = "去年冬天"
    r = _post_dataset(client, rows)
    assert r.status_code == 422
    assert "period" in json.dumps(r.json())


def test_api_non_numeric_outcome(client):
    rows = make_2x2_rows("N")
    rows[0]["outcome"] = "十五"
    r = _post_dataset(client, rows)
    assert r.status_code == 422
    body = json.dumps(r.json(), ensure_ascii=False)
    assert "outcome" in body and "数值" in body


def test_api_no_treated_units(client):
    rows = make_2x2_rows("U")
    for row in rows:
        row["treated"] = False
    r = _post_dataset(client, rows)
    assert r.status_code == 201  # 上传本身允许
    vid = r.json()["version_id"]
    r2 = client.post(f"/api/versions/{vid}/analyses",
                     json={"name": "x", "control_strategy": "never_treated"})
    assert r2.status_code == 422
    assert "没有任何单位接受处理" in json.dumps(r2.json(), ensure_ascii=False)


def test_api_delete_nonexistent(client):
    rows = make_2x2_rows("Z")
    r = _post_dataset(client, rows)
    ds = r.json()["dataset_id"]
    r2 = client.post(f"/api/datasets/{ds}/versions",
                     json={"deletes": [{"unit": "ghost", "period": 0}]})
    assert r2.status_code == 422


def test_api_diff_with_bad_delete_shape(client):
    r = _post_dataset(client, make_2x2_rows("Q"))
    ds = r.json()["dataset_id"]
    r2 = client.post(f"/api/datasets/{ds}/versions", json={"deletes": [{"unit": "x"}]})
    assert r2.status_code == 422


def test_api_not_found(client):
    assert client.get("/api/datasets/999999").status_code == 404
    assert client.get("/api/results/999999").status_code == 404

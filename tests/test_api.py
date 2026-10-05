"""API + PostgreSQL 集成测试。

需 PostgreSQL：TEST_DATABASE_URL=postgresql://... pytest
compose 内：docker compose run --rm -e TEST_DATABASE_URL=... api pytest
"""
from __future__ import annotations

import random

import pytest


def _make_dataset(client, name="pilot"):
    r = client.post("/datasets", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _staggered_rows():
    """小规模交错处理数据（无噪声真值）：A g=2022 效应 2,4；B g=2023 效应 2。"""
    rows = []
    for gname, g, units in (("A", "2022", 30), ("B", "2023", 30), ("U", None, 30)):
        for k in range(units):
            for t, p in enumerate(("2020", "2021", "2022", "2023", "2024")):
                treated = g is not None and p >= g
                gi = {"2022": 2, "2023": 3}.get(g, 99)
                eff = (2 * (t - gi + 1)) if treated else 0.0
                rows.append(
                    {
                        "unit": f"{gname}{k}",
                        "period": p,
                        "outcome": eff,
                        "treated": treated,
                        "covariates": {"x": k * 0.01},
                    }
                )
    return rows


def test_full_version_lifecycle_and_idempotent_estimate(client):
    ds = _make_dataset(client)
    rows = _staggered_rows()

    r = client.post(f"/datasets/{ds}/versions", json={"rows": rows})
    assert r.status_code == 201, r.text
    v1 = r.json()
    assert v1["version_no"] == 1 and v1["n_rows"] == len(rows)

    r = client.get(f"/datasets/{ds}/versions/1")
    assert r.status_code == 200
    got = {(x["unit"], x["period"]): x["outcome"] for x in r.json()["rows"]}
    assert len(got) == len(rows)

    spec = {"control_group": "never", "adjust_covariates": False, "include_twfe": True}
    r1 = client.post(
        f"/datasets/{ds}/estimates", json={"version": "1", "spec": spec}
    )
    assert r1.status_code == 201, r1.text
    est1 = r1.json()
    assert est1["result"]["att"]["estimate"] == pytest.approx(
        (3.0 + 2.0) / 2, abs=1e-9
    )  # A 平均(2+4)=3，B=2

    # 同版本同设定重复提交 -> 直接返回已有结果（200 + cached，id 相同）
    r2 = client.post(
        f"/datasets/{ds}/estimates", json={"version": "1", "spec": spec}
    )
    assert r2.status_code == 200
    assert r2.json()["cached"] is True
    assert r2.json()["id"] == est1["id"]


def test_new_version_does_not_affect_old_results(client):
    ds = _make_dataset(client)
    rows = _staggered_rows()
    client.post(f"/datasets/{ds}/versions", json={"rows": rows})
    spec = {"control_group": "never", "adjust_covariates": False, "include_twfe": True}
    old = client.post(
        f"/datasets/{ds}/estimates", json={"version": "1", "spec": spec}
    ).json()

    # 发布 v2：把所有结果加 100
    v2_rows = [dict(r, outcome=r["outcome"] + 100.0) for r in rows]
    client.post(f"/datasets/{ds}/versions", json={"rows": v2_rows})

    # 挂在 v1 上的结果原样可取
    got = client.get(f"/datasets/{ds}/estimates/{old['id']}").json()
    assert got["result"]["att"]["estimate"] == pytest.approx(2.5, abs=1e-9)
    assert got["version_id"] == old["version_id"]

    # v2 是新估计（效应不变，因为是整体平移）
    new = client.post(
        f"/datasets/{ds}/estimates", json={"version": "2", "spec": spec}
    )
    assert new.status_code == 201
    assert new.json()["id"] != old["id"]
    assert new.json()["result"]["att"]["estimate"] == pytest.approx(2.5, abs=1e-9)


def test_diff_version_equivalent_to_full_upload(client):
    ds1 = _make_dataset(client, "diff-ds")
    rows = _staggered_rows()
    client.post(f"/datasets/{ds1}/versions", json={"rows": rows})

    # 目标 v2：改 5 行、补 3 个新单位、删 2 行
    target = {(r["unit"], r["period"]): dict(r) for r in rows}
    rng = random.Random(0)
    sample_keys = rng.sample(list(target.keys()), 5)
    for k in sample_keys:
        target[k] = dict(target[k], outcome=target[k]["outcome"] + 1.5)
    for k in range(3):
        for p in ("2020", "2021"):
            target[(f"N{k}", p)] = {
                "unit": f"N{k}",
                "period": p,
                "outcome": 0.3,
                "treated": False,
                "covariates": {"x": 0.0},
            }
    delete_keys = rng.sample(
        [k for k in target if k[0].startswith("U") and k[1] == "2024"], 2
    )
    for k in delete_keys:
        target.pop(k)

    changes = []
    for k in sample_keys:
        changes.append({"op": "upsert", "row": target[k]})
    for k in range(3):
        for p in ("2020", "2021"):
            changes.append({"op": "upsert", "row": target[(f"N{k}", p)]})
    for (u, p) in delete_keys:
        changes.append({"op": "delete", "unit": u, "period": p})

    r = client.post(f"/datasets/{ds1}/versions/diff", json={"changes": changes})
    assert r.status_code == 201, r.text
    assert r.json()["version_no"] == 2 and r.json()["created_via"] == "diff"

    diff_est = client.post(
        f"/datasets/{ds1}/estimates",
        json={
            "version": "2",
            "spec": {"control_group": "never"},
        },
    ).json()

    # 对照数据集：直接上传同样的完整内容
    ds2 = _make_dataset(client, "full-ds")
    full_rows = list(target.values())
    client.post(f"/datasets/{ds2}/versions", json={"rows": full_rows})
    full_est = client.post(
        f"/datasets/{ds2}/estimates",
        json={
            "version": "1",
            "spec": {"control_group": "never"},
        },
    ).json()

    assert full_est["result"]["att"] == diff_est["result"]["att"]
    assert full_est["result"]["dynamic_effects"] == diff_est["result"]["dynamic_effects"]
    assert full_est["result"]["standard_errors"] == diff_est["result"]["standard_errors"]

    # diff 版本取回的内容与全量上传一致（按键集合与值）
    got_diff = client.get(f"/datasets/{ds1}/versions/2").json()["rows"]
    got_full = client.get(f"/datasets/{ds2}/versions/1").json()["rows"]
    norm = lambda xs: sorted(
        ((x["unit"], x["period"], round(x["outcome"], 10), x["treated"]) for x in xs)
    )
    assert norm(got_diff) == norm(got_full)


def test_validation_errors_via_api(client):
    ds = _make_dataset(client)

    # 非数值结果变量
    bad = [
        {"unit": "a", "period": "2020", "outcome": "bad", "treated": 0},
        {"unit": "a", "period": "2021", "outcome": 1.0, "treated": 1},
    ]
    r = client.post(f"/datasets/{ds}/versions", json={"rows": bad})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "outcome_not_numeric" and err["field"] == "outcome"

    # 同一单位同一时期两行
    dup = [
        {"unit": "a", "period": "2020", "outcome": 1.0, "treated": 0},
        {"unit": "a", "period": "2020", "outcome": 2.0, "treated": 0},
    ]
    r = client.post(f"/datasets/{ds}/versions", json={"rows": dup})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "duplicate_unit_period"

    # 时期无法排序
    badp = [
        {"unit": "a", "period": "元年", "outcome": 1.0, "treated": 0},
        {"unit": "a", "period": "2021", "outcome": 1.0, "treated": 0},
    ]
    r = client.post(f"/datasets/{ds}/versions", json={"rows": badp})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "period_unparseable"

    # 没有任何单位处理：上传成功，估计时报 422
    clean = [
        {"unit": "a", "period": "2020", "outcome": 1.0, "treated": 0},
        {"unit": "a", "period": "2021", "outcome": 1.0, "treated": 0},
    ]
    r = client.post(f"/datasets/{ds}/versions", json={"rows": clean})
    assert r.status_code == 201
    r = client.post(
        f"/datasets/{ds}/estimates", json={"version": "latest", "spec": {}}
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "no_treated_units"


def test_diff_first_version_rejected(client):
    ds = _make_dataset(client)
    r = client.post(
        f"/datasets/{ds}/versions/diff",
        json={"changes": [{"op": "delete", "unit": "a", "period": "2020"}]},
    )
    assert r.status_code == 404


def test_404s(client):
    assert client.get("/datasets/00000000-0000-0000-0000-000000000000/versions").status_code == 404

"""面板校验与版本内容一致性测试（不依赖数据库）。"""

from __future__ import annotations

import random

import pytest

from app.did import estimate_did
from app.errors import ValidationError
from app.panel import (
    InputRow,
    build_diff_rows,
    rows_for_estimation,
    validate_panel,
)


def base_rows():
    out = []
    for i in range(20):
        g = 1 if i < 10 else None
        for t in range(4):
            treated = g is not None and t >= g
            y = float(i) + 2.0 * t + (1.5 * (t - g) if treated else 0.0)
            out.append(InputRow(f"u{i:02d}", t, y, treated, {"pop": float(i + t)}))
    return out


def test_validate_accepts_good_panel():
    rows, labels, kind, upserts = validate_panel(base_rows())
    assert kind == "year"
    assert len(rows) == 80
    assert labels == {0: 0, 1: 1, 2: 2, 3: 3}
    assert len(upserts) == 80


def test_duplicate_unit_period_rejected():
    rows = base_rows()
    rows.append(InputRow("u00", 0, 99.0, False, {}))
    with pytest.raises(ValidationError) as ei:
        validate_panel(rows)
    reasons = " ".join(d.get("reason", "") for d in ei.value.details)
    assert "同一单位" in reasons or "重复" in reasons


def test_treatment_reversal_rejected():
    rows = [r for r in base_rows() if not (r.unit == "u00" and r.period == 3)]
    # u00 在 t=1 已处理，t=3 若未处理即撤回
    rows.append(InputRow("u00", 3, 1.0, False, {"pop": 1.0}))
    with pytest.raises(ValidationError) as ei:
        validate_panel(rows)
    assert any("撤回" in d.get("reason", "") for d in ei.value.details)


def test_unorderable_period_rejected():
    rows = base_rows()
    rows[5] = InputRow(rows[5].unit, "很久很久以前", 1.0, False, {})
    with pytest.raises(ValidationError) as ei:
        validate_panel(rows)
    assert any("period" in d.get("field", "") for d in ei.value.details)


def test_mixed_year_quarter_rejected():
    rows = base_rows()
    rows[5] = InputRow(rows[5].unit, "2020Q2", 1.0, True, {})
    with pytest.raises(ValidationError) as ei:
        validate_panel(rows)
    assert any("类型" in d.get("reason", "") for d in ei.value.details)


def test_non_numeric_outcome_rejected():
    rows = base_rows()
    rows[10] = InputRow(rows[10].unit, rows[10].period, "十五", False, {})
    with pytest.raises(ValidationError) as ei:
        validate_panel(rows)
    assert any("outcome" in d.get("field", "") and "数值" in d.get("reason", "")
               for d in ei.value.details)


def test_nan_outcome_rejected():
    rows = base_rows()
    rows[10] = InputRow(rows[10].unit, rows[10].period, float("nan"), False, {})
    with pytest.raises(ValidationError):
        validate_panel(rows)


def test_non_boolean_treated_rejected():
    rows = base_rows()
    rows[10] = InputRow(rows[10].unit, rows[10].period, 1.0, "yes", {})
    with pytest.raises(ValidationError) as ei:
        validate_panel(rows)
    assert any("treated" in d.get("field", "") for d in ei.value.details)


def test_empty_panel_rejected():
    with pytest.raises(ValidationError):
        validate_panel([])


def test_quarter_parsing_and_ordering():
    rows = [
        InputRow("a", "2019Q4", 0.0, False, {}),
        InputRow("a", "2020Q1", 1.0, True, {}),
        InputRow("b", "2019Q4", 0.0, False, {}),
        InputRow("b", "2020Q1", 0.2, False, {}),
    ]
    valid, labels, kind, _ = validate_panel(rows)
    assert kind == "quarter"
    est_rows, _ = rows_for_estimation(valid, [])
    res = estimate_did(est_rows)
    # Δa = 1, Δb = 0.2
    assert res["att"] == pytest.approx(0.8, abs=1e-10)


# ---------------------------------------------------------------------------
# 差异修订版本 vs 直接上传完整内容，估计结果一致
# ---------------------------------------------------------------------------


def _full_content(valid_rows):
    return {
        (r.unit, r.period): (r.label, r.outcome, r.treated, dict(r.covariates))
        for r in valid_rows
    }


def test_diff_version_matches_full_upload():
    v1 = validate_panel(base_rows())[0]

    # 在 v1 基础上做几类修订：改几行 outcome、补一批新单位、删几行
    next_input = base_rows()
    # 改数
    for i, r in enumerate(next_input):
        if r.unit == "u00" and r.period == 2:
            next_input[i] = InputRow(r.unit, r.period, r.outcome + 3.0, r.treated, r.covariates)
    # 删几行
    next_input = [r for r in next_input if not (r.unit == "u01" and r.period == 3)]
    # 补新单位
    for t in range(4):
        next_input.append(InputRow("u99", t, 5.0 + t, t >= 2, {"pop": float(t)}))

    parent_map = _full_content(v1)
    upserts, deletes = build_diff_rows(parent_map, next_input)
    assert upserts and deletes  # 差异确实非空

    # 差异合并出的 v2
    deleted = {(d["unit"], d["period"]) for d in deletes}
    v2_diff, *_ = validate_panel(upserts, existing=parent_map, deleted=deleted)
    # 直接整体上传
    v2_full, *_ = validate_panel(next_input)

    key_diff = sorted((r.unit, r.period, r.outcome, r.treated, r.covariates) for r in v2_diff)
    key_full = sorted((r.unit, r.period, r.outcome, r.treated, r.covariates) for r in v2_full)
    assert key_diff == key_full

    est_diff, _ = rows_for_estimation(v2_diff, [])
    est_full, _ = rows_for_estimation(v2_full, [])
    r1 = estimate_did(est_diff)
    r2 = estimate_did(est_full)
    assert r1["att"] == pytest.approx(r2["att"], abs=1e-12)
    assert r1["std_error"] == pytest.approx(r2["std_error"], abs=1e-12)


def test_diff_with_missing_covariate_column_errors_at_estimation():
    rows = base_rows()
    valid, *_ = validate_panel(rows)
    # 分析指定一个不存在的协变量列
    with pytest.raises(ValidationError):
        rows_for_estimation(valid, ["does_not_exist"])


def test_delete_nonexistent_record_rejected():
    v1 = validate_panel(base_rows())[0]
    with pytest.raises(ValidationError):
        validate_panel(
            [],
            existing=_full_content(v1),
            expect_period_kind="year",
            deleted={("u00", 4000)},  # 不存在
        )


def test_upsert_and_delete_same_key_rejected():
    v1 = validate_panel(base_rows())[0]
    one = base_rows()[0]
    with pytest.raises(ValidationError):
        validate_panel(
            [one],
            existing=_full_content(v1),
            expect_period_kind="year",
            deleted={(one.unit, one.period)},
        )

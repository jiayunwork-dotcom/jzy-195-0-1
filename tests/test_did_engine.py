"""引擎层测试：不依赖数据库与 Web，覆盖题目要求的全部成立条件与报错条件。

运行：pytest tests/test_did_engine.py
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.did import PanelValidationError, build_panel, estimate


# --------------------------------------------------------------------------- #
# 模拟数据工具
# --------------------------------------------------------------------------- #
def staggered_data(seed=0, noise=1.0):
    """交错处理 + 效应随相对时期线性增长的面板（含从未处理组）。

    时期 0..5，三个组：
      A：g=2（200 县），效应 κ*(t-g+1), t>=g
      B：g=4（200 县），效应 κ*(t-g+1), t>=g
      U：从未处理（200 县）
    κ=2。误差 iid N(0, noise^2)，县固定效应与时期冲击为组间对称的确定性项，
    平行趋势成立。
    """
    rng = np.random.default_rng(seed)
    T = 6
    n_each = 200
    kappa = 2.0
    rows = []

    def make(units, group_name, g):
        for k in range(units):
            unit_fe = rng.normal(0, 0.5)
            base = rng.normal(0, 1.0)
            for t in range(T):
                period_fe = 0.3 * t  # 共同时期趋势
                treated = g >= 0 and t >= g
                eff = 0.0
                if treated:
                    eff = kappa * (t - g + 1)
                y = base + unit_fe + period_fe + eff + rng.normal(0, noise)
                rows.append(
                    {
                        "unit": f"{group_name}{k:04d}",
                        "period": f"201{t}",
                        "outcome": y,
                        "treated": treated,
                        "covariates": {"x": base},
                    }
                )

    make(n_each, "A", 2)
    make(n_each, "B", 4)
    make(n_each, "U", -1)
    return rows, kappa


def two_by_two_rows():
    """两组两期教科书例子：处理前 T=10/C=8，处理后 T=15/C=10，真值 3。"""
    rows = []
    for u in range(20):
        treated_group = u < 10
        for t, p in enumerate(("2020", "2021")):
            base = 10.0 if treated_group else 8.0
            gain = 5.0 if treated_group else 2.0
            rows.append(
                {
                    "unit": f"u{u}",
                    "period": p,
                    "outcome": base if t == 0 else base + gain,
                    "treated": bool(treated_group and t == 1),
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# 1) 两组两期：效应恰为 3
# --------------------------------------------------------------------------- #
def test_two_by_two_equals_three():
    panel = build_panel(two_by_two_rows())
    res = estimate(panel)
    assert res["att"]["estimate"] == pytest.approx(3.0, abs=1e-10)
    # 常数均值、无噪声 -> 完全拟合，残差为零，聚类 SE=0
    assert res["att"]["std_error"] == pytest.approx(0.0, abs=1e-12)
    # TWFE 在这个干净 2x2 上同样无偏
    assert res["twfe"]["estimate"] == pytest.approx(3.0, abs=1e-10)


# --------------------------------------------------------------------------- #
# 2) 交错处理 + 动态效应：估计量还原真值；TWFE 有偏
# --------------------------------------------------------------------------- #
def test_staggered_recovers_truth_and_twfe_biased():
    rows, kappa = staggered_data(seed=42)
    panel = build_panel(rows)
    res = estimate(panel)

    # 总 ATT：所有 post 的 ATT(g,t) 按组规模 n_g 加权。两组规模相等，
    # A 贡献 4 个单元（真值 2,4,6,8）、B 贡献 2 个单元（真值 2,4），
    # 合计 (2+4+6+8+2+4)/6 = 13/3。
    assert res["att"]["estimate"] == pytest.approx(13 / 3, abs=0.15)
    cohort = {c["cohort"]: c["estimate"] for c in res["cohort_att"]}
    assert cohort["2012"] == pytest.approx(5.0, abs=0.25)  # A 组内平均
    assert cohort["2014"] == pytest.approx(3.0, abs=0.25)  # B 组内平均

    # 动态效应：e=0 为处理开始期，真值 2；两组在 e=0/1 均有，e=2/3 仅 A。
    dyn = {d["relative_period"]: d["estimate"] for d in res["dynamic_effects"]}
    assert dyn[0] == pytest.approx(2.0, abs=0.2)
    assert dyn[1] == pytest.approx(4.0, abs=0.25)
    assert dyn[2] == pytest.approx(6.0, abs=0.3)
    assert dyn[3] == pytest.approx(8.0, abs=0.3)

    # 平行趋势成立，检验不应拒绝（p 不显著）
    assert res["pretrend_test"]["p_value"] > 0.01

    # TWFE 在“后处理组作对照 + 效应递增”下被向下拉偏
    twfe = res["twfe"]["estimate"]
    assert twfe < 4.1  # 明显低于真值 13/3（负偏误）
    assert abs(twfe - 13 / 3) > 3 * res["twfe"]["std_error"] + 0.05


def test_staggered_seed_robustness():
    # 多个种子下都应还原真值，防止偶发通过
    for seed in (1, 7, 123):
        rows, _ = staggered_data(seed=seed)
        res = estimate(build_panel(rows))
        assert res["att"]["estimate"] == pytest.approx(13 / 3, abs=0.2)


# --------------------------------------------------------------------------- #
# 3) 加常数不变；乘正数 k，估计与 SE 都乘 k
# --------------------------------------------------------------------------- #
def _affine_rows(rows, a=0.0, k=1.0):
    out = []
    for r in rows:
        rr = dict(r)
        rr["outcome"] = k * float(r["outcome"]) + a
        out.append(rr)
    return out


def test_additive_constant_invariant():
    rows, _ = staggered_data(seed=3, noise=1.0)
    base = estimate(build_panel(rows))
    shifted = estimate(build_panel(_affine_rows(rows, a=1000.0)))
    assert shifted["att"]["estimate"] == pytest.approx(base["att"]["estimate"], abs=1e-10)
    assert shifted["att"]["std_error"] == pytest.approx(base["att"]["std_error"], rel=1e-10)
    for d1, d2 in zip(shifted["dynamic_effects"], base["dynamic_effects"]):
        assert d1["estimate"] == pytest.approx(d2["estimate"], abs=1e-10)
        assert d1["std_error"] == pytest.approx(d2["std_error"], rel=1e-10)
    assert shifted["twfe"]["estimate"] == pytest.approx(base["twfe"]["estimate"], abs=1e-10)
    assert shifted["pretrend_test"]["statistic"] == pytest.approx(
        base["pretrend_test"]["statistic"], rel=1e-9
    )


def test_positive_scale_linear():
    rows, _ = staggered_data(seed=5, noise=1.0)
    base = estimate(build_panel(rows))
    k = 3.0
    scaled = estimate(build_panel(_affine_rows(rows, k=k)))
    assert scaled["att"]["estimate"] == pytest.approx(k * base["att"]["estimate"], rel=1e-9)
    assert scaled["att"]["std_error"] == pytest.approx(
        k * base["att"]["std_error"], rel=1e-9
    )
    for d1, d0 in zip(scaled["dynamic_effects"], base["dynamic_effects"]):
        assert d1["estimate"] == pytest.approx(k * d0["estimate"], rel=1e-9)
        assert d1["std_error"] == pytest.approx(k * d0["std_error"], rel=1e-9)
    assert scaled["twfe"]["estimate"] == pytest.approx(
        k * base["twfe"]["estimate"], rel=1e-9
    )
    assert scaled["twfe"]["std_error"] == pytest.approx(
        k * base["twfe"]["std_error"], rel=1e-9
    )


# --------------------------------------------------------------------------- #
# 4) 打乱单位编号与行序不变
# --------------------------------------------------------------------------- #
def test_permutation_and_row_order_invariant():
    rows, _ = staggered_data(seed=9)
    base = estimate(build_panel(rows))

    # 行序完全打乱
    shuffled = rows[:]
    rng = np.random.default_rng(0)
    rng.shuffle(shuffled)
    res1 = estimate(build_panel(shuffled))
    assert res1["att"]["estimate"] == pytest.approx(base["att"]["estimate"], abs=1e-12)
    assert res1["att"]["std_error"] == pytest.approx(base["att"]["std_error"], rel=1e-12)

    # 单位编号重命名（排列），时期与值不变
    units = sorted({r["unit"] for r in rows})
    perm = units[:]
    rng.shuffle(perm)
    mapping = dict(zip(units, perm))
    relabeled = []
    for r in shuffled:
        rr = dict(r)
        rr["unit"] = "z_" + mapping[r["unit"]]
        relabeled.append(rr)
    res2 = estimate(build_panel(relabeled))
    assert res2["att"]["estimate"] == pytest.approx(base["att"]["estimate"], abs=1e-12)
    assert res2["att"]["std_error"] == pytest.approx(base["att"]["std_error"], rel=1e-12)


# --------------------------------------------------------------------------- #
# 5) 报错条件
# --------------------------------------------------------------------------- #
def _expect_error(rows, code):
    with pytest.raises(PanelValidationError) as exc:
        build_panel(rows)
    assert exc.value.code == code, exc.value


def test_error_duplicate_unit_period():
    rows = two_by_two_rows()
    rows.append(dict(rows[0]))
    _expect_error(rows, "duplicate_unit_period")


def test_error_treatment_reversal():
    rows = [
        {"unit": "a", "period": "2020", "outcome": 1.0, "treated": 0},
        {"unit": "a", "period": "2021", "outcome": 1.0, "treated": 1},
        {"unit": "a", "period": "2022", "outcome": 1.0, "treated": 0},
    ]
    _expect_error(rows, "treatment_reversed")


def test_error_period_not_sortable():
    rows = [
        {"unit": "a", "period": "2020春", "outcome": 1.0, "treated": 0},
        {"unit": "a", "period": "2021", "outcome": 1.0, "treated": 1},
    ]
    _expect_error(rows, "period_unparseable")


def test_error_mixed_frequency():
    rows = [
        {"unit": "a", "period": "2020Q1", "outcome": 1.0, "treated": 0},
        {"unit": "a", "period": "2020", "outcome": 1.0, "treated": 0},
    ]
    _expect_error(rows, "period_frequency_mixed")


def test_error_outcome_non_numeric():
    rows = [
        {"unit": "a", "period": "2020", "outcome": "十", "treated": 0},
        {"unit": "a", "period": "2021", "outcome": 1.0, "treated": 1},
    ]
    _expect_error(rows, "outcome_not_numeric")


def test_error_outcome_nan():
    rows = [
        {"unit": "a", "period": "2020", "outcome": float("nan"), "treated": 0},
        {"unit": "a", "period": "2021", "outcome": 1.0, "treated": 1},
    ]
    _expect_error(rows, "outcome_not_numeric")


def test_error_no_treated_units():
    rows = [
        {"unit": "a", "period": "2020", "outcome": 1.0, "treated": 0},
        {"unit": "a", "period": "2021", "outcome": 2.0, "treated": 0},
        {"unit": "b", "period": "2020", "outcome": 1.0, "treated": 0},
        {"unit": "b", "period": "2021", "outcome": 2.0, "treated": 0},
    ]
    panel = build_panel(rows)
    with pytest.raises(PanelValidationError) as exc:
        estimate(panel)
    assert exc.value.code == "no_treated_units"


def test_error_missing_field():
    _expect_error(
        [{"unit": "a", "period": "2020", "outcome": 1.0}], "missing_field"
    )


def test_error_treated_not_binary():
    rows = [
        {"unit": "a", "period": "2020", "outcome": 1.0, "treated": 2},
        {"unit": "a", "period": "2021", "outcome": 1.0, "treated": 1},
    ]
    _expect_error(rows, "treated_not_binary")


def test_error_empty():
    _expect_error([], "empty_dataset")


def test_error_no_never_treated_switches_to_not_yet():
    # 所有单位最终都处理了：never 口径报错，not_yet 口径可估
    rows = [
        {"unit": "a", "period": "2020", "outcome": 1.0, "treated": 0},
        {"unit": "a", "period": "2021", "outcome": 3.0, "treated": 1},
        {"unit": "a", "period": "2022", "outcome": 5.0, "treated": 1},
        {"unit": "b", "period": "2020", "outcome": 1.0, "treated": 0},
        {"unit": "b", "period": "2021", "outcome": 2.0, "treated": 0},
        {"unit": "b", "period": "2022", "outcome": 4.0, "treated": 1},
    ]
    panel = build_panel(rows)
    with pytest.raises(PanelValidationError) as exc:
        estimate(panel, control_group="never")
    assert exc.value.code == "no_never_treated_controls"
    res = estimate(panel, control_group="not_yet")
    assert math.isfinite(res["att"]["estimate"])


# --------------------------------------------------------------------------- #
# 6) 两种对照口径何时明显不同：后处理组有预期/事前跳变时
# --------------------------------------------------------------------------- #
def test_never_vs_not_yet_diverge_when_later_cohort_reacts_early():
    """B 组在自己处理开始前一期就出现 +4 的跳变（预期效应或平行趋势失效）。

    用从未处理 U 作对照：A 组效应估计不受 B 的跳变影响。
    用尚未处理 B 作对照：B 的事前跳变被当成“共同趋势”，A 的效应被向下污染。
    """
    rows = []
    # A 组 g=1，真实处理效应恒为 5
    for k in range(100):
        base = 0.0
        for t in range(4):
            y = base + (5.0 if t >= 1 else 0.0)
            rows.append(
                {"unit": f"A{k}", "period": f"201{t}", "outcome": y, "treated": t >= 1}
            )
    # B 组 g=3：自身处理效应为 0，但在 t=2（尚未处理）时就出现 +4 跳变
    for k in range(100):
        for t in range(4):
            y = 4.0 if t == 2 else 0.0
            rows.append(
                {"unit": f"B{k}", "period": f"201{t}", "outcome": y, "treated": t >= 3}
            )
    # U 从未处理，始终 0
    for k in range(100):
        for t in range(4):
            rows.append(
                {"unit": f"U{k}", "period": f"201{t}", "outcome": 0.0, "treated": False}
            )
    panel = build_panel(rows)
    never = estimate(panel, control_group="never")
    notyet = estimate(panel, control_group="not_yet")
    # 干净队列 A 的 ATT：never 对照还原真值 5；not_yet 对照把 B 的事前
    # +4 跳变当成共同趋势，A 在 2012（B 尚未处理且已跳变）那一期的单元
    # 估计被拉低到 3，队列平均也从 5 掉到 4.33，两种口径明显不同。
    cohort_never = {c["cohort"]: c["estimate"] for c in never["cohort_att"]}
    cohort_notyet = {c["cohort"]: c["estimate"] for c in notyet["cohort_att"]}
    assert cohort_never["2011"] == pytest.approx(5.0, abs=1e-9)
    assert abs(cohort_never["2011"] - cohort_notyet["2011"]) > 0.5
    cell_2012_never = {
        c["period"]: c["estimate"]
        for c in never["group_time_att"]
        if c["cohort"] == "2011" and c["kind"] == "post"
    }
    cell_2012_notyet = {
        c["period"]: c["estimate"]
        for c in notyet["group_time_att"]
        if c["cohort"] == "2011" and c["kind"] == "post"
    }
    assert cell_2012_never["2012"] == pytest.approx(5.0, abs=1e-9)
    assert cell_2012_notyet["2012"] == pytest.approx(3.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 7) 季度面板与协变量调整的冒烟测试
# --------------------------------------------------------------------------- #
def test_quarterly_panel():
    rows, _ = staggered_data(seed=1)
    qrows = []
    for r in rows:
        rr = dict(r)
        rr["period"] = f"{r['period']}Q1"
        qrows.append(rr)
    res = estimate(build_panel(qrows))
    assert res["att"]["estimate"] == pytest.approx(13 / 3, abs=0.2)


def test_covariate_adjustment_runs():
    rows, _ = staggered_data(seed=2)
    res = estimate(build_panel(rows), adjust_covariates=True)
    assert res["adjust_covariates"] is True
    assert res["covariates"] == ["x"]
    assert math.isfinite(res["att"]["estimate"])

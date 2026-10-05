"""估计量数值性质测试（不依赖数据库）。"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from app.did import Row, estimate_did, estimate_twfe
from app.errors import EstimationError


# ---------------------------------------------------------------------------
# 模拟数据生成
# ---------------------------------------------------------------------------


def staggered_panel(seed: int = 0, n_per_cohort: int = 120, n_never: int = 120,
                    n_periods: int = 6, cohorts=(1, 3), slope: float = 1.0,
                    noise_sd: float = 1.0, include_never: bool = True,
                    cohort_drift: dict | None = None):
    """交错处理、效应随相对时期线性增长的面板。

    y_it = α_i + ε_it + slope·(t-g)_+ + cohort_drift[g]·t
    （cohort_drift 默认 0，平行趋势在条件期望上成立。）
    """
    rng = np.random.default_rng(seed)
    drift = cohort_drift or {}
    rows = []
    units = [(f"c{g}_{i}", g) for g in cohorts for i in range(n_per_cohort)]
    if include_never:
        units += [(f"nev_{i}", None) for i in range(n_never)]
    for u, g in units:
        alpha = rng.normal(0, 1)
        ar = 0.0
        for t in range(n_periods):
            ar = 0.5 * ar + rng.normal(0, noise_sd)
            eff = slope * (t - g) if (g is not None and t >= g) else 0.0
            trend = drift.get(g, 0.0) * t if g is not None else 0.0
            rows.append(Row(u, t, alpha + ar + eff + trend, g is not None and t >= g, ()))
    return rows


def expected_att(cohorts=(1, 3), n_periods: int = 6, n_per_cohort: int = 120,
                 slope: float = 1.0):
    """处理后所有 (g,t>=g) 的 ATT(g,t) 按处理组单位数等权汇总的解析真值。"""
    total, wsum = 0.0, 0
    for g in cohorts:
        for t in range(n_periods):
            if t >= g:
                total += slope * (t - g) * n_per_cohort
                wsum += n_per_cohort
    return total / wsum


# ---------------------------------------------------------------------------
# 需求 1：教科书 2×2，处理效应 = 3
# ---------------------------------------------------------------------------


def test_two_by_two_att_is_three():
    rows = []
    for i in range(30):
        rows += [
            Row(f"T{i}", 0, 10.0, False, ()),
            Row(f"T{i}", 1, 15.0, True, ()),
            Row(f"C{i}", 0, 8.0, False, ()),
            Row(f"C{i}", 1, 10.0, False, ()),
        ]
    res = estimate_did(rows, "never_treated")
    assert res["att"] == pytest.approx(3.0, abs=1e-10)
    assert res["std_error"] == pytest.approx(0.0, abs=1e-10)
    # 零残差（确定性数据）下 z/p 值不可识别，应为 None 而不是天文数字
    assert res["z"] is None and res["p_value"] is None
    # 动态效应 e=0 即 3
    e0 = [r for r in res["event_study"] if r["relative_period"] == 0][0]
    assert e0["att"] == pytest.approx(3.0, abs=1e-10)
    # TWFE 在此同口径情形应给出相同答案
    tw = estimate_twfe(rows)
    assert tw["coef_static"] == pytest.approx(3.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 需求 2：交错 + 增长效应，堆叠 DID 还原真值；TWFE 有偏
# ---------------------------------------------------------------------------


def test_staggered_recovers_truth_and_twfe_biased():
    rows = staggered_panel(seed=42)
    truth = expected_att()
    res = estimate_did(rows, "never_treated")
    tw = estimate_twfe(rows)

    assert res["att"] == pytest.approx(truth, abs=0.08)
    # 事件研究（g-1 长差分口径）：e>=0 的累积效应应在容差内等于 slope*e
    # （本 DGP 设定效应 slope*(t-g)，故 e=0 真值为 0；e=0 只能由首期队列识别，
    # 该队列退化为 t-1 基期，噪声下精度略低，容差放宽。）
    for r in res["event_study"]:
        if r["relative_period"] == 0:
            assert r["att"] == pytest.approx(0.0, abs=0.25)
        elif r["relative_period"] > 0:
            assert r["att"] == pytest.approx(float(r["relative_period"]), abs=0.25)
        elif r["relative_period"] <= -2:
            assert r["att"] == pytest.approx(0.0, abs=0.3)

    # TWFE 在该设定下系统性偏小（已处理单位当对照造成的污染）
    assert tw["coef_static"] != pytest.approx(truth, abs=0.2)
    assert abs(tw["coef_static"] - truth) > abs(res["att"] - truth)


def test_staggered_sign_reversal_demo():
    """没有从未处理单位、效应快速增长时，TWFE 甚至可以反号。

    队列取 (2,4)、共 6 期。not_yet_treated 口径下，g=4 队列在末期没有
    尚未处理对照（案例被跳过），可识别案例覆盖 1/4/2/3 的处理权重，
    估计量仍为正且接近"可识别案例"的真值；TWFE 则为负，符号相反。
    """
    rows = staggered_panel(seed=7, cohorts=(2, 4), slope=3.0, noise_sd=0.5,
                           include_never=False)
    res = estimate_did(rows, "not_yet_treated")
    tw = estimate_twfe(rows)

    # 可识别案例：g=2 基期固定为 g-1=1；t=4 起 g=4 队列已处理，故只剩
    # (2,2)、(2,3) 两个案例（e=0,1），其余出现在 skipped_cases_no_control。
    n = 120
    cases = [(2, 2), (2, 3)]
    truth_ident = sum(n * 3.0 * (t - g) for g, t in cases) / (n * len(cases))
    assert res["att"] == pytest.approx(truth_ident, abs=0.2)
    assert res["att"] > 0
    assert tw["coef_static"] < 0  # TWFE 反号
    assert len(res["skipped_cases_no_control"]) >= 1


# ---------------------------------------------------------------------------
# 需求 3：加常数，效应与标准误不变
# ---------------------------------------------------------------------------


def test_constant_shift_invariance():
    rows = staggered_panel(seed=99)
    base = estimate_did(rows, "never_treated")
    shifted = [Row(r.unit, r.period, r.outcome + 123.456, r.treated, r.covariates)
               for r in rows]
    res = estimate_did(shifted, "never_treated")
    assert res["att"] == pytest.approx(base["att"], abs=1e-10)
    assert res["std_error"] == pytest.approx(base["std_error"], abs=1e-12)
    for a, b in zip(base["event_study"], res["event_study"]):
        assert a["att"] == pytest.approx(b["att"], abs=1e-10)
        assert (a["std_error"] or 0) == pytest.approx(b["std_error"] or 0, abs=1e-12)


# ---------------------------------------------------------------------------
# 需求 4：乘正数 k，效应与标准误乘 k
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0.37, 2.5, 10.0])
def test_positive_scaling(k):
    rows = staggered_panel(seed=5)
    base = estimate_did(rows, "never_treated")
    scaled = [Row(r.unit, r.period, r.outcome * k, r.treated, r.covariates)
              for r in rows]
    res = estimate_did(scaled, "never_treated")
    assert res["att"] == pytest.approx(k * base["att"], rel=1e-9)
    assert res["std_error"] == pytest.approx(k * base["std_error"], rel=1e-9)


def test_quarterly_staggered_data_works():
    """季度标签下的交错估计（2019Q1..2020Q4，队列 2019Q2 与 2020Q1）。"""
    labels = ["2019Q1", "2019Q2", "2019Q3", "2019Q4", "2020Q1", "2020Q2",
              "2020Q3", "2020Q4"]
    idx = {lab: 2019 * 4 + i for i, lab in enumerate(labels)}
    rng = np.random.default_rng(0)
    rows = []
    for i in range(90):
        g_lab = labels[1] if i < 30 else (labels[4] if i < 60 else None)
        for lab in labels:
            treated = g_lab is not None and idx[lab] >= idx[g_lab]
            e = idx[lab] - idx[g_lab] if treated else 0
            rows.append(Row(f"u{i}", idx[lab], rng.normal(0, 0.2) + float(e),
                            treated, ()))
    res = estimate_did(rows, "never_treated")
    # 处理后 e>=0 效应 = e（线性增长），总体应为正且事件路径恢复
    assert res["att"] > 0
    e4 = [r for r in res["event_study"] if r["relative_period"] == 4][0]
    assert e4["att"] == pytest.approx(4.0, abs=0.2)


# ---------------------------------------------------------------------------
# 需求 5：打乱单位编号 / 行序不变
# ---------------------------------------------------------------------------


def test_permutation_and_row_order_invariance():
    rows = staggered_panel(seed=11)
    base = estimate_did(rows, "never_treated")

    shuffled = rows[:]
    random.Random(0).shuffle(shuffled)
    res1 = estimate_did(shuffled, "never_treated")
    assert res1["att"] == pytest.approx(base["att"], abs=1e-12)
    assert res1["std_error"] == pytest.approx(base["std_error"], abs=1e-12)

    # 重命名单位（双射）
    units = sorted({r.unit for r in rows})
    mapping = {u: f"z{i:04d}" for i, u in enumerate(units)}
    relabeled = [Row(mapping[r.unit], r.period, r.outcome, r.treated, r.covariates)
                 for r in shuffled]
    res2 = estimate_did(relabeled, "never_treated")
    assert res2["att"] == pytest.approx(base["att"], abs=1e-12)
    assert res2["std_error"] == pytest.approx(base["std_error"], abs=1e-12)


# ---------------------------------------------------------------------------
# 需求 6（估计侧）：没有处理单位要报错
# ---------------------------------------------------------------------------


def test_no_treated_units_errors():
    rows = [Row(f"u{i}", t, float(t), False, ()) for i in range(10) for t in range(4)]
    with pytest.raises(EstimationError, match="没有任何单位接受处理"):
        estimate_did(rows)
    with pytest.raises(EstimationError, match="没有任何单位接受处理"):
        estimate_twfe(rows)


def test_bad_control_strategy():
    rows = [Row("u", 0, 0.0, False, ()), Row("u", 1, 1.0, True, ())]
    with pytest.raises(EstimationError):
        estimate_did(rows, control="later_maybe")


# ---------------------------------------------------------------------------
# 平行趋势：零效应数据 p 值不小；植入处理前差异时统计量上升
# ---------------------------------------------------------------------------


def test_pretrend_test_behavior():
    # 无处理前效应
    rows = staggered_panel(seed=3, noise_sd=1.0)
    res = estimate_did(rows, "never_treated")
    pt = res["pretrend_test"]
    assert pt is not None
    assert pt["p_value"] > 0.01

    # 植入处理前趋势：让 g=3 队列在 t=0->1 有 2 的额外增长
    rows2 = []
    for r in rows:
        y = r.outcome
        if r.unit.startswith("c3") and r.period >= 2:
            y += 2.0
        rows2.append(Row(r.unit, r.period, y, r.treated, ()))
    res2 = estimate_did(rows2, "never_treated")
    assert res2["pretrend_test"]["p_value"] < pt["p_value"]


# ---------------------------------------------------------------------------
# 两种对照口径在"无从未处理 + 队列效应异质"数据上可以明显不同
# ---------------------------------------------------------------------------


def test_control_strategies_can_differ():
    """两种口径何时明显不同：晚期队列有自身的处理前趋势（违反平行趋势的方向在
    从未处理对照上成立、但在尚未处理对照上不成立）。never_treated 用始终未处理
    单位做反事实；not_yet_treated 用早期队列（趋势不同）做反事实，二者答案分叉。
    """
    rng = np.random.default_rng(0)
    rows = []
    # g=3 队列每期有 +1.2 的自身增长；g=1 队列无漂移；从未处理单位无漂移
    for g, drift in ((1, 0.0), (3, 1.2)):
        for i in range(120):
            u = f"c{g}_{i}"
            a = rng.normal()
            for t in range(6):
                eff = 1.0 * (t - g) if t >= g else 0.0
                rows.append(Row(u, t, a + rng.normal(0, 0.5) + eff + drift * t,
                                t >= g, ()))
    for i in range(120):
        u = f"n{i}"
        a = rng.normal()
        for t in range(6):
            rows.append(Row(u, t, a + rng.normal(0, 0.5), False, ()))
    a1 = estimate_did(rows, "never_treated")["att"]
    a2 = estimate_did(rows, "not_yet_treated")["att"]
    assert abs(a1 - a2) > 0.1


# ---------------------------------------------------------------------------
# 标准误随噪声尺度变化、聚类数越多越小（健全性）
# ---------------------------------------------------------------------------


def test_se_decreases_with_more_clusters():
    small = staggered_panel(seed=1, n_per_cohort=15, n_never=15)
    large = staggered_panel(seed=1, n_per_cohort=120, n_never=120)
    se_s = estimate_did(small)["std_error"]
    se_l = estimate_did(large)["std_error"]
    assert se_l < se_s


def test_covariate_adjustment_removes_confounding():
    """时变混杂协变量 x：不控制时估计偏大，控制 ΔX 后回到真值附近。"""
    rng = np.random.default_rng(3)
    rows_nox, rows_x = [], []
    for i in range(120):
        g = 1 if i < 40 else (3 if i < 80 else None)
        a = rng.normal()
        for t in range(6):
            d = g is not None and t >= g
            x = a + rng.normal(0, 1) + (0.5 * t if g == 1 else 0.0)
            y = a + rng.normal(0, 0.5) + 1.5 * x + (2.0 if d else 0.0)
            rows_nox.append(Row(f"u{i}", t, y, d, ()))
            rows_x.append(Row(f"u{i}", t, y, d, (x,)))
    biased = estimate_did(rows_nox)["att"]
    adjusted = estimate_did(rows_x)["att"]
    assert abs(adjusted - 2.0) < abs(biased - 2.0)
    assert adjusted == pytest.approx(2.0, abs=0.3)

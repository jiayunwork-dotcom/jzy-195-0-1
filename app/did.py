"""双重差分估计量：手写实现，不依赖任何计量/统计建模库。

口径选择（Callaway & Sant'Anna 2021 思路的堆叠实现）
---------------------------------------------------
交错处理（staggered adoption）下直接跑

    y_it = α_i + λ_t + τ D_it + ε_it

的双向固定效应（TWFE）回归，会把 *已经接受处理* 的单位在后续时期也拿来当对照。
在处理效应随队列（cohort）和时间变化时，各干净的 2×2 比较被 OLS 强行加权成
同一个 τ，权重还可能为负，于是 TWFE 可能偏离真值，甚至符号相反。

本模块避开"已处理单位当对照"，做法是按 (队列 g, 时期 t) 构造干净的 2×2：

* 处理组：队列恰为 g、且在 t 期与紧邻前一期 t-1 都有观测的单位；
* 对照组：在 t 与 t-1 两期都未处理的单位，两种口径——
    - ``never_treated``（默认）：始终未处理的单位；
    - ``not_yet_treated``：t 期还没处理的单位（其 t-1 期自然也未处理）；
* 每个 2×2 用 t-1→t 的一期一阶差分 Δy，估计
  ATT(g,t) = E[Δy | 队列g] - E[Δy | 对照]。
  这是半动态（事件研究）口径：按相对时期 e=t-g 汇总后，事件 e 的系数即
  "处理后第 e 期的当期效应"（e=0 为处理当期，e=-1 为基准）。Callaway &
  Sant'Anna (2021) 原方法以 g-1 为共同基期（长差分，得到的是相对 g-1 的
  累积效应）；两种口径都不使用已处理单位作对照，本服务选一期差分，
  因为它与"按距处理开始的相对时期展开"直接对应。

全部 2×2 堆叠为一个含"案例截距 + 案例×处理组交互"的 OLS（ΔX 线性控制变量
以差分形式进入），交互系数即各 ATT(g,t)，标准误按单位聚类（CR1）。

汇总：
* 总体 ATT：各 ATT(g,t) 按处理组单位数加总；
* 动态效应（事件研究）：按相对时期 e = t-g 以处理组单位数加总；
* 平行趋势：对 e <= -2 的处理前安慰剂系数做联合 Wald 检验（以 e=-1 为基准）。

代价与前提：识别依赖条件平行趋势；不参与任何 2×2 的观测不影响估计；
``not_yet_treated`` 口径还额外依赖"无预期效应"假设，且末期才开始处理的队列
在该口径下可能没有对照（相应案例被跳过并在 skipped_cases_no_control 中列出）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import EstimationError

# ---------------------------------------------------------------------------
# 输入结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """清洗后的一行面板。period 是全数据集统一排序后的整数时期编号。"""

    unit: str
    period: int
    outcome: float
    treated: bool
    covariates: tuple[float, ...] = ()


# ---------------------------------------------------------------------------
# 小工具：正态分布与卡方分布函数（标准误检验用，手写不调统计库）
# ---------------------------------------------------------------------------


def normal_sf(x: float) -> float:
    """标准正态上尾概率 P(Z >= x)（双侧检验时传 |z| 再乘 2）。"""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _gammq(a: float, x: float) -> float:
    """正则化上不完全伽马函数 Q(a,x)，Numerical Recipes 算法。"""
    if x < 0.0 or a <= 0.0:
        return float("nan")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        # 级数表示 1 - P(a,x)
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(200):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-15:
                break
        p = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return 1.0 - p
    # 连分式表示
    b = x + 1.0 - a
    c = 1e30
    d = 1.0 / b
    h = d
    for i in range(1, 201):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x: float, df: int) -> float:
    """卡方分布上尾概率 P(χ²_df >= x)，即 Wald 检验的 p 值。"""
    if x < 0.0:
        return 1.0
    if df <= 0:
        return float("nan")
    return _gammq(df / 2.0, x / 2.0)


def _se_z_p(coef: float, se: float) -> tuple[float | None, float | None]:
    if se is None or not np.isfinite(se) or se <= 1e-12:
        return None, None
    z = coef / se
    p = 2.0 * normal_sf(abs(z))
    return z, min(1.0, max(0.0, p))


# ---------------------------------------------------------------------------
# 面板整理
# ---------------------------------------------------------------------------


def _organize(rows: list[Row]):
    """把行整理成单位面板，返回索引映射与各单位的数组视图。"""
    units = sorted({r.unit for r in rows})
    if not units:
        raise EstimationError("数据为空，无法估计")
    unit_index = {u: i for i, u in enumerate(units)}

    # 先按 (单位, 时期) 排序
    ordered = sorted(rows, key=lambda r: (r.unit, r.period))
    periods = sorted({r.period for r in rows})
    if len(periods) < 2:
        raise EstimationError("至少需要两个时期才能做双重差分")
    k = len(rows[0].covariates)

    # 每个单位 -> dict(period -> (outcome, treated, X))
    panel: dict[str, dict[int, tuple[float, bool, np.ndarray]]] = {u: {} for u in units}
    for r in ordered:
        if r.unit in panel and r.period in panel[r.unit]:
            # 正常情况下 validation 已拦截，这里再兜底
            raise EstimationError(f"单位 {r.unit} 在时期 {r.period} 出现重复记录")
        panel[r.unit][r.period] = (r.outcome, r.treated, np.asarray(r.covariates, dtype=float))

    cohort: dict[str, int | None] = {}
    for u, ps in panel.items():
        first_treated = None
        for p in periods:
            if p not in ps:
                continue
            t = ps[p][1]
            if t and first_treated is None:
                first_treated = p
            if first_treated is not None and not t:
                # validation 已保证不撤回；兜底
                raise EstimationError(f"单位 {u} 在时期 {p} 处理状态撤回（处理后又回到未处理）")
        cohort[u] = first_treated

    treated_units = [u for u in units if cohort[u] is not None]
    never_units = [u for u in units if cohort[u] is None]
    if not treated_units:
        raise EstimationError("没有任何单位接受处理，无法估计处理效应")

    return units, unit_index, periods, panel, cohort, treated_units, never_units, k


# ---------------------------------------------------------------------------
# 主估计：堆叠 2×2（Callaway-Sant'Anna 风格）
# ---------------------------------------------------------------------------

VALID_CONTROLS = ("never_treated", "not_yet_treated")


def _stacked_ols(
    unit_codes: np.ndarray,
    dy: np.ndarray,
    dX: np.ndarray,
    case_ids: np.ndarray,
    is_treated_record: np.ndarray,
    n_cases: int,
    n_clusters_total: int,
):
    """堆叠 2×2 差分 OLS。

    设计矩阵列序：[案例截距 n_cases 列, 案例×处理交互 n_cases 列, ΔX k 列]。
    交互系数即各案例 ATT。返回 ``(att_gt, 交互系数的聚类协方差, n_treated_per_case)``。
    """
    n = len(dy)
    k = dX.shape[1]
    X = np.zeros((n, 2 * n_cases + k))
    X[np.arange(n), case_ids] = 1.0                      # 案例截距
    inter_idx = n_cases + case_ids
    X[np.arange(n)[is_treated_record], inter_idx[is_treated_record]] = 1.0
    if k:
        X[:, 2 * n_cases:] = dX

    beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
    resid = dy - X @ beta
    xtx_inv = np.linalg.pinv(X.T @ X)
    xe = X * resid[:, None]
    agg = np.zeros((n_clusters_total, X.shape[1]))
    np.add.at(agg, unit_codes, xe)
    adj = n_clusters_total / (n_clusters_total - 1)
    V_full = adj * (xtx_inv @ (agg.T @ agg) @ xtx_inv)

    att_gt = beta[n_cases:2 * n_cases]
    V = V_full[n_cases:2 * n_cases, n_cases:2 * n_cases]

    n_treated = np.zeros(n_cases)
    np.add.at(n_treated, case_ids[is_treated_record], 1.0)
    return att_gt, V, n_treated


def estimate_did(rows: list[Row], control: str = "never_treated") -> dict:
    """估计组平均处理效应、动态效应与平行趋势检验。

    参数
    ----
    rows:
        清洗后的完整面板行。
    control:
        * ``never_treated``：只用从未处理单位做对照（默认）；
        * ``not_yet_treated``：用在 t 期及紧前一期尚未处理的单位做对照。
    """
    if control not in VALID_CONTROLS:
        raise EstimationError(f"未知对照口径 {control!r}，可选 {VALID_CONTROLS}")

    units, unit_index, periods, panel, cohort, treated_units, never_units, k = _organize(rows)
    period_pos = {p: i for i, p in enumerate(periods)}

    # 队列（首次处理时期）集合，按首次处理时期排序
    groups = sorted({cohort[u] for u in treated_units})  # type: ignore[arg-type]

    # 左删失单位：首次处理就在第一个观测时期，没有处理前基期 g-1
    left_censored = [u for u in treated_units if cohort[u] == periods[0]]

    # 构造 2×2 案例 (g, t, base)：t>=g。
    # 基期取处理前一期 g-1（Callaway–Sant'Anna 长差分：处理后各期共用同一
    # 处理前基期，ATT(g,t) 是 t 相对 g-1 的累积效应）；若 g-1 不在观测网格
    # （队列恰在首个观测期，即经典 2×2 两期情形），该队列退化为以 t-1 为基期。
    #
    # 干净对照：对照单位在 t 与基期都必须未处理——任何已处理单位都不进入对照，
    # TWFE 的"已处理单位当对照"污染因此被彻底排除。
    def _eligible(g: int, t: int, base: int) -> bool:
        """该 (g,t) 是否同时存在处理组与对照的两期观测。"""
        has_t = any(
            cohort[u] == g and t in panel[u] and base in panel[u]
            for u in treated_units
        )
        if not has_t:
            return False
        for u in units:
            ps = panel[u]
            if t not in ps or base not in ps:
                continue
            if control == "never_treated":
                if cohort[u] is None:
                    return True
            elif not ps[t][1] and not ps[base][1]:
                return True
        return False

    cases: list[tuple[int, int, int]] = []
    skipped_no_control: list[tuple[int, int]] = []
    for g in groups:
        long_base = g - 1
        long_ok = long_base in period_pos
        for t in periods:
            if t < g:
                continue  # 处理前的对比放到安慰剂/平行趋势部分
            base = long_base if long_ok else t - 1
            if base not in period_pos:
                continue
            if _eligible(g, t, base):
                cases.append((g, t, base))
            else:
                skipped_no_control.append((g, t))
    cases.sort()
    if not cases:
        raise EstimationError(
            "没有可识别的 (队列, 时期) 处理后案例：可能是缺少处理后观测，"
            "或指定口径下没有任何对照单位"
        )

    # 每个案例收集 (单位, Δy, ΔX, 是否处理)；基期统一为 g-1
    records: list[tuple[int, float, np.ndarray, int]] = []
    # 每个案例的处理组单位（用于按处理组单位数加权）
    case_treated_units: list[list[str]] = [[] for _ in cases]

    for ci, (g, t, base) in enumerate(cases):
        # 处理组：队列恰为 g，且 t 与基期都在
        for u in treated_units:
            if cohort[u] != g:
                continue
            ps = panel[u]
            if t in ps and base in ps:
                yt, _, xt = ps[t]
                yb, treated_b, xb = ps[base]
                if treated_b:
                    continue  # 基期应未处理；正常不会发生
                records.append((unit_index[u], yt - yb, xt - xb, ci))
                case_treated_units[ci].append(u)
        # 对照组
        for u in units:
            ps = panel[u]
            if t not in ps or base not in ps:
                continue
            if control == "never_treated":
                if cohort[u] is not None:
                    continue
            else:  # not_yet_treated：t 与基期都尚未处理
                if ps[t][1] or ps[base][1]:
                    continue
            yt, _, xt = ps[t]
            yb, _, xb = ps[base]
            records.append((unit_index[u], yt - yb, xt - xb, ci))

    if not records:
        raise EstimationError("对照与处理组在任何 (队列, 时期) 上都没有可配对的两期观测")

    unit_codes = np.fromiter((r[0] for r in records), dtype=np.int64, count=len(records))
    dy = np.fromiter((r[1] for r in records), dtype=np.float64, count=len(records))
    if k:
        dX = np.vstack([r[2] for r in records])  # type: ignore[assignment]
    else:
        dX = np.empty((len(records), 0))
    case_ids = np.fromiter((r[3] for r in records), dtype=np.int64, count=len(records))

    # 标记每条记录是否为处理组（记录按"先处理组后对照组"加入，用集合稳妥区分）
    treated_record_codes = {
        (unit_index[u], ci) for ci, us in enumerate(case_treated_units) for u in us
    }
    is_treated_record = np.fromiter(
        ((int(unit_codes[i]), int(case_ids[i])) in treated_record_codes for i in range(len(records))),
        dtype=bool,
        count=len(records),
    )

    n_cases = len(cases)
    n_treated_case = np.array([len(v) for v in case_treated_units], dtype=np.float64)

    # -----------------------------------------------------------------------
    # 堆叠 2×2 OLS：每个案例一列截距 + 一列"案例×处理组"交互（无权重）。
    # 交互系数即各案例的均值差 ATT(g,t)；协变量以 ΔX 形式进入（跨案例共用系数）。
    # 标准误按单位聚类（CR1，G 取全面板单位数）。
    # -----------------------------------------------------------------------

    G_total = len(units)
    att_gt, V, _ = _stacked_ols(
        unit_codes, dy, dX, case_ids, is_treated_record, n_cases, G_total
    )
    skipped_cases = [{"group": g, "period": t} for (g, t) in skipped_no_control]

    se_gt = np.sqrt(np.clip(np.diag(V), 0.0, None))

    # -----------------------------------------------------------------------
    # 汇总 1：总体 ATT（按处理组单位数加权）
    # -----------------------------------------------------------------------
    w_overall = n_treated_case / n_treated_case.sum()
    att = float(w_overall @ att_gt)
    se_att = float(np.sqrt(max(0.0, w_overall @ V @ w_overall)))

    # -----------------------------------------------------------------------
    # 汇总 2：按相对时期 e = t - g 的动态效应 / 事件研究
    # -----------------------------------------------------------------------
    e_values = sorted({t - g for (g, t, _b) in cases})
    event_rows = []
    # e=-1 基准
    event_rows.append(
        {
            "relative_period": -1,
            "att": 0.0,
            "std_error": 0.0,
            "z": None,
            "p_value": None,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "n_cases": 0,
            "baseline": True,
        }
    )
    e_weight = {}
    for ci, (g, t, _b) in enumerate(cases):
        e = t - g
        e_weight.setdefault(e, np.zeros(n_cases))[ci] = n_treated_case[ci]
    for e in e_values:
        wv = e_weight[e]
        wv = wv / wv.sum()
        coef = float(wv @ att_gt)
        se = float(np.sqrt(max(0.0, wv @ V @ wv)))
        z, p = _se_z_p(coef, se)
        event_rows.append(
            {
                "relative_period": e,
                "att": coef,
                "std_error": se,
                "z": z,
                "p_value": p,
                "ci_lower": coef - 1.96 * se,
                "ci_upper": coef + 1.96 * se,
                "n_cases": int(np.count_nonzero(wv)),
                "baseline": False,
            }
        )
    event_rows.sort(key=lambda r: r["relative_period"])

    # -----------------------------------------------------------------------
    # 平行趋势：处理前 e <= -2 的系数联合 Wald（基准 e=-1）
    # 处理前系数来自"安慰剂案例"：g 队列在 t<g 且 t>=1 时的 ATT(g,t)
    # 为构造它们，需要把案例集扩展到处理前。
    # -----------------------------------------------------------------------
    pre_rows, pretrend = _pretrend(
        rows,
        control=control,
        periods=periods,
        panel=panel,
        cohort=cohort,
        treated_units=treated_units,
        units=units,
        unit_index=unit_index,
        k=k,
        G_total=G_total,
    )

    # 处理前事件研究行并入动态效应（它们不在 cases 里，单独估的）
    if pre_rows:
        known = {r["relative_period"] for r in event_rows}
        for r in pre_rows:
            if r["relative_period"] not in known:
                event_rows.append(r)
        event_rows.sort(key=lambda r: r["relative_period"])

    # -----------------------------------------------------------------------
    # 组×时期明细
    # -----------------------------------------------------------------------
    gt_rows = []
    for ci, (g, t, _b) in enumerate(cases):
        se = float(se_gt[ci])
        z, p = _se_z_p(float(att_gt[ci]), se)
        gt_rows.append(
            {
                "group": g,
                "period": t,
                "relative_period": t - g,
                "att": float(att_gt[ci]),
                "std_error": se,
                "z": z,
                "p_value": p,
                "ci_lower": float(att_gt[ci]) - 1.96 * se,
                "ci_upper": float(att_gt[ci]) + 1.96 * se,
                "n_treated": int(n_treated_case[ci]),
            }
        )

    z_att, p_att = _se_z_p(att, se_att)

    return {
        "estimand": "average_treatment_effect_on_the_treated",
        "method": "stacked_did_callaway_santanna_style",
        "control_strategy": control,
        "att": att,
        "std_error": se_att,
        "z": z_att,
        "p_value": p_att,
        "ci_lower": att - 1.96 * se_att,
        "ci_upper": att + 1.96 * se_att,
        "n_units": len(units),
        "n_periods": len(periods),
        "n_treated_units": len(treated_units),
        "n_never_treated_units": len(never_units),
        "n_left_censored_units": len(left_censored),
        "n_clusters": G_total,
        "n_covariates": k,
        "groups": groups,
        "skipped_cases_no_control": skipped_cases,
        "group_time": gt_rows,
        "event_study": event_rows,
        "pretrend_test": pretrend,
        "cluster": "unit",
        "se_adjustment": "cluster_robust_cr1",
    }


# ---------------------------------------------------------------------------
# 处理前安慰剂案例 + 平行趋势 Wald
# ---------------------------------------------------------------------------


def _pretrend(
    rows,
    *,
    control,
    periods,
    panel,
    cohort,
    treated_units,
    units,
    unit_index,
    k,
    G_total,
):
    """对每个队列构造处理前的安慰剂 2×2，估计 ATT(g,t), t<g，并按相对时期汇总。"""
    period_pos = {p: i for i, p in enumerate(periods)}

    pre_cases: list[tuple[int, int]] = []
    for u in treated_units:
        g = cohort[u]
        assert g is not None
        if (g - 1) not in period_pos:
            continue  # 左删失队列没有处理前基期
        for t in periods:
            if t >= g:
                continue
            base = g - 1  # 与处理后案例同一基期
            # 处理组两期齐全
            if not any(cohort[uu] == g and t in panel[uu] and base in panel[uu]
                       for uu in treated_units):
                continue
            # 至少一个对照
            ok = False
            for uu in units:
                ps = panel[uu]
                if t not in ps or base not in ps:
                    continue
                if control == "never_treated":
                    if cohort[uu] is None:
                        ok = True
                        break
                elif not ps[t][1] and not ps[base][1]:
                    ok = True
                    break
            if ok and (g, t) not in pre_cases:
                pre_cases.append((g, t))
    pre_cases.sort()
    if not pre_cases:
        return [], None

    records = []
    case_treated_units: list[list[str]] = [[] for _ in pre_cases]
    for ci, (g, t) in enumerate(pre_cases):
        base = g - 1
        for u in treated_units:
            if cohort[u] != g:
                continue
            ps = panel[u]
            if t in ps and base in ps:
                yt, _, xt = ps[t]
                yb, _, xb = ps[base]
                records.append((unit_index[u], yt - yb, xt - xb, ci))
                case_treated_units[ci].append(u)
        for u in units:
            ps = panel[u]
            if t not in ps or base not in ps:
                continue
            if control == "never_treated":
                if cohort[u] is not None:
                    continue
            else:
                # 尚未处理口径：t 与基期都尚未处理
                if ps[t][1] or ps[base][1]:
                    continue
            yt, _, xt = ps[t]
            yb, _, xb = ps[base]
            records.append((unit_index[u], yt - yb, xt - xb, ci))

    if not records:
        return [], None

    unit_codes = np.fromiter((r[0] for r in records), dtype=np.int64, count=len(records))
    dy = np.fromiter((r[1] for r in records), dtype=np.float64, count=len(records))
    if k:
        dX = np.vstack([r[2] for r in records])
    else:
        dX = np.empty((len(records), 0))
    case_ids = np.fromiter((r[3] for r in records), dtype=np.int64, count=len(records))
    treated_record_codes = {
        (unit_index[u], ci) for ci, us in enumerate(case_treated_units) for u in us
    }
    is_treated_record = np.fromiter(
        ((int(unit_codes[i]), int(case_ids[i])) in treated_record_codes for i in range(len(records))),
        dtype=bool,
        count=len(records),
    )
    n_cases = len(pre_cases)
    att_gt, V, n_treated_case = _stacked_ols(
        unit_codes, dy, dX, case_ids, is_treated_record, n_cases, G_total
    )

    # 按相对时期汇总（只保留 e<=-2；e=-1 是基准，恒为 0）
    e_map: dict[int, list[int]] = {}
    for ci, (g, t) in enumerate(pre_cases):
        e = t - g
        if e <= -2:
            e_map.setdefault(e, []).append(ci)

    pre_rows = []
    pre_w = []
    for e in sorted(e_map):
        idxs = e_map[e]
        wv = np.zeros(n_cases)
        denom = n_treated_case[idxs].sum()
        for ci in idxs:
            wv[ci] = n_treated_case[ci] / denom
        coef = float(wv @ att_gt)
        se = float(np.sqrt(max(0.0, wv @ V @ wv)))
        z, p = _se_z_p(coef, se)
        pre_rows.append(
            {
                "relative_period": e,
                "att": coef,
                "std_error": se,
                "z": z,
                "p_value": p,
                "ci_lower": coef - 1.96 * se,
                "ci_upper": coef + 1.96 * se,
                "n_cases": len(idxs),
                "baseline": False,
            }
        )
        pre_w.append(wv)

    pretrend = None
    if pre_w:
        L = np.vstack(pre_w)
        theta = L @ att_gt
        Vp = L @ V @ L.T
        # 用广义逆处理奇异（无噪声数据、某相对时期只有单一可识别案例等），
        # 自由度取协方差矩阵的秩。
        rank = int(np.linalg.matrix_rank(Vp, tol=1e-10))
        if rank == 0:
            all_zero = bool(np.allclose(theta, 0.0, atol=1e-10))
            if all_zero:
                stat, pval = 0.0, 1.0
            else:
                # 点估计非零而方差为零（典型：无噪声确定性数据），统计量不可有限表示
                stat, pval = None, 0.0
        else:
            stat = float(theta @ np.linalg.pinv(Vp) @ theta)
            pval = chi2_sf(stat, rank)
        pretrend = {
            "statistic": stat,
            "distribution": "chi2",
            "df": rank,
            "p_value": pval,
            "null": "所有处理前相对时期(e<=-2)的效应均为0（以e=-1为基准）",
            "relative_periods": sorted(e_map),
        }

    return pre_rows, pretrend


# ---------------------------------------------------------------------------
# 双向固定效应（TWFE）：用于展示交错情形下的偏误，不作为主估计
# ---------------------------------------------------------------------------


def _twoway_demean(y: np.ndarray, unit_codes: np.ndarray, time_codes: np.ndarray,
                   weights: np.ndarray | None = None, max_iter: int = 200,
                   tol: float = 1e-12):
    """双向固定效应吸收：迭代去掉单位均值与时期均值（允许非平衡面板）。"""
    if weights is None:
        weights = np.ones_like(y)
    z = y.copy()
    for _ in range(max_iter):
        prev = z.copy()
        # 去单位均值
        sums = np.bincount(unit_codes, weights=weights * z)
        ns = np.bincount(unit_codes, weights=weights)
        z -= (sums / ns)[unit_codes]
        # 去时期均值
        sums = np.bincount(time_codes, weights=weights * z)
        ns = np.bincount(time_codes, weights=weights)
        z -= (sums / ns)[time_codes]
        if np.max(np.abs(z - prev)) <= tol * (np.max(np.abs(y)) + 1e-12):
            break
    return z


def estimate_twfe(rows: list[Row], dynamic: bool = False) -> dict:
    """经典双向固定效应回归。

    static:  y = α_i + λ_t + τ D + Xβ + ε
    dynamic: 用相对时期虚拟变量替换 D（-1 期为基准），用于展示事件研究口径。

    标准误按单位聚类（CR1）。
    """
    units = sorted({r.unit for r in rows})
    unit_index = {u: i for i, u in enumerate(units)}
    periods = sorted({r.period for r in rows})
    time_index = {p: i for i, p in enumerate(periods)}
    k = len(rows[0].covariates)

    ordered = sorted(rows, key=lambda r: (r.unit, r.period))
    n = len(ordered)
    y = np.array([r.outcome for r in ordered], dtype=float)
    uc = np.array([unit_index[r.unit] for r in ordered], dtype=np.int64)
    tc = np.array([time_index[r.period] for r in ordered], dtype=np.int64)
    d = np.array([1.0 if r.treated else 0.0 for r in ordered], dtype=float)
    Xc = np.vstack([np.asarray(r.covariates, dtype=float) for r in ordered]) if k else np.empty((n, 0))

    cohort = {}
    panel: dict[str, dict[int, bool]] = {}
    for r in ordered:
        panel.setdefault(r.unit, {})[r.period] = r.treated
    for u, ps in panel.items():
        cohort[u] = min((p for p, t in ps.items() if t), default=None)
    has_treated = any(v is not None for v in cohort.values())
    if not has_treated:
        raise EstimationError("没有任何单位接受处理，TWFE 无法估计处理效应")

    # 相对时期（未处理单位为 None）
    def rel_period(r):
        g = cohort[r.unit]
        return None if g is None else r.period - g

    if dynamic:
        rels = sorted({rp for r in ordered for rp in [rel_period(r)] if rp is not None and rp != -1})
        D = np.zeros((n, len(rels)))
        for j, e in enumerate(rels):
            for i, r in enumerate(ordered):
                rp = rel_period(r)
                if rp == e:
                    D[i, j] = 1.0
        col_names = [f"event_time_{e}" for e in rels]
    else:
        D = d[:, None]
        col_names = ["treated"]
        rels = None

    X = np.hstack([D, Xc]) if k else D
    ncol = X.shape[1]

    # 吸收双向固定效应
    y_star = _twoway_demean(y, uc, tc)
    X_star = np.column_stack([_twoway_demean(X[:, j], uc, tc) for j in range(ncol)])

    beta, *_ = np.linalg.lstsq(X_star, y_star, rcond=None)
    resid = y_star - X_star @ beta

    G = len(units)
    adj = G / (G - 1)
    xtx_inv = np.linalg.pinv(X_star.T @ X_star)
    xr = X_star * resid[:, None]
    agg = np.zeros((G, ncol))
    np.add.at(agg, uc, xr)
    Vbeta = adj * (xtx_inv @ (agg.T @ agg) @ xtx_inv)
    se = np.sqrt(np.clip(np.diag(Vbeta), 0.0, None))

    results = []
    for j, name in enumerate(col_names):
        z, p = _se_z_p(float(beta[j]), float(se[j]))
        results.append(
            {
                "term": name,
                "coef": float(beta[j]),
                "std_error": float(se[j]),
                "z": z,
                "p_value": p,
                "ci_lower": float(beta[j]) - 1.96 * float(se[j]),
                "ci_upper": float(beta[j]) + 1.96 * float(se[j]),
            }
        )

    return {
        "method": "two_way_fixed_effects_ols",
        "n_obs": n,
        "n_units": G,
        "n_periods": len(periods),
        "cluster": "unit",
        "coef_static": float(beta[0]) if not dynamic else None,
        "se_static": float(se[0]) if not dynamic else None,
        "coefficients": results,
        "warning": (
            "TWFE 在交错处理+异质效应下会用已处理单位做对照，系数可能有偏甚至反号；"
            "请以 stacked_did 结果为准。"
        ),
    }

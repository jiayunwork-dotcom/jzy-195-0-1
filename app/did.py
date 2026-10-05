"""双重差分估计引擎：全部回归、聚类标准误与估计量均为自行实现。

不依赖任何计量/统计建模库（仅用 NumPy 做矩阵运算）。

口径概述（详见 README「方法论」一节）：
  * 目标量为组别-时期平均处理效应 ATT(g, t)：对处理组 g（在 g 期开始处理的
    一组单位）在时期 t 的 2x2 双重差分（处理组从“最近的处理前时期”到 t 的
    变化，减去对照组同期变化）。
  * 对照组默认只用“从未处理”单位（control_group="never"）；可选
    control_group="not_yet"（尚未处理：参照期 b<g 且 t 期均未处理）。
    已处理单位永远不会进入对照，因此避免了交错处理 + 异质性动态效应下
    TWFE 的“禁止比较”偏误。
  * 汇总：总 ATT 按各处理组在处理开始时的规模 n_g 加权；动态事件研究按
    相对时期 e=t-g 聚合（组内仍按 n_g 加权）。
  * 标准误：在单位层面对角求和（CR1，Stata 默认小样本修正因子
    G/(G-1) * (N-1)/(N-k)）。
  * 另在同一份数据上拟合 TWFE（单位与时期固定效应）作为诊断，展示偏误。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

ENGINE_VERSION = "did-cs-2x2-v1"

_YEAR_RE = re.compile(r"^(\d{4})$")
_QUARTER_RE = re.compile(r"^(\d{4})-?[qQ]([1-4])$")


class PanelValidationError(ValueError):
    """面板数据或分析设定不合法。code 为机器可读的错误码，field 指出问题字段。"""

    def __init__(self, code: str, message: str, field: str | None = None):
        super().__init__(message)
        self.code = code
        self.field = field
        self.message = message


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Row:
    unit: str
    period: str
    outcome: float
    treated: bool
    covariates: tuple[tuple[str, float], ...]


@dataclass
class Panel:
    """规范化后的面板：行列对齐、时期已映射为可排序的序号。"""

    units: np.ndarray            # shape (n,), 字符串
    periods: np.ndarray          # shape (T,), 已排序的原始时期标签
    period_index: dict[str, int]
    frequency: str               # "year" | "quarter"
    # 形状 (n, T) 的稠密数组；缺失观测为 nan
    Y: np.ndarray
    D: np.ndarray                # bool
    has: np.ndarray              # bool，该单位该期是否有观测
    cov_names: list[str]
    X: np.ndarray                # shape (n, T, p)，缺失观测不参与差分
    dropped_units: list[str]     # 因没有任何有效观测被丢弃的单位（正常为空）


# --------------------------------------------------------------------------- #
# 解析与校验
# --------------------------------------------------------------------------- #
def _parse_period_label(label: str) -> tuple[str, int, int]:
    """返回 (frequency, 可比较序号, 规范化标签)。

    支持：
      * 年度："2015"
      * 季度："2015Q1" / "2015q1" / "2015-Q1"
    整个数据集的频率必须一致。
    """
    m = _YEAR_RE.match(label)
    if m:
        y = int(m.group(1))
        return "year", y, f"{y:04d}"
    m = _QUARTER_RE.match(label)
    if m:
        y, q = int(m.group(1)), int(m.group(2))
        return "quarter", y * 4 + (q - 1), f"{y:04d}Q{q}"
    raise PanelValidationError(
        "period_unparseable",
        f"时期 {label!r} 无法解析为年度(YYYY)或季度(YYYYQq)，因此无法排序",
        field="period",
    )


def _to_float(value: Any, field: str, unit: Any = None, period: Any = None) -> float:
    if isinstance(value, bool):
        # bool 是 int 的子类，显式拒绝以免歧义。
        raise PanelValidationError(
            f"{field}_not_numeric",
            f"字段 {field!r} 必须是数值（单位 {unit!r}，时期 {period!r} 收到布尔值）",
            field=field,
        )
    if isinstance(value, (int, float)):
        f = float(value)
    elif isinstance(value, str):
        try:
            f = float(value.strip())
        except ValueError:
            f = float("nan")
    else:
        f = float("nan")
    if not math.isfinite(f):
        raise PanelValidationError(
            f"{field}_not_numeric",
            f"字段 {field!r} 必须是有限数值（单位 {unit!r}，时期 {period!r}）",
            field=field,
        )
    return f


def normalize_rows(raw_rows: Iterable[dict]) -> list[Row]:
    """把接口收到的行字典规范化为 Row 列表，完成逐行校验。"""
    rows: list[Row] = []
    seen: set[tuple[str, str]] = set()

    for i, r in enumerate(raw_rows):
        if not isinstance(r, dict):
            raise PanelValidationError(
                "row_not_object", f"第 {i} 行不是对象", field="rows"
            )
        for name in ("unit", "period", "outcome", "treated"):
            if name not in r:
                raise PanelValidationError(
                    "missing_field",
                    f"第 {i} 行缺少必需字段 {name!r}",
                    field=name,
                )

        unit = r["unit"]
        period = r["period"]
        if not isinstance(unit, (str, int)) or isinstance(unit, bool) or str(unit) == "":
            raise PanelValidationError(
                "unit_invalid",
                f"第 {i} 行的 unit 必须是非空字符串或整数",
                field="unit",
            )
        unit = str(unit)
        if not isinstance(period, (str, int)) or isinstance(period, bool) or str(period) == "":
            raise PanelValidationError(
                "period_invalid",
                f"第 {i} 行的 period 必须是非空字符串或整数",
                field="period",
            )
        period = str(period).strip()
        freq, _, norm_period = _parse_period_label(period)

        outcome = _to_float(r["outcome"], "outcome", unit, period)

        treated_raw = r["treated"]
        if isinstance(treated_raw, bool):
            treated = treated_raw
        elif isinstance(treated_raw, (int, float)) and treated_raw in (0, 1) and not isinstance(
            treated_raw, bool
        ):
            treated = bool(int(treated_raw))
        else:
            raise PanelValidationError(
                "treated_not_binary",
                f"treated 必须为 0/1 或 true/false（单位 {unit!r}，时期 {period!r}）",
                field="treated",
            )

        covs_raw = r.get("covariates")
        cov_items: tuple[tuple[str, float], ...] = ()
        if covs_raw is not None:
            if not isinstance(covs_raw, dict):
                raise PanelValidationError(
                    "covariates_not_object",
                    f"covariates 必须是对象（单位 {unit!r}，时期 {period!r}）",
                    field="covariates",
                )
            tmp: list[tuple[str, float]] = []
            for k, v in covs_raw.items():
                tmp.append((str(k), _to_float(v, f"covariate:{k}", unit, period)))
            tmp.sort(key=lambda kv: kv[0])
            cov_items = tuple(tmp)

        key = (unit, norm_period)
        if key in seen:
            raise PanelValidationError(
                "duplicate_unit_period",
                f"同一单位 {unit!r} 在同一时期 {norm_period!r} 出现了多于一行",
                field="unit,period",
            )
        seen.add(key)
        rows.append(Row(unit, norm_period, outcome, treated, cov_items))

    if not rows:
        raise PanelValidationError("empty_dataset", "数据为空：至少需要一行", field="rows")
    return rows


def build_panel(raw_rows: Iterable[dict]) -> Panel:
    """规范化行并组装成稠密面板，同时做跨行校验（处理路径一致性、频率一致性）。"""
    rows = normalize_rows(raw_rows)

    units = sorted({r.unit for r in rows})
    unit_idx = {u: i for i, u in enumerate(units)}

    frequency = _parse_period_label(rows[0].period)[0]
    parsed: set[tuple[str, int]] = set()
    for r in rows:
        f, ordinal, label = _parse_period_label(r.period)
        if f != frequency:
            raise PanelValidationError(
                "period_frequency_mixed",
                f"时期频率不一致：{label!r} 为 {f}，但数据集首条为 {frequency}（年度与季度不能混用）",
                field="period",
            )
        parsed.add((label, ordinal))
    periods = [lab for lab, _ in sorted(parsed, key=lambda x: x[1])]
    period_index = {p: i for i, p in enumerate(periods)}

    n, T = len(units), len(periods)
    Y = np.full((n, T), np.nan)
    D = np.zeros((n, T), dtype=bool)
    has = np.zeros((n, T), dtype=bool)

    cov_names: list[str] | None = None
    for r in rows:
        keys = [k for k, _ in r.covariates]
        if cov_names is None:
            cov_names = keys
        elif keys != cov_names:
            raise PanelValidationError(
                "covariate_keys_inconsistent",
                f"单位 {r.unit!r} 时期 {r.period!r} 的协变量键集合与其他行不一致："
                f"{keys} != {cov_names}",
                field="covariates",
            )
    cov_names = cov_names or []
    p = len(cov_names)
    X = np.full((n, T, p), np.nan)

    for r in rows:
        i, t = unit_idx[r.unit], period_index[r.period]
        Y[i, t] = r.outcome
        D[i, t] = r.treated
        has[i, t] = True
        for j, (_, v) in enumerate(r.covariates):
            X[i, t, j] = v

    # 处理开始后不能撤回：每个单位的处理指示必须是 0...0 1...1（忽略缺失期）。
    for i, u in enumerate(units):
        observed = np.where(has[i])[0]
        d = D[i, observed]
        bad = np.flatnonzero(d[:-1] & ~d[1:])
        if bad.size:
            t0 = observed[int(bad[0]) + 1]
            raise PanelValidationError(
                "treatment_reversed",
                f"单位 {u!r} 在处理开始后又回到未处理（时期 {periods[t0]!r} 处 treated=0）",
                field="treated",
            )

    return Panel(
        units=np.asarray(units, dtype=object),
        periods=np.asarray(periods, dtype=object),
        period_index=period_index,
        frequency=frequency,
        Y=Y,
        D=D,
        has=has,
        cov_names=cov_names,
        X=X,
        dropped_units=[],
    )


# --------------------------------------------------------------------------- #
# 分布工具：正态、卡方 p 值（不调用统计库）
# --------------------------------------------------------------------------- #
def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _chi2_sf(x: float, df: int) -> float:
    """正则化下不完全伽马 Q(df/2, x/2) = P(chi2_df > x)。"""
    if x <= 0.0 or df <= 0:
        return 1.0
    a = df / 2.0
    x = x / 2.0
    if x < a + 1.0:
        # 级数展开
        ap = a
        total = 1.0 / a
        term = total
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, min(1.0, total * math.exp(-x + a * math.log(x) - math.lgamma(a))))
    # 连分式展开
    b = x + 1.0 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 1001):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return max(0.0, min(1.0, math.exp(-x + a * math.log(x) - math.lgamma(a)) * h))


# --------------------------------------------------------------------------- #
# 主估计
# --------------------------------------------------------------------------- #
def _solve_coef(xtx: np.ndarray, xty: np.ndarray) -> np.ndarray | None:
    try:
        return np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return None


def estimate(
    panel: Panel,
    *,
    control_group: str = "never",
    adjust_covariates: bool = False,
    include_twfe: bool = True,
) -> dict:
    """估计组别-时期 ATT、动态效应、处理前平行趋势联合检验与 TWFE 诊断。"""
    if control_group not in ("never", "not_yet"):
        raise PanelValidationError(
            "control_group_invalid",
            "control_group 只能是 'never' 或 'not_yet'",
            field="control_group",
        )

    n, T = panel.has.shape
    Y, D, has, X = panel.Y, panel.D, panel.has, panel.X

    # g_i：每个单位的处理开始时期序号（位置），从未处理为 -1。
    first_treat = np.full(n, -1, dtype=int)
    for i in range(n):
        obs = np.where(has[i])[0]
        if obs.size and bool(D[i, obs[0]]):
            raise PanelValidationError(
                "treatment_at_first_period",
                f"单位 {panel.units[i]!r} 在首个观测期即已处理，没有处理前观测，无法估计",
                field="treated",
            )
        treated_obs = obs[D[i, obs]]
        if treated_obs.size:
            first_treat[i] = int(treated_obs[0])

    cohorts = sorted(int(g) for g in set(first_treat.tolist()) if g >= 0)
    if not cohorts:
        raise PanelValidationError(
            "no_treated_units", "没有任何单位接受处理（treated 全为 0），无法估计处理效应"
        )

    never = np.where(first_treat < 0)[0]
    if control_group == "never" and never.size == 0:
        raise PanelValidationError(
            "no_never_treated_controls",
            "数据中没有从未处理的单位；若接受“尚未处理”对照口径，"
            "请改用 control_group='not_yet'",
        )

    p = panel.X.shape[2]
    use_x = bool(adjust_covariates and p > 0)
    kx = p if use_x else 0

    # ------------------------------------------------------------------ #
    # 构造 2x2 单元（cell）。
    # 每个 cell 形如 (g, a, b, kind)：对处理组 g 在 a 期的结果变化（相对
    # 参照期 b=a 的最近处理前时期）做差分；post 的系数即 ATT(g,a)，
    # pre  的系数即安慰剂效应（用于平行趋势与动态展开的处理前部分）。
    # ------------------------------------------------------------------ #
    cell_specs: list[dict] = []
    skipped: list[dict] = []

    def control_ok(i: int, a: int, b: int, g: int) -> bool:
        if not (has[i, a] and has[i, b]):
            return False
        gi = int(first_treat[i])
        if gi < 0:  # 从未处理
            return True
        # 尚未处理：参照期 b 与结果期 a 都必须严格早于其处理开始。
        return gi > a  # b < a，故 b 自动满足

    for g in cohorts:
        g_units = np.where(first_treat == g)[0]
        # 该组最近的处理前观测位置（所有同组单位的差分参照逐单位各自取，
        # 但为保持 2x2 干净且可解释，这里取该组共同使用的“最近处理前时期”
        # 中全组都观测到的一期；缺失该期的单位在逐单元层面会被剔除）。
        pre_candidates = [t for t in range(g - 1, -1, -1) if has[g_units, t].all()]
        if not pre_candidates:
            for i in g_units:
                skipped.append(
                    {"unit": str(panel.units[i]), "reason": "cohort_without_common_baseline"}
                )
            continue
        b_common = pre_candidates[0]

        for a in range(T):
            if a == b_common:
                continue
            if a < g:
                kind = "pre"
            elif a >= g:
                kind = "post"
            else:
                continue
            treated_members = [
                int(i)
                for i in g_units
                if has[i, a] and has[i, b_common]
            ]
            ctrl = np.array(
                [
                    int(i)
                    for i in range(n)
                    if int(first_treat[i]) != g and control_ok(i, a, b_common, g)
                ],
                dtype=int,
            )
            # not_yet 口径下尚未处理单位已经在 control_ok 里放行；
            # never 口径则只保留从未处理单位。
            if control_group == "never":
                ctrl = ctrl[first_treat[ctrl] < 0]
            if not treated_members or ctrl.size == 0:
                skipped.append(
                    {
                        "cohort": _period_label(panel, g),
                        "period": _period_label(panel, a),
                        "kind": kind,
                        "reason": "empty_treated_or_control_cell",
                    }
                )
                continue
            cell_specs.append(
                dict(
                    g=g,
                    a=a,
                    b=b_common,
                    kind=kind,
                    # 事件时间：处理后单元相对队列开始 a-g（b=g-1 时与 a-b-1 等价）；
                    # 处理前安慰剂单元相对其差分参照期 b 定义，即 a-b（恒为负）。
                    event_offset=int(a - g) if a >= g else int(a - b_common),
                    treated_units=np.asarray(treated_members, dtype=int),
                    control_units=ctrl,
                )
            )

    if not cell_specs:
        raise PanelValidationError(
            "no_estimable_cells", "没有任何可估计的处理组-时期单元（对照或基线缺失）"
        )

    # ------------------------------------------------------------------ #
    # 逐 cell 堆叠的 2x2（一阶差分）OLS：
    #   Δy_i = α_p + β_p · D_i^g  (+ Γ_p Δx_i) + e_i
    # β_p 即 ATT(g,a)。所有 cell 横向拼成块状矩阵做联合求解，
    # 以便按单位聚类计算覆盖跨 cell 相关性的协方差。
    # ------------------------------------------------------------------ #
    rows_y: list[float] = []
    rows_unit: list[int] = []
    rows_cell: list[int] = []
    rows_treat: list[float] = []
    rows_xdiff: list[list[float]] = []
    # 每个 cell 实际保留的协变量列（差分后方差为 0 的列与截距共线，剔除）。
    cell_xcols: dict[int, list[int]] = {}

    for pid, cell in enumerate(cell_specs):
        g, a, b = cell["g"], cell["a"], cell["b"]
        members = np.concatenate([cell["treated_units"], cell["control_units"]])
        treat_flag = np.concatenate(
            [
                np.ones(cell["treated_units"].size, dtype=float),
                np.zeros(cell["control_units"].size, dtype=float),
            ]
        )
        dy = Y[members, a] - Y[members, b]
        if use_x:
            dx = X[members, a] - X[members, b]  # type: ignore[name-defined]
        else:
            dx = None
        # 剔除差分后含 nan 的行（协变量缺失时）。
        finite = np.isfinite(dy)
        if dx is not None:
            finite &= np.isfinite(dx).all(axis=1)
        keep = np.flatnonzero(finite)
        members, treat_flag, dy = members[keep], treat_flag[keep], dy[keep]
        if dx is not None:
            dx = dx[keep]

        active_xcols: list[int] = []
        if use_x:
            # 保留在当前 2x2 样本中有变异的协变量；恒定列与截距共线，剔除。
            dxl = dx.tolist()
            for j2 in range(kx):
                col = [row[j2] for row in dxl]
                if max(col) - min(col) > 1e-12:
                    active_xcols.append(j2)
        cell_xcols[pid] = active_xcols

        # 保证两组都还在。
        if treat_flag.sum() == 0 or (1 - treat_flag).sum() == 0:
            skipped.append(
                {
                    "cohort": _period_label(panel, g),
                    "period": _period_label(panel, a),
                    "kind": cell["kind"],
                    "reason": "cell_degenerate_after_dropping_missing",
                }
            )
            # 从 specs 中移除该 cell（打标记，过滤后重建索引）
            cell["_drop"] = True
            continue
        for row_i, u in enumerate(members):
            rows_y.append(float(dy[row_i]))
            rows_unit.append(int(u))
            rows_cell.append(pid)
            rows_treat.append(float(treat_flag[row_i]))
            rows_xdiff.append(
                [dx[row_i, j2] for j2 in active_xcols] if use_x else []
            )

    live_cells = [c for c in cell_specs if not c.get("_drop")]
    if not live_cells:
        raise PanelValidationError("no_estimable_cells", "差分后没有可估计的单元")

    P = len(live_cells)
    # 重新映射旧 pid -> 新 pid（被标记删除的 cell 不参与联合估计）
    old_to_new: dict[int, int] = {}
    newj = 0
    for old_pid, c in enumerate(cell_specs):
        if not c.get("_drop"):
            old_to_new[old_pid] = newj
            newj += 1

    # 每个 cell 的参数维度（截距 + 处理虚拟变量 + 该 cell 保留的协变量斜率）。
    ordered_old_pids = [
        old_pid
        for old_pid, c in enumerate(cell_specs)
        if not c.get("_drop")
    ]
    k_list = [2 + len(cell_xcols.get(old_pid, [])) for old_pid in ordered_old_pids]
    offsets = []
    cur = 0
    for kk in k_list:
        offsets.append(cur)
        cur += kk
    K = cur
    new_pid_k = {newj: k_list[newj] for newj in range(P)}
    new_pid_off = {newj: offsets[newj] for newj in range(P)}

    Nrows = len(rows_y)
    max_k = max(k_list)
    # 直接累加法方程，内存友好；显式标量索引以兼容最简数组后端。
    XtX = np.zeros((K, K))
    XtY = np.zeros(K)
    # 保存每行的非零设计块（截距、处理虚拟变量、协变量差分）。
    design = np.zeros((Nrows, max_k))
    design_nk = np.asarray(
        [new_pid_k[old_to_new[c]] for c in rows_cell], dtype=int
    )
    design_cell = np.asarray([old_to_new[c] for c in rows_cell], dtype=int)
    design_unit = np.asarray(rows_unit, dtype=int)
    yv = np.asarray(rows_y)

    for r in range(Nrows):
        nk = int(design_nk[r])
        v = [0.0] * nk
        v[0] = 1.0
        v[1] = rows_treat[r]
        xv = rows_xdiff[r]
        for j2 in range(nk - 2):
            v[2 + j2] = xv[j2]
        for j2 in range(nk):
            design[r, j2] = v[j2]
        off = int(new_pid_off[int(design_cell[r])])
        yr = yv[r]
        for a2 in range(nk):
            XtY[off + a2] += v[a2] * yr
            for b2 in range(a2, nk):
                val = v[a2] * v[b2]
                XtX[off + a2, off + b2] += val
                if a2 != b2:
                    XtX[off + b2, off + a2] += val

    beta = _solve_coef(XtX, XtY)
    if beta is None:
        raise PanelValidationError(
            "design_singular", "设计矩阵奇异（通常是某个单元内两组完全共线）"
        )

    fitted = np.zeros(Nrows)
    for r in range(Nrows):
        nk = int(design_nk[r])
        off = int(new_pid_off[int(design_cell[r])])
        s = 0.0
        for j2 in range(nk):
            s += float(design[r, j2]) * beta[off + j2]
        fitted[r] = s
    resid = yv - fitted

    # ------------------------------------------------------------------ #
    # 按单位聚类的肉矩阵（CR1）。
    # M = sum_g X_g' e_g e_g' X_g ；V = factor * (X'X)^-1 M (X'X)^-1
    # ------------------------------------------------------------------ #
    XtX_inv = np.linalg.inv(XtX)
    meat = np.zeros((K, K))
    G_clusters = 0
    for u in np.unique(design_unit):
        idx = np.flatnonzero(design_unit == u)
        scores = np.zeros(K)
        for r in idx.tolist():
            nk = int(design_nk[r])
            off = int(new_pid_off[int(design_cell[r])])
            er = resid[r]
            for j2 in range(nk):
                scores[off + j2] += float(design[r, j2]) * er
        meat += np.outer(scores, scores)
        G_clusters += 1

    cluster_factor = G_clusters / (G_clusters - 1)
    # 残差自由度：N 不大于参数数时（完全拟合的小样本，如教科书 2x2）
    # 无法做标准异方差修正；残差为零时聚类肉矩阵本身为零，SE 即零。
    # 此时保留聚类修正 G/(G-1)，但不乘会爆炸的 (N-1)/(N-K)。
    if G_clusters <= 1:
        raise PanelValidationError(
            "insufficient_clusters",
            "单位聚类数少于 2，无法计算聚类标准误",
        )
    if Nrows > K:
        dof_factor = cluster_factor * ((Nrows - 1) / (Nrows - K))
    else:
        dof_factor = cluster_factor
    V = dof_factor * (XtX_inv @ meat @ XtX_inv)

    # 各 cell 的 ATT 系数位置（每个 cell 块的第 2 列）。
    att_beta = np.asarray(
        [new_pid_off[j] + 1 for j in range(P)], dtype=int
    )
    att = np.asarray([beta[j] for j in att_beta.tolist()])
    # V[np.ix_(att_beta, att_beta)]：分两次一维花式索引取对称子矩阵
    Vatt = np.asarray(
        [[V[i, j] for j in att_beta.tolist()] for i in att_beta.tolist()]
    )
    se = np.sqrt(np.clip(np.asarray([Vatt[j, j] for j in range(P)]), 0.0, None))

    # ------------------------------------------------------------------ #
    # 汇总权重
    # n_g：处理开始期实际观测到的处理组成员数。
    # ------------------------------------------------------------------ #
    n_g = {g: int(has[np.where(first_treat == g)[0], g].sum()) for g in cohorts}

    def cohort_weight_vec(g: int, kinds: set[str]) -> np.ndarray:
        w = np.zeros(P)
        n_total = 0.0
        for j, c in enumerate(live_cells):
            if c["g"] == g and c["kind"] in kinds:
                n_total += n_g[c["g"]]
        for j, c in enumerate(live_cells):
            if c["g"] == g and c["kind"] in kinds:
                w[j] = n_g[c["g"]]
        if w.sum() > 0:
            w /= w.sum()
        return w

    # 总 ATT（Callaway–Sant'Anna 汇总）：每一个处理后单元 ATT(g,t) 的权重
    # 与处理组规模 n_g 成正比，对全部可用的 post 单元归一化。这等价于
    # “对所有处理单位的所有处理后期求平均”，避免早处理组因其时期更多而
    # 被重复计权。
    w_overall = np.zeros(P)
    for j, c in enumerate(live_cells):
        # 只有“该处理组在自身处理开始之后”的单元才计入总 ATT；
        # 后处理组的处理前安慰剂单元（kind='pre'）绝不参与。
        if c["kind"] == "post" and c["a"] >= c["g"]:
            w_overall[j] = n_g[c["g"]]
    if float(w_overall.sum()) > 0:
        w_overall /= float(w_overall.sum())

    # ------------------------------------------------------------------ #
    # 动态效应：按相对时期 e = a - g
    # ------------------------------------------------------------------ #
    es: list[int] = sorted({int(c["event_offset"]) for c in live_cells})
    es_rows = []
    W_es = np.zeros((len(es), P))
    for row_i, e in enumerate(es):
        gs_here = [g for g in cohorts if any(
            c["g"] == g and int(c["event_offset"]) == e for c in live_cells
        )]
        w = np.zeros(P)
        for g in gs_here:
            wg = np.zeros(P)
            for j, c in enumerate(live_cells):
                if c["g"] == g and int(c["event_offset"]) == e:
                    wg[j] = n_g[g]
            if wg.sum() > 0:
                wg /= wg.sum()
            share = n_g[g] / sum(n_g[gg] for gg in gs_here)
            w += share * wg
        W_es[row_i] = w

    theta_es = W_es @ att
    V_es = W_es @ Vatt @ W_es.T
    se_es = np.sqrt(np.clip(np.asarray([V_es[i, i] for i in range(len(es))]), 0.0, None))

    # ------------------------------------------------------------------ #
    # 平行趋势：处理前各 e（e<0）联合 Wald（H0: 全部为 0）。
    # ------------------------------------------------------------------ #
    pre_idx = [i for i, e in enumerate(es) if e < 0]
    if pre_idx:
        R = W_es[pre_idx]
        rtheta = R @ att
        RvR = R @ Vatt @ R.T
        try:
            inv = np.linalg.inv(RvR)
            wald = float(rtheta @ inv @ rtheta)
            df = len(pre_idx)
            p_chi2 = _chi2_sf(wald, df)
        except np.linalg.LinAlgError:
            wald, p_chi2, df = float("nan"), float("nan"), len(pre_idx)
    else:
        wald, p_chi2, df = float("nan"), float("nan"), 0

    def _ci(est: float, s: float) -> list[float]:
        if not math.isfinite(s) or s == 0.0:
            return [est, est]
        return [est - 1.959963984540054 * s, est + 1.959963984540054 * s]

    dynamic = []
    for i, e in enumerate(es):
        dynamic.append(
            {
                "relative_period": int(e),
                "period_type": "post" if e >= 0 else "pre",
                "estimate": float(theta_es[i]),
                "std_error": float(se_es[i]),
                "ci95": [float(x) for x in _ci(theta_es[i], se_es[i])],
            }
        )

    overall_est = float(w_overall @ att)
    overall_se = float(math.sqrt(max(0.0, w_overall @ Vatt @ w_overall)))

    # 各组 ATT（诊断用）：组内对各处理后单元等权（即按 n_g 组内统一）。
    cohorts_with_post = [
        g for g in cohorts
        if any(c["g"] == g and c["kind"] == "post" for c in live_cells)
    ]
    cohort_atts = []
    for g in cohorts_with_post:
        wg = cohort_weight_vec(g, {"post"})
        est_g = float(wg @ att)
        se_g = float(math.sqrt(max(0.0, wg @ Vatt @ wg)))
        cohort_atts.append(
            {
                "cohort": _period_label(panel, g),
                "n_units_at_onset": n_g[g],
                "estimate": est_g,
                "std_error": se_g,
                "ci95": [float(x) for x in _ci(est_g, se_g)],
            }
        )

    # 全部 cell 的 ATT(g,t) 明细
    cells_out = []
    for j, c in enumerate(live_cells):
        cells_out.append(
            {
                "cohort": _period_label(panel, c["g"]),
                "period": _period_label(panel, c["a"]),
                "relative_period": int(c["event_offset"]),
                "kind": c["kind"],
                "estimate": float(att[j]),
                "std_error": float(se[j]),
                "n_treated": int(c["treated_units"].size),
                "n_control": int(c["control_units"].size),
                "reference_period": _period_label(panel, c["b"]),
            }
        )

    result: dict = {
        "engine_version": ENGINE_VERSION,
        "control_group": control_group,
        "adjust_covariates": bool(use_x),
        "covariates": panel.cov_names if use_x else [],
        "n_units": int(n),
        "n_periods": int(T),
        "periods": [str(x) for x in panel.periods.tolist()],
        "att": {
            "estimate": overall_est,
            "std_error": overall_se,
            "ci95": [float(x) for x in _ci(overall_est, overall_se)],
            "weights": "group_size_n_g",
        },
        "dynamic_effects": dynamic,
        "pretrend_test": {
            "type": "wald_joint_zero",
            "statistic": float(wald) if math.isfinite(wald) else None,
            "distribution": f"chi2({df})" if df else None,
            "df": int(df),
            "p_value": float(p_chi2) if math.isfinite(p_chi2) else None,
            "note": "处理前各期（相对时期<0）动态效应联合为零的 Wald 检验",
        },
        "cohort_att": cohort_atts,
        "group_time_att": cells_out,
        "standard_errors": {
            "clustered_by": "unit",
            "type": "CR1",
            "small_sample_factor": float(dof_factor),
            "n_clusters": int(G_clusters),
            "n_obs_in_design": int(Nrows),
        },
        "skipped": skipped,
        "twfe": _twfe_diagnostic(panel, compute=include_twfe),
    }
    return result


def _period_label(panel: Panel, t: int) -> str:
    return str(panel.periods[t])


# --------------------------------------------------------------------------- #
# TWFE 诊断：y_it = alpha_i + gamma_t + tau · D_it + e_it
# 用双向 within 变换（迭代去均值）+ 单位聚类标准误，全部自行实现。
# --------------------------------------------------------------------------- #
def _twfe_diagnostic(panel: Panel, compute: bool = True) -> dict | None:
    if not compute:
        return None
    Y, D, has = panel.Y, panel.D, panel.has
    n, T = has.shape
    idx = np.argwhere(has)
    flat = idx[:, 0] * T + idx[:, 1]
    y = Y.reshape(-1)[flat]
    d = D.reshape(-1)[flat].astype(float)
    ui = idx[:, 0].astype(int)
    ti = idx[:, 1].astype(int)
    N = y.size

    # 迭代双向去均值（FWL）：z 为去单位/时期均值后的 D，yy 为去均值后的 y。
    z = d.copy()
    yy = y.copy()

    def _demean(arr, group_idx, g):
        mu = np.zeros(g)
        cnt = np.zeros(g)
        np.add.at(mu, group_idx, arr)
        np.add.at(cnt, group_idx, 1.0)
        # 只对被观测到的组取均值，避免未观测组 0/0。
        means = np.asarray(
            [mu[j] / cnt[j] if cnt[j] > 0 else 0.0 for j in range(g)]
        )
        return arr - means[group_idx]

    for _ in range(10000):
        z0 = z.copy()
        z = _demean(z, ui, n)
        yy = _demean(yy, ui, n)
        z = _demean(z, ti, T)
        yy = _demean(yy, ti, T)
        # 截距与单位/时期固定效应共线，去均值后整体平移不影响斜率，减去防漂移。
        z = z - float(np.mean(z))
        if float(np.max(np.abs(z - z0))) < 1e-12 * (float(np.max(np.abs(z0))) + 1e-12):
            break
    denom = float(z @ z)
    if denom <= 0:
        raise PanelValidationError("twfe_singular", "TWFE 处理变量在去均值后无变化")
    tau = float((z @ yy) / denom)
    resid = yy - tau * z

    # 聚类（单位）方差：score_i = sum_t z_it e_it；V = G/(G-1)*(N-1)/(N-k)
    # × sum score^2 / (z'z)^2 （k=1：只估计斜率；固定效应被吸收）。
    score = np.zeros(n)
    np.add.at(score, ui, z * resid)
    G = int(np.unique(ui).size)
    factor = (G / (G - 1)) * ((N - 1) / (N - 1))
    var = factor * float(score @ score) / (denom * denom)
    se_tau = math.sqrt(max(0.0, var))

    return {
        "estimate": tau,
        "std_error": se_tau,
        "note": "双向固定效应（单位+时期），仅作诊断；交错处理+异质动态效应下"
        "会把已处理单位当作对照，可能产生严重偏误甚至符号相反",
    }

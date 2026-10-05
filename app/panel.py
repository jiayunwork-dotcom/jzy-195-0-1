"""面板数据校验。

校验规则（任何一条不满足都返回 422，并指出字段或原因）：

1. 必须含 ``unit``、``period``、``outcome``、``treated`` 四个字段；
2. 同一单位同一时期不得出现两行；
3. 处理状态只能单调：一旦在某期处理，之后所有有观测的时期必须仍在处理；
4. 时期必须可排序：全数据集使用同一套口径（年份整数/四位年份字符串，或
   ``YYYYQn`` 季度字符串），且能解析；
5. 结果变量必须是数值（且有限）；
6. 至少存在一个被处理单位时才能估计（该校验在估计阶段抛出，上传时允许，
   例如先存数据再做安慰剂分析）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .errors import ValidationError

_QUARTER_RE = re.compile(r"^(\d{4})[Qq]([1-4])$")
_YEAR_RE = re.compile(r"^\d{4}$")


# ---------------------------------------------------------------------------
# 时期解析：返回 (排序索引, 规范化标签, 类型)
# ---------------------------------------------------------------------------


def period_kind(period) -> str:
    """判断时期类型：'year' 或 'quarter'。无法识别返回 'unknown'。"""
    if isinstance(period, bool):  # bool 是 int 子类，先排除
        return "unknown"
    if isinstance(period, int):
        return "year"
    if isinstance(period, str):
        if _YEAR_RE.match(period):
            return "year"
        if _QUARTER_RE.match(period):
            return "quarter"
    return "unknown"


def parse_period(period):
    """把时期标签解析成可比较的整数索引。

    年份（int 或 'YYYY'）-> (year, 'YYYY', 'year')
    季度 'YYYYQn' -> (year*4+n-1, 'YYYYQn', 'quarter')
    """
    kind = period_kind(period)
    if kind == "year":
        if isinstance(period, int):
            return period, str(period), "year"
        return int(period), period, "year"
    if kind == "quarter":
        m = _QUARTER_RE.match(period)
        year, q = int(m.group(1)), int(m.group(2))
        return year * 4 + (q - 1), f"{year}Q{q}", "quarter"
    raise ValueError(f"时期 {period!r} 无法解析：应为年份整数、'YYYY' 或 'YYYYQn' 季度字符串")


# ---------------------------------------------------------------------------
# 输入行（已经过 pydantic 初步类型检查后的内部表示）
# ---------------------------------------------------------------------------


@dataclass
class InputRow:
    unit: str
    period: object
    outcome: float
    treated: bool
    covariates: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidRow:
    """校验后用于存储与估计的行（period 为整数排序索引）。"""

    unit: str
    period: int
    label: object
    outcome: float
    treated: bool
    covariates: dict[str, float] = field(default_factory=dict)


def _err(message: str, details: list[dict]) -> ValidationError:
    return ValidationError(message=message, details=details)


def validate_panel(
    input_rows: list[InputRow],
    *,
    existing: dict[tuple[str, int], tuple[object, float, bool, dict]] | None = None,
    expect_period_kind: str | None = None,
    deleted: set[tuple[str, object]] | None = None,
) -> tuple[list[ValidRow], dict[int, object], str, list]:
    """校验一张完整面板（差异场景下为"父版本 + 本次变更"合并后的完整内容）。

    existing:
        父版本内容 {(unit, period_index): (period_label, outcome, treated, covariates)}。

    返回 ``(合并后的完整行, period_index -> 原始时期标签, 时期类型, 本次 upsert 原始行)``。
    协变量列是否齐全不在此处校验，而在估计时按分析设定检查。
    """
    details: list[dict] = []
    deleted = deleted or set()

    # 1. 解析本次提交的时期，确定类型
    kind = expect_period_kind
    parsed_upserts: list[tuple[str, int, object, float, bool, dict[str, float]]] = []
    for i, r in enumerate(input_rows):
        row_ok = True
        try:
            idx, label, parsed_kind = parse_period(r.period)
        except ValueError:
            details.append({
                "field": f"rows[{i}].period",
                "value": r.period,
                "reason": "时期无法排序/解析：应为年份整数、'YYYY' 或 'YYYYQn' 季度字符串",
            })
            row_ok = False
            idx = label = parsed_kind = None
        if row_ok:
            if kind is None:
                kind = parsed_kind
            elif kind != parsed_kind:
                details.append({
                    "field": f"rows[{i}].period",
                    "value": r.period,
                    "reason": f"时期类型与数据集中既有类型 {kind!r} 不一致（年份与季度不能混用）",
                })
        if not isinstance(r.outcome, (int, float)) or isinstance(r.outcome, bool):
            details.append({
                "field": f"rows[{i}].outcome",
                "value": r.outcome,
                "reason": "结果变量必须是数值",
            })
            row_ok = False
        elif not (r.outcome == r.outcome) or r.outcome in (float("inf"), float("-inf")):
            details.append({
                "field": f"rows[{i}].outcome",
                "value": r.outcome,
                "reason": "结果变量必须是有限数值（不能为 NaN 或无穷）",
            })
            row_ok = False
        if not isinstance(r.treated, bool):
            details.append({
                "field": f"rows[{i}].treated",
                "value": r.treated,
                "reason": "处理状态必须为 true/false",
            })
            row_ok = False
        if not isinstance(r.unit, str) or not r.unit:
            details.append({
                "field": f"rows[{i}].unit",
                "value": r.unit,
                "reason": "单位编号必须是非空字符串",
            })
            row_ok = False
        for cname, cval in r.covariates.items():
            if not isinstance(cval, (int, float)) or isinstance(cval, bool):
                details.append({
                    "field": f"rows[{i}].covariates.{cname}",
                    "value": cval,
                    "reason": "协变量必须是数值",
                })
        if row_ok:
            parsed_upserts.append(
                (r.unit, idx, r.period, float(r.outcome), bool(r.treated), dict(r.covariates))
            )

    if not input_rows and existing is None:
        raise _err("面板数据为空：上传或差异合并后没有任何记录", [
            {"field": "rows", "reason": "数据集版本必须至少包含一行"}
        ])

    if details:
        raise _err("面板数据校验失败", details)

    if kind is None:
        # 本次无 upsert 且没有既有版本可供推断类型（不应走到这里，兜底）
        raise _err("面板数据为空：无法确定时期类型", [
            {"field": "rows", "reason": "数据集版本必须至少包含一行"}
        ])

    # 2. 本次 upsert 内不得重复
    seen: dict[tuple[str, int], int] = {}
    for j, (u, pidx, _label, _y, _t, _x) in enumerate(parsed_upserts):
        key = (u, pidx)
        if key in seen:
            details.append({
                "field": f"rows",
                "value": {"unit": u, "period": _label},
                "reason": f"同一单位 {u!r} 同一时期 {_label!r} 在本次提交中出现多行",
            })
        seen[key] = j
    if details:
        raise _err("面板数据校验失败", details)

    # 3. 删除键校验 + 与 upsert 互斥
    deleted_parsed: set[tuple[str, int]] = set()
    for (du, dp) in deleted:
        try:
            didx, _, dkind = parse_period(dp)
            if dkind != kind:
                details.append({
                    "field": "deletes",
                    "value": {"unit": du, "period": dp},
                    "reason": f"删除项时期类型 {dkind!r} 与数据集类型 {kind!r} 不一致",
                })
                continue
        except ValueError:
            details.append({
                "field": "deletes",
                "value": {"unit": du, "period": dp},
                "reason": "删除项的时期无法解析",
            })
            continue
        if existing is not None and (du, didx) not in existing:
            details.append({
                "field": "deletes",
                "value": {"unit": du, "period": dp},
                "reason": "要删除的记录在父版本中不存在",
            })
        if (du, didx) in seen:
            details.append({
                "field": "deletes",
                "value": {"unit": du, "period": dp},
                "reason": "同一记录不能既删除又 upsert",
            })
        deleted_parsed.add((du, didx))
    if details:
        raise _err("面板数据校验失败", details)

    # 4. 合并出完整面板（existing 各值既支持元组也支持 db 层的 dict 表示）
    merged: dict[tuple[str, int], tuple[object, float, bool, dict[str, float]]] = {}
    label_by_idx: dict[int, object] = {}

    def _as_tuple(v):
        if isinstance(v, dict):
            return v["label"], v["outcome"], v["treated"], dict(v.get("covariates") or {})
        return v

    if existing:
        for key, val in existing.items():
            label, y, t, x = _as_tuple(val)
            merged[key] = (label, y, t, dict(x))
            label_by_idx[key[1]] = label
    for (u, pidx, label, y, t, x) in parsed_upserts:
        merged[(u, pidx)] = (label, y, t, x)
        label_by_idx[pidx] = label
    for key in deleted_parsed:
        merged.pop(key, None)

    if not merged:
        raise _err("面板数据为空：上传或差异合并后没有任何记录", [
            {"field": "rows", "reason": "数据集版本必须至少包含一行"}
        ])

    # 5. 同一单位同一时期重复（upsert 内部重复已在前面报错；merged 为 dict 天然唯一）

    # 6. 处理状态单调性
    by_unit: dict[str, list[int]] = {}
    for (u, pidx) in merged:
        by_unit.setdefault(u, []).append(pidx)
    for u, pidxs in by_unit.items():
        pidxs.sort()
        started = False
        start_period = None
        for pidx in pidxs:
            t = merged[(u, pidx)][2]
            if t and not started:
                started = True
                start_period = label_by_idx[pidx]
            if started and not t:
                details.append({
                    "field": "treated",
                    "value": {"unit": u, "period": label_by_idx[pidx]},
                    "reason": (
                        f"单位 {u!r} 在时期 {start_period!r} 已开始处理，"
                        f"之后时期 {label_by_idx[pidx]!r} 又回到未处理（处理不允许撤回）"
                    ),
                })
    if details:
        raise _err("处理状态违反单调性（处理一旦开始不得撤回）", details)

    # 7. 组装 ValidRow（按单位、时期排序，保证后续估计对行序不敏感）
    out_rows: list[ValidRow] = []
    for u in sorted(by_unit):
        for pidx in sorted(by_unit[u]):
            label, y, t, x = merged[(u, pidx)]
            out_rows.append(ValidRow(unit=u, period=pidx, label=label, outcome=y,
                                     treated=t, covariates=dict(x)))

    return out_rows, label_by_idx, kind, parsed_upserts


def build_diff_rows(
    parent_rows: dict[tuple[str, int], tuple[object, float, bool, dict]],
    next_rows: list[InputRow],
) -> tuple[list[InputRow], list[dict]]:
    """纯 Python 差异计算：给定父版本完整内容和下一版完整内容，算出 upsert/delete。

    用于保证"按差异提交的版本"和"直接上传同样内容的完整数据"内容一致（测试用）。
    """
    next_map: dict[tuple[str, int], InputRow] = {}
    for r in next_rows:
        idx, _label, _kind = parse_period(r.period)
        next_map[(r.unit, idx)] = r

    upserts: list[InputRow] = []
    for key in sorted(next_map, key=lambda k: (k[0], k[1])):
        r = next_map[key]
        old = parent_rows.get(key)
        if old is None or old[1] != r.outcome or old[2] != r.treated or old[3] != dict(r.covariates):
            upserts.append(r)

    deletes = [
        {"unit": u, "period": old[0]}
        for (u, pidx), old in sorted(parent_rows.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        if (u, pidx) not in next_map
    ]
    return upserts, deletes


def stored_to_valid_rows(stored: dict) -> list[ValidRow]:
    """把 db.load_version_rows 的输出还原为按 (单位, 时期) 排序的 ValidRow。"""
    out = []
    for (u, pidx), rec in sorted(stored.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        out.append(
            ValidRow(
                unit=u,
                period=pidx,
                label=rec["label"],
                outcome=rec["outcome"],
                treated=rec["treated"],
                covariates=rec["covariates"],
            )
        )
    return out


def rows_for_estimation(valid_rows: list[ValidRow], covariate_names: list[str] | None):
    """把存储行转换成估计引擎需要的 :class:`~app.did.Row`。

    指定协变量列时，每行必须齐全且为数值，否则报 422 并指出字段。
    """
    from .did import Row

    names = sorted(covariate_names or [])
    out: list[Row] = []
    missing: list[dict] = []
    for r in valid_rows:
        vals = []
        for name in names:
            if name not in r.covariates:
                missing.append({
                    "field": f"covariates.{name}",
                    "value": {"unit": r.unit, "period": r.label},
                    "reason": f"该记录缺少分析设定指定的协变量 {name!r}",
                })
                continue
            v = r.covariates[name]
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                missing.append({
                    "field": f"covariates.{name}",
                    "value": v,
                    "reason": f"协变量 {name!r} 必须是数值",
                })
                continue
            vals.append(float(v))
        out.append(Row(unit=r.unit, period=r.period, outcome=r.outcome,
                       treated=r.treated, covariates=tuple(vals)))
    if missing:
        raise _err("分析设定引用的协变量在数据中缺失或非数值", missing[:20])
    return out, names

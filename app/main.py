"""FastAPI 入口：数据集版本管理 + DID 估计提交。"""
from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import storage
from .did import ENGINE_VERSION, PanelValidationError, _parse_period_label, build_panel, estimate
from .models import (
    CreateDatasetIn,
    CreateDiffVersionIn,
    CreateEstimateIn,
    CreateVersionIn,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.init_db()
    yield


app = FastAPI(
    title="县域政策面板评估服务",
    version="1.0.0",
    description="面板数据版本管理与双重差分（组别-时期 ATT / 动态效应 / 平行趋势）",
    lifespan=lifespan,
)


@app.exception_handler(PanelValidationError)
async def panel_validation_handler(request: Request, exc: PanelValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "field": exc.field,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    first = errors[0] if errors else {}
    loc = [str(x) for x in first.get("loc", []) if x not in ("body",)]
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "request_schema_invalid",
                "message": first.get("msg", "请求体不符合接口模式"),
                "field": ".".join(loc) if loc else None,
                "details": errors,
            }
        },
    )


@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError):
    code = exc.args[0] if exc.args else "not_found"
    status = 404
    messages = {
        "dataset_not_found": "数据集不存在",
        "version_not_found": "版本不存在",
        "estimate_not_found": "估计结果不存在",
        "no_base_version": "该数据集还没有任何版本，首个版本必须整表上传",
    }
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": messages.get(code, code)}},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    code = exc.args[0] if exc.args else "invalid_value"
    if code == "bad_version_ref":
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "bad_version_ref",
                    "message": "version 必须是 'latest' 或整数版本号",
                    "field": "version",
                }
            },
        )
    raise exc


@app.get("/health")
def health():
    return {"status": "ok"}


def _normalize_row_dicts(raw_rows: list[dict]) -> list[dict]:
    """落库前把 unit/period 规范化成字符串、period 统一大小写格式。"""
    out = []
    for r in raw_rows:
        rr = dict(r)
        rr["unit"] = str(r["unit"])
        rr["period"] = _parse_period_label(str(r["period"]).strip())[2]
        out.append(rr)
    return out


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
@app.post("/datasets", status_code=201)
def create_dataset(body: CreateDatasetIn):
    return storage.create_dataset(body.name)


@app.get("/datasets/{dataset_id}/versions")
def list_versions(dataset_id: str):
    return storage.list_versions(dataset_id)


@app.post("/datasets/{dataset_id}/versions", status_code=201)
def create_full_version(dataset_id: str, body: CreateVersionIn):
    raw_rows = [r.model_dump() for r in body.rows]
    # 先校验再落库：不合规的上传不会产生任何版本。
    build_panel(raw_rows)
    return storage.create_version_full(dataset_id, _normalize_row_dicts(raw_rows))


@app.post("/datasets/{dataset_id}/versions/diff", status_code=201)
def create_diff_version(dataset_id: str, body: CreateDiffVersionIn):
    changes_raw = [
        {
            "op": c.op,
            "unit": c.unit,
            "period": c.period,
            "row": c.row.model_dump() if c.row is not None else None,
        }
        for c in body.changes
    ]

    # 在 Python 侧先把差异应用到父版本全量内容上做一次完整校验，
    # 落库阶段再用 SQL 幂等地物化同样的结果。
    parent_summary, parent_rows = storage.get_version_rows(dataset_id, "latest")
    merged = {(r["unit"], r["period"]): r for r in parent_rows}
    for ch in changes_raw:
        if ch["op"] == "upsert":
            row = _normalize_row_dicts([ch["row"]])[0]
            merged[(row["unit"], row["period"])] = row
        else:
            key = (str(ch["unit"]), _parse_period_label(str(ch["period"]).strip())[2])
            merged.pop(key, None)
    build_panel(list(merged.values()))

    # 落库的 upsert 行同样用规范化后的键。
    db_changes = []
    for ch in changes_raw:
        if ch["op"] == "upsert":
            db_changes.append({"op": "upsert", "row": _normalize_row_dicts([ch["row"]])[0]})
        else:
            db_changes.append(ch)
    return storage.create_version_diff(dataset_id, db_changes)


@app.get("/datasets/{dataset_id}/versions/{version_ref}")
def get_version(dataset_id: str, version_ref: str):
    summary, rows = storage.get_version_rows(dataset_id, version_ref)
    return {"version": summary, "rows": rows}


# --------------------------------------------------------------------------- #
# 估计
# --------------------------------------------------------------------------- #
def _canonical_spec(spec: dict) -> str:
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


@app.post("/datasets/{dataset_id}/estimates", status_code=201)
def submit_estimate(dataset_id: str, body: CreateEstimateIn):
    spec_full = body.spec.model_dump(exclude_none=False)
    # include_twfe 只影响响应里是否返回诊断字段，不改变主估计口径，
    # 因此不参与幂等哈希：同一份主结果只计算/存储一次。
    spec_for_key = {k: v for k, v in spec_full.items() if k != "include_twfe"}
    spec_hash = _canonical_spec({"engine": ENGINE_VERSION, "spec": spec_for_key})

    version_uuid, version_no = storage.resolve_version(dataset_id, body.version)
    _, rows = storage.get_version_rows(dataset_id, str(version_no))

    # 取版本全量数据 -> 校验/组装 -> 估计。
    panel = build_panel(rows)
    result = estimate(
        panel,
        control_group=spec_full["control_group"],
        adjust_covariates=spec_full["adjust_covariates"],
        include_twfe=True,  # 存一份完整结果；响应阶段按请求裁剪
    )

    record, created = storage.get_or_create_estimate(
        dataset_id=dataset_id,
        version_uuid=version_uuid,
        spec=spec_full,
        spec_hash=spec_hash,
        engine_version=ENGINE_VERSION,
        result=result,
    )
    if not spec_full["include_twfe"]:
        record = dict(record)
        stored_result = dict(record["result"])
        stored_result["twfe"] = None
        record["result"] = stored_result
    return JSONResponse(status_code=201 if created else 200, content=record)


@app.get("/datasets/{dataset_id}/estimates")
def list_estimates(dataset_id: str):
    return storage.list_estimates(dataset_id)


@app.get("/datasets/{dataset_id}/estimates/{estimate_id}")
def get_estimate(dataset_id: str, estimate_id: str):
    return storage.get_estimate(dataset_id, estimate_id)

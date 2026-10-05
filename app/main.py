"""FastAPI 应用：面板数据版本、估计与结果的后端服务。"""

from __future__ import annotations

import json
import math
from contextlib import asynccontextmanager
from typing import Any

import numpy as np

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import db
from .did import estimate_did, estimate_twfe
from .errors import AppError, EstimationError, NotFoundError, ValidationError
from .panel import InputRow, rows_for_estimation, stored_to_valid_rows, validate_panel
from .schemas import AnalysisRequest, CreateDatasetRequest, DiffRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(
    title="县域政策 DID 评估服务",
    version="1.0.0",
    description=(
        "管理面板数据版本、双重差分估计与结果。主估计量为避开"
        "“已处理单位当对照”的堆叠 2×2 DID（Callaway–Sant'Anna 风格）。"
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 异常处理：统一 {error: {code, message, details}}
# ---------------------------------------------------------------------------


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    body: dict[str, Any] = {"code": exc.code, "message": exc.message}
    details = getattr(exc, "details", None)
    if details:
        body["details"] = details
    return JSONResponse(status_code=exc.status_code, content={"error": body})


def _to_input_rows(rows) -> list[InputRow]:
    return [
        InputRow(
            unit=r.unit,
            period=r.period,
            outcome=r.outcome,
            treated=r.treated,
            covariates=dict(r.covariates or {}),
        )
        for r in rows
    ]


def _json_default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def _sanitize(o: Any) -> Any:
    """把 NaN/±inf 换成 None（JSON 不支持，且数值上不可识别时不应给出伪值）。"""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return _sanitize(o.item())
    return o


def _dump(payload: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        json.loads(json.dumps(_sanitize(payload), default=_json_default, allow_nan=False)),
        status_code=status_code,
    )


def _version_payload(meta: dict, valid_rows=None, include_data: bool = False) -> dict:
    out = {
        "version_id": meta["id"],
        "dataset_id": meta["dataset_id"],
        "version": meta["version"],
        "parent_version": meta.get("parent_version"),
        "change_note": meta.get("change_note", ""),
        "n_rows": meta["n_rows"],
        "n_units": meta["n_units"],
    }
    if include_data and valid_rows is not None:
        out["rows"] = [
            {
                "unit": r.unit,
                "period": r.label,
                "outcome": r.outcome,
                "treated": r.treated,
                "covariates": r.covariates,
            }
            for r in valid_rows
        ]
    return out


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 数据集与版本
# ---------------------------------------------------------------------------


@app.post("/api/datasets", status_code=201)
def create_dataset(req: CreateDatasetRequest):
    inputs = _to_input_rows(req.rows)
    valid_rows, label_by_idx, kind, _upserts = validate_panel(inputs)

    with db.get_conn() as conn:
        dataset_id = db.create_dataset(
            conn, name=req.name, description=req.description, period_kind=kind
        )
        version_id = db.insert_version_with_rows(
            conn,
            dataset_id=dataset_id,
            version_no=1,
            parent_version=None,
            change_note="初始完整上传",
            change_type="create",
            upserts=[r.model_dump() for r in req.rows],
            deletes=[],
            valid_rows=valid_rows,
            label_by_idx=label_by_idx,
        )
        vmeta = db.load_version_meta(conn, version_id)
        return _dump(_version_payload(vmeta, valid_rows, include_data=False), 201)


@app.get("/api/datasets")
def list_datasets():
    with db.get_conn() as conn:
        return _dump({"datasets": db.list_datasets(conn)})


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: int):
    with db.get_conn() as conn:
        ds = db.get_dataset(conn, dataset_id)
        versions = db.list_versions(conn, dataset_id)
        return _dump({"dataset": ds, "versions": versions})


@app.post("/api/datasets/{dataset_id}/versions", status_code=201)
def create_diff_version(dataset_id: int, req: DiffRequest):
    bad_deletes = [
        {"field": "deletes", "value": d,
         "reason": "删除项必须同时包含非空 'unit' 与 'period' 两个字段"}
        for d in req.deletes
        if not isinstance(d, dict) or "unit" not in d or "period" not in d
        or d.get("unit") in (None, "") or d.get("period") is None
    ]
    if bad_deletes:
        raise ValidationError("差异提交的 deletes 字段不合法", bad_deletes)
    with db.get_conn() as conn:
        ds = db.get_dataset(conn, dataset_id)
        parent = db.find_dataset_version(conn, dataset_id, None)
        parent_rows = db.load_version_rows(conn, parent["id"])

        # 删除项的时期要先按数据集口径解析
        deleted = {(d["unit"], d["period"]) for d in req.deletes}
        inputs = _to_input_rows(req.upserts)
        valid_rows, label_by_idx, _kind, _ups = validate_panel(
            inputs,
            existing=parent_rows,
            expect_period_kind=ds["period_kind"],
            deleted=deleted,
        )

        version_no = parent["version"] + 1
        version_id = db.insert_version_with_rows(
            conn,
            dataset_id=dataset_id,
            version_no=version_no,
            parent_version=parent["version"],
            change_note=req.change_note,
            change_type="diff",
            upserts=[r.model_dump() for r in req.upserts],
            deletes=req.deletes,
            valid_rows=valid_rows,
            label_by_idx=label_by_idx,
        )
        vmeta = db.load_version_meta(conn, version_id)
        return _dump(_version_payload(vmeta, valid_rows), 201)


@app.get("/api/datasets/{dataset_id}/versions/{version}")
def get_version(dataset_id: int, version: int, data: bool = True):
    with db.get_conn() as conn:
        db.get_dataset(conn, dataset_id)
        v = db.find_dataset_version(conn, dataset_id, version)
        meta = db.load_version_meta(conn, v["id"])
        valid_rows = None
        if data:
            stored = db.load_version_rows(conn, v["id"])
            valid_rows = stored_to_valid_rows(stored)
        return _dump(_version_payload(meta, valid_rows, include_data=data))


# ---------------------------------------------------------------------------
# 分析与结果
# ---------------------------------------------------------------------------


def _run_estimation(valid_rows, analysis: AnalysisRequest) -> dict:
    est_rows, cov_names = rows_for_estimation(valid_rows, analysis.covariates)
    result = estimate_did(est_rows, control=analysis.control_strategy)
    result["covariates_used"] = cov_names
    if analysis.also_twfe:
        try:
            result["twfe"] = estimate_twfe(est_rows)
        except EstimationError as exc:
            result["twfe"] = {"error": exc.message}
    return result


@app.post("/api/versions/{version_id}/analyses", status_code=201)
def submit_analysis(version_id: int, req: AnalysisRequest):
    """在指定数据版本上提交分析；同设定重复提交直接返回已有结果（幂等）。"""
    with db.get_conn() as conn:
        meta = db.load_version_meta(conn, version_id)
        analysis_id, _created = db.get_or_create_analysis(
            conn,
            name=req.name,
            control_strategy=req.control_strategy,
            covariates=req.covariates,
            also_twfe=req.also_twfe,
        )
        existing = db.find_result(conn, version_id, analysis_id)
        if existing is not None:
            payload = existing["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            return _dump({
                "cached": True,
                "result_id": existing["id"],
                "version_id": version_id,
                "analysis_id": analysis_id,
                "result": payload,
            }, 200)  # 幂等命中：200 OK，区别于新建 201

        stored = db.load_version_rows(conn, version_id)
        valid_rows = stored_to_valid_rows(stored)

        result = _run_estimation(valid_rows, req)
        result_id = db.insert_result(
            conn,
            version_id=version_id,
            analysis_id=analysis_id,
            att=float(result["att"]),
            std_error=float(result["std_error"]),
            payload=result,
        )
        return _dump({
            "cached": False,
            "result_id": result_id,
            "version_id": version_id,
            "analysis_id": analysis_id,
            "result": result,
        }, 201)


@app.get("/api/results/{result_id}")
def get_result(result_id: int):
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.*, a.name AS analysis_name
                FROM results r JOIN analyses a ON a.id = r.analysis_id
                WHERE r.id=%s
                """,
                (result_id,),
            )
            row = cur.fetchone()
        if row is None:
            raise NotFoundError(f"结果 id={result_id} 不存在")
        payload = row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return _dump({
            "result_id": row["id"],
            "version_id": row["version_id"],
            "analysis_id": row["analysis_id"],
            "analysis_name": row["analysis_name"],
            "att": row["att"],
            "std_error": row["std_error"],
            "created_at": row["created_at"],
            "result": payload,
        })


@app.get("/api/results")
def list_all_results(version_id: int | None = None):
    with db.get_conn() as conn:
        return _dump({"results": db.list_results(conn, version_id)})


@app.get("/api/analyses")
def list_analyses():
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM analyses ORDER BY id")
            return _dump({"analyses": cur.fetchall()})

"""PostgreSQL 存储层（psycopg3）：数据集/版本/差异/估计结果的持久化。"""
from __future__ import annotations

import json
import os

import psycopg
import psycopg.errors
from psycopg.types.json import Json

from .config import DATABASE_URL

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


class ConnectionPool:
    def __init__(self, dsn: str):
        from psycopg_pool import ConnectionPool as _CP

        self._pool = _CP(dsn, min_size=1, max_size=10, open=True)

    def conn(self):
        return self._pool.connection()

    def close(self):
        self._pool.close()


_pool: ConnectionPool | None = None


def init_pool(dsn: str | None = None) -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(dsn or DATABASE_URL)
    return _pool


def get_pool() -> ConnectionPool:
    assert _pool is not None, "连接池尚未初始化"
    return _pool


def init_db() -> None:
    pool = init_pool()
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        ddl = f.read()
    with pool.conn() as conn:
        conn.execute(ddl)
        conn.commit()


# --------------------------------------------------------------------------- #
# 数据集与版本
# --------------------------------------------------------------------------- #
def create_dataset(name: str) -> dict:
    with get_pool().conn() as conn:
        row = conn.execute(
            "INSERT INTO datasets (name) VALUES (%s) RETURNING id, name, created_at",
            (name,),
        ).fetchone()
        conn.commit()
        return {"id": str(row[0]), "name": row[1], "created_at": row[2].isoformat()}


def list_versions(dataset_id: str) -> list[dict]:
    with get_pool().conn() as conn:
        ds = conn.execute("SELECT 1 FROM datasets WHERE id = %s", (dataset_id,)).fetchone()
        if ds is None:
            raise KeyError("dataset_not_found")
        rows = conn.execute(
            """
            SELECT v.id, v.version_no, v.parent_id, v.created_via, v.n_rows, v.created_at
            FROM dataset_versions v
            WHERE v.dataset_id = %s
            ORDER BY v.version_no
            """,
            (dataset_id,),
        ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": str(r[0]),
                    "version_no": int(r[1]),
                    "parent_id": str(r[2]) if r[2] else None,
                    "created_via": r[3],
                    "n_rows": r[4],
                    "created_at": r[5].isoformat(),
                }
            )
        return out


def _resolve_version(conn, dataset_id: str, ref: str) -> tuple[str, int]:
    """把 'latest' 或版本号解析为 (version_uuid, version_no)。"""
    if ref == "latest":
        row = conn.execute(
            """
            SELECT id, version_no FROM dataset_versions
            WHERE dataset_id = %s ORDER BY version_no DESC LIMIT 1
            """,
            (dataset_id,),
        ).fetchone()
    else:
        try:
            no = int(ref)
        except ValueError:
            raise ValueError("bad_version_ref")
        row = conn.execute(
            "SELECT id, version_no FROM dataset_versions WHERE dataset_id = %s AND version_no = %s",
            (dataset_id, no),
        ).fetchone()
    if row is None:
        raise KeyError("version_not_found")
    return str(row[0]), int(row[1])


def resolve_version(dataset_id: str, ref: str) -> tuple[str, int]:
    with get_pool().conn() as conn:
        if conn.execute("SELECT 1 FROM datasets WHERE id = %s", (dataset_id,)).fetchone() is None:
            raise KeyError("dataset_not_found")
        return _resolve_version(conn, dataset_id, ref)


def create_version_full(dataset_id: str, rows: list[dict]) -> dict:
    """整表上传：校验通过后整体写入新版本（不可变）。"""
    pool = get_pool()
    with pool.conn() as conn:
        if conn.execute("SELECT 1 FROM datasets WHERE id = %s", (dataset_id,)).fetchone() is None:
            raise KeyError("dataset_not_found")
        parent = conn.execute(
            """
            SELECT id FROM dataset_versions
            WHERE dataset_id = %s ORDER BY version_no DESC LIMIT 1
            """,
            (dataset_id,),
        ).fetchone()
        new_no = (
            conn.execute(
                "SELECT COALESCE(MAX(version_no), 0) + 1 FROM dataset_versions WHERE dataset_id = %s",
                (dataset_id,),
            ).fetchone()[0]
        )
        version_id = conn.execute(
            """
            INSERT INTO dataset_versions
                (dataset_id, version_no, parent_id, created_via, n_rows)
            VALUES (%s, %s, %s, 'full', %s) RETURNING id
            """,
            (dataset_id, new_no, parent[0] if parent else None, len(rows)),
        ).fetchone()[0]
        _bulk_insert_rows(conn, version_id, rows)
        conn.commit()
        return _version_summary(conn, version_id)


def create_version_diff(dataset_id: str, changes: list[dict]) -> dict:
    """以差异提交生成新版本：从父版本完整内容复制后应用 upsert/delete。"""
    pool = get_pool()
    with pool.conn() as conn:
        if conn.execute("SELECT 1 FROM datasets WHERE id = %s", (dataset_id,)).fetchone() is None:
            raise KeyError("dataset_not_found")
        parent_row = conn.execute(
            """
            SELECT id, version_no FROM dataset_versions
            WHERE dataset_id = %s ORDER BY version_no DESC LIMIT 1
            """,
            (dataset_id,),
        ).fetchone()
        if parent_row is None:
            raise KeyError("no_base_version")
        parent_id, parent_no = parent_row[0], parent_row[1]

        normalized_changes = _normalize_changes(changes)

        new_no = parent_no + 1
        version_id = conn.execute(
            """
            INSERT INTO dataset_versions
                (dataset_id, version_no, parent_id, created_via, n_rows)
            VALUES (%s, %s, %s, 'diff', 0) RETURNING id
            """,
            (dataset_id, new_no, parent_id),
        ).fetchone()[0]

        # 物化父版本的完整内容（SQL 层复制，避免往返）。
        conn.execute(
            """
            INSERT INTO panel_rows (version_id, unit_id, period, outcome, treated, covariates)
            SELECT %s, unit_id, period, outcome, treated, covariates FROM panel_rows
            WHERE version_id = %s
            """,
            (version_id, parent_id),
        )

        # 应用差异：先 delete 再 upsert。
        deletes = [(c["unit"], c["period"]) for c in normalized_changes if c["op"] == "delete"]
        upserts = [c for c in normalized_changes if c["op"] == "upsert"]

        for (u, p) in deletes:
            conn.execute(
                "DELETE FROM panel_rows WHERE version_id = %s AND unit_id = %s AND period = %s",
                (version_id, u, p),
            )
        if upserts:
            _bulk_insert_rows(
                conn,
                version_id,
                [c["row"] for c in upserts],
                upsert=True,
            )

        # 记录差异清单（审计/血缘）。
        for ch in normalized_changes:
            conn.execute(
                """
                INSERT INTO version_changes (version_id, op, unit_id, period, payload)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    version_id,
                    ch["op"],
                    ch["unit"],
                    ch["period"],
                    Json(ch.get("row")) if ch.get("row") is not None else None,
                ),
            )

        n = conn.execute(
            "SELECT count(*) FROM panel_rows WHERE version_id = %s", (version_id,)
        ).fetchone()[0]
        conn.execute(
            "UPDATE dataset_versions SET n_rows = %s WHERE id = %s", (n, version_id)
        )
        conn.commit()
        return _version_summary(conn, version_id)


def _normalize_changes(changes: list[dict]) -> list[dict]:
    from .did import PanelValidationError, _parse_period_label

    if not isinstance(changes, list) or not changes:
        raise PanelValidationError("empty_diff", "差异为空：至少需要一条 upsert/delete", "changes")
    out = []
    for i, ch in enumerate(changes):
        if not isinstance(ch, dict) or ch.get("op") not in ("upsert", "delete"):
            raise PanelValidationError(
                "change_op_invalid",
                f"第 {i} 条变更必须带 op='upsert' 或 op='delete'",
                field="changes.op",
            )
        op = ch["op"]
        if op == "upsert":
            row = ch.get("row")
            if not isinstance(row, dict):
                raise PanelValidationError(
                    "change_row_missing",
                    f"第 {i} 条 upsert 必须包含完整 row",
                    field="changes.row",
                )
            unit, period = str(row["unit"]), _parse_period_label(str(row["period"]).strip())[2]
        else:
            for k in ("unit", "period"):
                if k not in ch:
                    raise PanelValidationError(
                        "change_key_missing",
                        f"第 {i} 条 delete 缺少 {k!r}",
                        field=f"changes.{k}",
                    )
            unit = str(ch["unit"])
            period = _parse_period_label(str(ch["period"]).strip())[2]
        out.append(
            {
                "op": op,
                "unit": unit,
                "period": period,
                "row": ch["row"] if op == "upsert" else None,
            }
        )
    return out


def _bulk_insert_rows(conn, version_id, rows: list[dict], upsert: bool = False) -> None:
    records = []
    for r in rows:
        treated = r["treated"]
        if not isinstance(treated, bool):
            treated = bool(int(treated))
        records.append(
            (
                version_id,
                str(r["unit"]),
                str(r["period"]).strip(),
                float(r["outcome"]),
                treated,
                Json(r.get("covariates") or {}),
            )
        )
    if upsert:
        conn.executemany(
            """
            INSERT INTO panel_rows (version_id, unit_id, period, outcome, treated, covariates)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (version_id, unit_id, period)
            DO UPDATE SET outcome = EXCLUDED.outcome,
                          treated = EXCLUDED.treated,
                          covariates = EXCLUDED.covariates
            """,
            records,
        )
    else:
        # 批量 COPY，psycopg3 会按目标列类型（JSONB）自动适配 Json 包装器。
        with conn.cursor().copy(
            "COPY panel_rows (version_id, unit_id, period, outcome, treated, covariates) "
            "FROM STDIN"
        ) as cp:
            for rec in records:
                cp.write_row(rec)


def _version_summary(conn, version_id) -> dict:
    row = conn.execute(
        """
        SELECT v.id, v.dataset_id, v.version_no, v.parent_id, v.created_via,
               v.n_rows, v.created_at
        FROM dataset_versions v WHERE v.id = %s
        """,
        (version_id,),
    ).fetchone()
    return {
        "id": str(row[0]),
        "dataset_id": str(row[1]),
        "version_no": int(row[2]),
        "parent_id": str(row[3]) if row[3] else None,
        "created_via": row[4],
        "n_rows": row[5],
        "created_at": row[6].isoformat(),
    }


def get_version_rows(dataset_id: str, ref: str) -> tuple[dict, list[dict]]:
    """取回某版本的完整内容（物化后的全量行）。"""
    with get_pool().conn() as conn:
        if conn.execute("SELECT 1 FROM datasets WHERE id = %s", (dataset_id,)).fetchone() is None:
            raise KeyError("dataset_not_found")
        version_uuid, _ = _resolve_version(conn, dataset_id, ref)
        rows = conn.execute(
            """
            SELECT unit_id, period, outcome, treated, covariates
            FROM panel_rows WHERE version_id = %s
            ORDER BY unit_id, period
            """,
            (version_uuid,),
        ).fetchall()
        summary = _version_summary(conn, version_uuid)
        out = [
            {
                "unit": r[0],
                "period": r[1],
                "outcome": float(r[2]),
                "treated": bool(r[3]),
                "covariates": r[4] if isinstance(r[4], dict) else json.loads(r[4]),
            }
            for r in rows
        ]
        return summary, out


def get_version_id(dataset_id: str, ref: str) -> str:
    vid, _ = resolve_version(dataset_id, ref)
    return vid


# --------------------------------------------------------------------------- #
# 估计结果
# --------------------------------------------------------------------------- #
def get_or_create_estimate(
    dataset_id: str,
    version_uuid: str,
    spec: dict,
    spec_hash: str,
    engine_version: str,
    result: dict,
) -> tuple[dict, bool]:
    """返回 (估计记录, 是否新建)。同版本同设定命中唯一约束时返回已有结果。"""
    with get_pool().conn() as conn:
        if conn.execute("SELECT 1 FROM datasets WHERE id = %s", (dataset_id,)).fetchone() is None:
            raise KeyError("dataset_not_found")
        if conn.execute(
            "SELECT 1 FROM dataset_versions WHERE id = %s", (version_uuid,)
        ).fetchone() is None:
            raise KeyError("version_not_found")

        existing = conn.execute(
            "SELECT id, result, created_at FROM estimates WHERE version_id = %s AND spec_hash = %s",
            (version_uuid, spec_hash),
        ).fetchone()
        if existing is not None:
            return (
                {
                    "id": str(existing[0]),
                    "dataset_id": dataset_id,
                    "version_id": str(version_uuid),
                    "spec": spec,
                    "spec_hash": spec_hash,
                    "engine_version": engine_version,
                    "result": existing[1],
                    "created_at": existing[2].isoformat(),
                    "cached": True,
                },
                False,
            )

        try:
            row = conn.execute(
                """
                INSERT INTO estimates
                    (dataset_id, version_id, spec_hash, spec, engine_version, result)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, created_at
                """,
                (
                    dataset_id,
                    version_uuid,
                    spec_hash,
                    Json(spec),
                    engine_version,
                    Json(result),
                ),
            ).fetchone()
            conn.commit()
        except psycopg.errors.UniqueViolation:
            # 并发提交了同一设定：回滚后返回已有结果。
            conn.rollback()
            existing = conn.execute(
                "SELECT id, result, created_at FROM estimates "
                "WHERE version_id = %s AND spec_hash = %s",
                (version_uuid, spec_hash),
            ).fetchone()
            return (
                {
                    "id": str(existing[0]),
                    "dataset_id": dataset_id,
                    "version_id": str(version_uuid),
                    "spec": spec,
                    "spec_hash": spec_hash,
                    "engine_version": engine_version,
                    "result": existing[1],
                    "created_at": existing[2].isoformat(),
                    "cached": True,
                },
                False,
            )
        return (
            {
                "id": str(row[0]),
                "dataset_id": dataset_id,
                "version_id": str(version_uuid),
                "spec": spec,
                "spec_hash": spec_hash,
                "engine_version": engine_version,
                "result": result,
                "created_at": row[1].isoformat(),
                "cached": False,
            },
            True,
        )


def list_estimates(dataset_id: str) -> list[dict]:
    with get_pool().conn() as conn:
        if conn.execute("SELECT 1 FROM datasets WHERE id = %s", (dataset_id,)).fetchone() is None:
            raise KeyError("dataset_not_found")
        rows = conn.execute(
            """
            SELECT e.id, e.version_id, v.version_no, e.spec_hash, e.engine_version, e.created_at
            FROM estimates e JOIN dataset_versions v ON v.id = e.version_id
            WHERE e.dataset_id = %s ORDER BY e.created_at
            """,
            (dataset_id,),
        ).fetchall()
        return [
            {
                "id": str(r[0]),
                "version_id": str(r[1]),
                "version_no": int(r[2]),
                "spec_hash": r[3],
                "engine_version": r[4],
                "created_at": r[5].isoformat(),
            }
            for r in rows
        ]


def get_estimate(dataset_id: str, estimate_id: str) -> dict:
    with get_pool().conn() as conn:
        row = conn.execute(
            """
            SELECT id, version_id, spec_hash, spec, engine_version, result, created_at
            FROM estimates WHERE dataset_id = %s AND id = %s
            """,
            (dataset_id, estimate_id),
        ).fetchone()
        if row is None:
            raise KeyError("estimate_not_found")
        return {
            "id": str(row[0]),
            "dataset_id": dataset_id,
            "version_id": str(row[1]),
            "spec_hash": row[2],
            "spec": row[3],
            "engine_version": row[4],
            "result": row[5],
            "created_at": row[6].isoformat(),
        }

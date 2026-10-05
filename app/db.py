"""PostgreSQL 访问层（psycopg 3，同步）。

存储模型
========
* ``datasets``：数据集元信息；
* ``dataset_versions``：版本号在数据集内自增；v1 存完整内容，后续版本同样存完整内容
  （由"父版本完整内容 + 本次差异 upsert/delete"物化而成），差异本身也落库，
  这样按版本号取回完整内容是一次简单查询，旧版本永不被覆盖；
* ``observations``：(version_id, unit, period_idx) 唯一，结果变量/处理状态/协变量；
* ``analyses``：分析设定（对照口径、协变量列等）；
* ``results``：(version_id, analysis_id) 唯一，保证"同版本同设定重复提交直接返回已有结果"。
"""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id              BIGSERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    period_kind     TEXT NOT NULL CHECK (period_kind IN ('year', 'quarter')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- 说明：时期排序索引是绝对时期（年份整数；季度为 year*4+(季度-1)），
-- 由 parse_period 统一计算，天然跨版本稳定，无需另存时期轴。

CREATE TABLE IF NOT EXISTS dataset_versions (
    id              BIGSERIAL PRIMARY KEY,
    dataset_id      BIGINT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    version         INTEGER NOT NULL,
    parent_version  INTEGER,
    change_note     TEXT NOT NULL DEFAULT '',
    n_rows          INTEGER NOT NULL,
    n_units         INTEGER NOT NULL,
    period_labels   JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (dataset_id, version)
);

CREATE TABLE IF NOT EXISTS version_changes (
    id                  BIGSERIAL PRIMARY KEY,
    version_id          BIGINT NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
    change_type         TEXT NOT NULL CHECK (change_type IN ('create', 'diff')),
    upserts             JSONB NOT NULL,
    deletes             JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    id              BIGSERIAL PRIMARY KEY,
    version_id      BIGINT NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
    unit_id         TEXT NOT NULL,
    period_idx      INTEGER NOT NULL,
    period_label    JSONB NOT NULL,
    outcome         DOUBLE PRECISION NOT NULL,
    treated         BOOLEAN NOT NULL,
    covariates      JSONB NOT NULL,
    UNIQUE (version_id, unit_id, period_idx)
);
CREATE INDEX IF NOT EXISTS idx_observations_version ON observations(version_id);

CREATE TABLE IF NOT EXISTS analyses (
    id                  BIGSERIAL PRIMARY KEY,
    name                TEXT NOT NULL,
    control_strategy    TEXT NOT NULL CHECK (control_strategy IN
                            ('never_treated', 'not_yet_treated')),
    covariates          JSONB NOT NULL,
    also_twfe           BOOLEAN NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (name, control_strategy, covariates, also_twfe)
);

CREATE TABLE IF NOT EXISTS results (
    id              BIGSERIAL PRIMARY KEY,
    version_id      BIGINT NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
    analysis_id     BIGINT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    att             DOUBLE PRECISION NOT NULL,
    std_error       DOUBLE PRECISION NOT NULL,
    payload         JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (version_id, analysis_id)
);
"""


def dsn() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://did:did@localhost:5432/did_panel",
    )


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(dsn(), autocommit=False, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(retries: int = 30, delay: float = 1.0) -> None:
    """建表；数据库容器刚启动时轮询等待。"""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(SCHEMA)
            return
        except psycopg.OperationalError as exc:
            last_exc = exc
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# 数据读写
# ---------------------------------------------------------------------------


def load_version_rows(conn: psycopg.Connection, version_id: int) -> dict[tuple[str, int], dict]:
    """取某版本完整内容，返回 {(unit, period_idx): {label, outcome, treated, covariates}}。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT unit_id, period_idx, period_label, outcome, treated, covariates
            FROM observations WHERE version_id = %s
            """,
            (version_id,),
        )
        out = {}
        for r in cur.fetchall():
            label = r["period_label"]
            if isinstance(label, str):
                try:
                    label = json.loads(label)
                except json.JSONDecodeError:
                    pass
            out[(r["unit_id"], r["period_idx"])] = {
                "label": label,
                "outcome": r["outcome"],
                "treated": r["treated"],
                "covariates": r["covariates"] or {},
            }
        return out


def load_version_meta(conn: psycopg.Connection, version_id: int) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT v.id, v.dataset_id, v.version, v.parent_version, v.change_note,
                   v.n_rows, v.n_units, v.period_labels, d.period_kind,
                   d.name AS dataset_name
            FROM dataset_versions v JOIN datasets d ON d.id = v.dataset_id
            WHERE v.id = %s
            """,
            (version_id,),
        )
        row = cur.fetchone()
        if row is None:
            from .errors import NotFoundError

            raise NotFoundError(f"版本 id={version_id} 不存在")
        return row


def find_dataset_version(conn: psycopg.Connection, dataset_id: int, version: int | None = None) -> dict:
    with conn.cursor() as cur:
        if version is None:
            cur.execute(
                "SELECT * FROM dataset_versions WHERE dataset_id=%s ORDER BY version DESC LIMIT 1",
                (dataset_id,),
            )
        else:
            cur.execute(
                "SELECT * FROM dataset_versions WHERE dataset_id=%s AND version=%s",
                (dataset_id, version),
            )
        row = cur.fetchone()
        if row is None:
            from .errors import NotFoundError

            what = "最新版本" if version is None else f"版本 {version}"
            raise NotFoundError(f"数据集 id={dataset_id} 的{what}不存在")
        return row


def insert_version_with_rows(
    conn: psycopg.Connection,
    *,
    dataset_id: int,
    version_no: int,
    parent_version: int | None,
    change_note: str,
    change_type: str,
    upserts: list[dict],
    deletes: list[dict],
    valid_rows: list,
    label_by_idx: dict[int, object],
) -> int:
    """在一个事务里写入版本、差异物与完整内容。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO dataset_versions
                (dataset_id, version, parent_version, change_note, n_rows, n_units, period_labels)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                dataset_id,
                version_no,
                parent_version,
                change_note,
                len(valid_rows),
                len({r.unit for r in valid_rows}),
                json.dumps({str(k): _jsonable(v) for k, v in label_by_idx.items()}),
            ),
        )
        version_id = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO version_changes (version_id, change_type, upserts, deletes) VALUES (%s,%s,%s,%s)",
            (version_id, change_type, json.dumps(upserts), json.dumps(deletes)),
        )
        with cur.copy(
            "COPY observations (version_id, unit_id, period_idx, period_label, outcome, treated, covariates) "
            "FROM STDIN"
        ) as copy:
            for r in valid_rows:
                copy.write_row((
                    version_id,
                    r.unit,
                    r.period,
                    json.dumps(_jsonable(r.label)),
                    r.outcome,
                    r.treated,
                    json.dumps(r.covariates),
                ))
        return version_id


def _jsonable(v: Any):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _clean_nan(o: Any) -> Any:
    """递归把 NaN/±inf 换成 None，保证 JSONB 与标准 JSON 可序列化。"""
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _clean_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_nan(v) for v in o]
    return o


def get_or_create_analysis(
    conn: psycopg.Connection,
    *,
    name: str,
    control_strategy: str,
    covariates: list[str],
    also_twfe: bool,
) -> tuple[int, bool]:
    """返回 (analysis_id, created)。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id FROM analyses
            WHERE name=%s AND control_strategy=%s AND covariates=%s AND also_twfe=%s
            """,
            (name, control_strategy, json.dumps(sorted(covariates)), also_twfe),
        )
        row = cur.fetchone()
        if row:
            return row["id"], False
        cur.execute(
            """
            INSERT INTO analyses (name, control_strategy, covariates, also_twfe)
            VALUES (%s, %s, %s, %s) RETURNING id
            """,
            (name, control_strategy, json.dumps(sorted(covariates)), also_twfe),
        )
        return cur.fetchone()["id"], True


def find_result(conn: psycopg.Connection, version_id: int, analysis_id: int) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM results WHERE version_id=%s AND analysis_id=%s",
            (version_id, analysis_id),
        )
        return cur.fetchone()


def insert_result(
    conn: psycopg.Connection,
    *,
    version_id: int,
    analysis_id: int,
    att: float,
    std_error: float,
    payload: dict,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO results (version_id, analysis_id, att, std_error, payload)
            VALUES (%s, %s, %s, %s, %s) RETURNING id
            """,
            (version_id, analysis_id, att, std_error, json.dumps(_clean_nan(payload))),
        )
        return cur.fetchone()["id"]


def list_versions(conn: psycopg.Connection, dataset_id: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, version, parent_version, change_note, n_rows, n_units, created_at
            FROM dataset_versions WHERE dataset_id=%s ORDER BY version
            """,
            (dataset_id,),
        )
        return cur.fetchall()


def get_dataset(conn: psycopg.Connection, dataset_id: int) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM datasets WHERE id=%s", (dataset_id,))
        row = cur.fetchone()
        if row is None:
            from .errors import NotFoundError

            raise NotFoundError(f"数据集 id={dataset_id} 不存在")
        return row


def create_dataset(conn: psycopg.Connection, *, name: str, description: str, period_kind: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO datasets (name, description, period_kind) VALUES (%s,%s,%s) RETURNING id",
            (name, description, period_kind),
        )
        return cur.fetchone()["id"]


def list_datasets(conn: psycopg.Connection) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, name, description, period_kind, created_at FROM datasets ORDER BY id")
        return cur.fetchall()


def list_results(conn: psycopg.Connection, version_id: int | None = None) -> list[dict]:
    q = """
        SELECT r.id, r.version_id, r.analysis_id, r.att, r.std_error, r.created_at,
               v.dataset_id, v.version, a.name AS analysis_name, a.control_strategy
        FROM results r
        JOIN dataset_versions v ON v.id = r.version_id
        JOIN analyses a ON a.id = r.analysis_id
    """
    args: tuple = ()
    if version_id is not None:
        q += " WHERE r.version_id=%s"
        args = (version_id,)
    q += " ORDER BY r.id"
    with conn.cursor() as cur:
        cur.execute(q, args)
        return cur.fetchall()

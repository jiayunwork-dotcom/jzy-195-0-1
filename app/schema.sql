-- 面板数据版本与 DID 结果管理（PostgreSQL 16）

CREATE TABLE IF NOT EXISTS datasets (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id      UUID NOT NULL REFERENCES datasets(id),
    version_no      BIGINT NOT NULL,
    parent_id       UUID REFERENCES dataset_versions(id),
    created_via     TEXT NOT NULL CHECK (created_via IN ('full', 'diff')),
    n_rows          INTEGER NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (dataset_id, version_no)
);

CREATE TABLE IF NOT EXISTS panel_rows (
    version_id  UUID NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
    unit_id     TEXT NOT NULL,
    period      TEXT NOT NULL,
    outcome     DOUBLE PRECISION NOT NULL,
    treated     BOOLEAN NOT NULL,
    covariates  JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (version_id, unit_id, period)
);

CREATE INDEX IF NOT EXISTS idx_panel_rows_version ON panel_rows(version_id);

CREATE TABLE IF NOT EXISTS version_changes (
    id          BIGSERIAL PRIMARY KEY,
    version_id  UUID NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
    op          TEXT NOT NULL CHECK (op IN ('upsert', 'delete')),
    unit_id     TEXT NOT NULL,
    period      TEXT NOT NULL,
    payload     JSONB
);

CREATE TABLE IF NOT EXISTS estimates (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id      UUID NOT NULL REFERENCES datasets(id),
    version_id      UUID NOT NULL REFERENCES dataset_versions(id),
    spec_hash       TEXT NOT NULL,
    spec            JSONB NOT NULL,
    engine_version  TEXT NOT NULL,
    result          JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (version_id, spec_hash)
);

CREATE INDEX IF NOT EXISTS idx_estimates_dataset ON estimates(dataset_id);

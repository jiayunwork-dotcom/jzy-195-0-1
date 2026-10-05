"""接口模型。

unit/period/outcome/treated 与协变量采用 Any 等宽容类型，使“字段语义”错误
（非数值、重复、撤回等）统一由引擎层的 PanelValidationError 报告，给出明确的
错误码与字段，而不是被 Pydantic 在反序列化阶段拦截成笼统的 422。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RowIn(BaseModel):
    unit: Any
    period: Any
    outcome: Any
    treated: Any
    covariates: dict[str, Any] | None = None

    model_config = {"extra": "ignore"}


class CreateDatasetIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class CreateVersionIn(BaseModel):
    rows: list[RowIn] = Field(min_length=1)


class ChangeIn(BaseModel):
    op: Literal["upsert", "delete"]
    unit: Any | None = None
    period: Any | None = None
    row: RowIn | dict[str, Any] | None = None


class CreateDiffVersionIn(BaseModel):
    changes: list[ChangeIn] = Field(min_length=1)


class EstimateSpec(BaseModel):
    # 当前数据集只有一个结果变量；字段保留以便未来支持多结果列选择。
    outcome: str = "outcome"
    control_group: Literal["never", "not_yet"] = "never"
    adjust_covariates: bool = False
    # 是否在响应里同时返回 TWFE 诊断。该开关不改变主估计，也不参与幂等键，
    # 因此同一主设定无论是否要诊断，都共享同一份缓存的主结果。
    include_twfe: bool = True


class CreateEstimateIn(BaseModel):
    version: str = "latest"  # "latest" 或版本号
    spec: EstimateSpec = EstimateSpec()

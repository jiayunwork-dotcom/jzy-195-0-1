"""请求/响应 Pydantic 模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ControlStrategy = Literal["never_treated", "not_yet_treated"]


class PanelRowIn(BaseModel):
    # 这些字段刻意不做 pydantic 强类型转换：非数值结果变量、无法排序的时期等
    # 错误由 panel.validate_panel 统一报告，保证错误里指出字段与原因。
    unit: Any = Field(..., description="单位编号（县 id），非空")
    period: Any = Field(..., description="时期：年份整数、'YYYY' 或 'YYYYQn'")
    outcome: Any = Field(..., description="结果变量，必须是有限数值")
    treated: Any = Field(..., description="该期是否处于处理状态（true/false）")
    covariates: dict[str, Any] = Field(default_factory=dict)


class CreateDatasetRequest(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""
    rows: list[PanelRowIn] = Field(..., min_length=1)


class DiffRequest(BaseModel):
    """差异提交：upsert（按 unit+period 覆盖/新增）与 deletes（按 unit+period 删除）。"""

    upserts: list[PanelRowIn] = Field(default_factory=list)
    deletes: list[dict[str, Any]] = Field(
        default_factory=list,
        description='每项形如 {"unit": "320123", "period": 2018}',
    )
    change_note: str = ""


class AnalysisRequest(BaseModel):
    name: str = Field("default", min_length=1)
    control_strategy: ControlStrategy = "never_treated"
    covariates: list[str] = Field(
        default_factory=list,
        description="作为线性控制变量的协变量列名；不填则不加协变量",
    )
    also_twfe: bool = Field(
        True,
        description="是否同时返回经典 TWFE 结果用于对照（主结论始终是堆叠 DID）",
    )

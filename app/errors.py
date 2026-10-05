"""统一的业务异常。

FastAPI 的异常处理器把它渲染成::

    {"error": {"code": ..., "message": ..., "details": [...]}}

这样数据错误和计量错误都能指出具体字段或原因。
"""

from __future__ import annotations


class AppError(Exception):
    status_code: int = 400
    code: str = "app_error"

    def __init__(self, message: str, code: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code


class ValidationError(AppError):
    status_code = 422
    code = "validation_error"

    def __init__(self, message: str, details: list[dict] | None = None):
        super().__init__(message)
        self.details = details or [{"reason": message}]


class EstimationError(AppError):
    status_code = 422
    code = "estimation_error"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"

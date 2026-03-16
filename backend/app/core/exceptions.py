import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette import status

logger = logging.getLogger(__name__)


# =========================================================
# Base Application Exception
# =========================================================

class AppException(Exception):
    """
    Base exception for application-level errors.
    """

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        error_code: str = "application_error",
        details: dict | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


# =========================================================
# Authentication / Authorization
# =========================================================

class AuthenticationError(AppException):
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="authentication_error",
        )


class AuthorizationError(AppException):
    def __init__(self, message: str = "Access denied"):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="authorization_error",
        )


# =========================================================
# Model Errors
# =========================================================

class ModelNotFoundError(AppException):
    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' not found",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="model_not_found",
        )


class ModelExecutionError(AppException):
    def __init__(self, message: str = "Model execution failed"):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="model_execution_error",
        )


# =========================================================
# Routing Errors
# =========================================================

class RoutingError(AppException):
    def __init__(self, message: str = "Routing failed"):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="routing_error",
        )


# =========================================================
# Billing Errors
# =========================================================

class BillingError(AppException):
    def __init__(self, message: str = "Billing processing failed"):
        super().__init__(
            message=message,
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            error_code="billing_error",
        )


# =========================================================
# Rate Limiting
# =========================================================

class RateLimitExceeded(AppException):
    def __init__(self):
        super().__init__(
            message="Rate limit exceeded",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="rate_limit_exceeded",
        )


# =========================================================
# Global Exception Handlers
# =========================================================

async def app_exception_handler(request: Request, exc: AppException):
    """
    Handles all custom AppException types.
    """

    logger.warning(
        "Application error",
        extra={
            "path": request.url.path,
            "error_code": exc.error_code,
            "details": exc.details,
        },
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "code": exc.error_code,
                "details": exc.details,
            }
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handles FastAPI HTTPExceptions.
    """

    logger.warning(
        "HTTP exception",
        extra={
            "path": request.url.path,
            "status_code": exc.status_code,
        },
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "code": "http_error",
            }
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Handles unexpected errors.
    """

    logger.exception("Unhandled exception occurred")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "Internal server error",
                "code": "internal_error",
            }
        },
    )
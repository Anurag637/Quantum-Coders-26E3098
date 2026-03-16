import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from backend.app.core.logging import logger
from backend.app.core.rate_limiter import check_rate_limit
from backend.app.core.config import settings


# ===============================
# Request ID Middleware
# ===============================

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response


# ===============================
# Logging Middleware
# ===============================

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.exception(
                "Unhandled exception",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                },
            )
            raise exc

        process_time = time.time() - start_time

        logger.info(
            "Request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration": round(process_time, 4),
                "request_id": getattr(request.state, "request_id", None),
            },
        )

        response.headers["X-Process-Time"] = str(round(process_time, 4))
        return response


# ===============================
# Rate Limiting Middleware
# ===============================

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)

        client_ip = request.client.host

        allowed = await check_rate_limit(client_ip)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
            )

        return await call_next(request)


# ===============================
# Security Headers Middleware
# ===============================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        response: Response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        if not settings.DEBUG:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response
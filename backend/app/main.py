import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, generate_latest
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.core.config import settings
from backend.app.core.logging import setup_logging
from backend.app.api.router import api_router
from backend.app.lifespan import lifespan

# Resolve frontend directory relative to the project root:
# main.py is in backend/app/main.py, so we go up 3 levels to reach the root.
_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_FRONTEND_DIR = _BASE_DIR / "frontend"

print(f"DEBUG: __file__ is {__file__}")
print(f"DEBUG: _BASE_DIR resolved to {_BASE_DIR}")
print(f"DEBUG: _FRONTEND_DIR resolved to {_FRONTEND_DIR} (exists: {_FRONTEND_DIR.exists()})")


# ===============================
# Metrics
# ===============================

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
)


# ===============================
# App Initialization
# ===============================

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
)


# ===============================
# Middleware
# ===============================

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        response = await call_next(request)

        process_time = time.time() - start_time

        REQUEST_COUNT.labels(
            request.method,
            request.url.path,
            response.status_code,
        ).inc()

        REQUEST_LATENCY.labels(
            request.method,
            request.url.path,
        ).observe(process_time)

        return response


app.add_middleware(MetricsMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===============================
# Routes
# ===============================

app.include_router(api_router, prefix="/api/v1")

# Serve static frontend assets under /frontend
if _FRONTEND_DIR.exists():
    app.mount(
        "/frontend",
        StaticFiles(directory=str(_FRONTEND_DIR), html=False),
        name="frontend-static",
    )


@app.get("/app", include_in_schema=False)
async def app_page():
    """Serve the main frontend HTML."""
    index_path = _FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse(
            status_code=500, 
            content={"detail": f"Frontend index.html not found at {index_path.absolute()}"}
        )
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root to the frontend UI."""
    return RedirectResponse(url="/app")


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "marker": "RELOAD_V3"}


@app.get("/version", tags=["System"])
async def version():
    return {"version": app.version}


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    return generate_latest()


# ===============================
# Global Exception Handler
# ===============================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )
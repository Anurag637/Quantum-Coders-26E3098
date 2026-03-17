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
from backend.app.services.groq_client import GroqClient
from backend.app.services.huggingface_client import HuggingFaceClient

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


@app.get("/debug/providers", include_in_schema=False)
async def debug_providers():
    """Debug whether provider keys are loaded (does not expose secrets)."""
    hf = bool(getattr(settings, "HUGGINGFACE_API_KEY", None))
    groq = bool(getattr(settings, "GROQ_API_KEY", None))
    return {"huggingface_key_loaded": hf, "groq_key_loaded": groq}


@app.get("/debug/provider-clients", include_in_schema=False)
async def debug_provider_clients():
    """
    Confirms what the *clients* see (no secrets exposed).
    If Groq still returns mock despite key_loaded=true, this will reveal it.
    """
    groq_client = GroqClient()
    hf_client = HuggingFaceClient()

    groq_key = groq_client.api_key or ""
    hf_key = hf_client.api_key or ""
    return {
        "groq_client_key_loaded": bool(groq_key),
        "groq_client_key_prefix": groq_key[:4] if groq_key else None,
        "hf_client_key_loaded": bool(hf_key),
        "hf_client_key_prefix": hf_key[:3] if hf_key else None,
    }


# ===============================
# Global Exception Handler
# ===============================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )
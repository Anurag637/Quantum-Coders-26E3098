import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.core.config import get_settings
from backend.app.core.logging import setup_logging
from backend.app.core.telemetry import init_telemetry
from backend.app.core.exceptions import AppException, app_exception_handler, unhandled_exception_handler
from backend.app.models.registry import model_registry
from backend.app.persistence.session import init_db, close_db
from backend.app.api.router import api_router
from backend.app.cache.redis import init_redis, close_redis

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup
    setup_logging()
    await init_db()
    await init_redis()
    await model_registry.load_from_config(settings.MODEL_DIR)
    
    yield
    
    # Teardown
    await model_registry.shutdown_all()
    await close_redis()
    await close_db()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
)

# Exception Handlers
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# CORS
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Telemetry
init_telemetry(app)

# API Router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Mount frontend (skip on Vercel — static files are served from public/ by the CDN)
if not os.environ.get("VERCEL"):
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
    if os.path.isdir(frontend_dir):
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)


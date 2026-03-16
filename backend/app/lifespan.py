from contextlib import asynccontextmanager
from fastapi import FastAPI

from backend.app.core.logging import setup_logging
from backend.app.core.config import settings
from backend.app.persistence.session import init_db, close_db
from backend.app.cache.redis import init_redis, close_redis
from backend.app.models.registry import model_registry
from backend.app.models.health_monitor import run_startup_checks


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.

    Handles:
    - Logging setup
    - Database initialization
    - Redis initialization
    - Model registry bootstrapping
    - Health checks
    """

    # ===============================
    # Startup Phase
    # ===============================

    setup_logging()

    print("Booting LLM Inference Platform...")

    # Initialize database
    await init_db()

    # Initialize Redis
    await init_redis()

    # Load registered models
    await model_registry.load_from_config(settings.MODEL_DIR)

    # Run health checks
    await run_startup_checks()

    print("System startup complete.")

    yield

    # ===============================
    # Shutdown Phase
    # ===============================

    print("Shutting down services...")

    await close_db()
    await close_redis()

    # Gracefully unload models
    await model_registry.shutdown_all()

    print("Shutdown complete.")
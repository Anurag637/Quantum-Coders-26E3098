from fastapi import APIRouter

from backend.app.api.v1.router import router as v1_router


# ===============================
# Root API Router
# ===============================

api_router = APIRouter()


# ===============================
# Versioned Routers
# ===============================

api_router.include_router(
    v1_router,
    prefix="",
)
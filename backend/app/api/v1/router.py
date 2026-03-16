from fastapi import APIRouter

from backend.app.api.v1.endpoints import (
    models,
    routing,
    chat,
    billing,
    monitoring,
    users,
)


# ===============================
# API v1 Router
# ===============================

router = APIRouter()


# Model Registry Endpoints
router.include_router(
    models.router,
    prefix="/models",
    tags=["Models"],
)

# Routing Analysis
router.include_router(
    routing.router,
    prefix="/routing",
    tags=["Routing"],
)

# Chat Completions (OpenAI-compatible)
router.include_router(
    chat.router,
    prefix="/chat",
    tags=["Chat"],
)

# Billing & Cost Tracking
router.include_router(
    billing.router,
    prefix="/billing",
    tags=["Billing"],
)

# Monitoring & Metrics
router.include_router(
    monitoring.router,
    prefix="/monitoring",
    tags=["Monitoring"],
)

# User Management
router.include_router(
    users.router,
    prefix="/users",
    tags=["Users"],
)
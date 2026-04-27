from fastapi import APIRouter

from backend.app.api.v1.endpoints import routing, models, monitoring

router = APIRouter()

router.include_router(routing.router, prefix="/routing", tags=["Routing"])
router.include_router(models.router, prefix="/models", tags=["Models"])
router.include_router(monitoring.router, prefix="/monitoring", tags=["Monitoring"])

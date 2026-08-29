"""
FarmRisk API Gateway.
Aggregates all API feature routers:
  - health.py   -> Root ('/') and health check ('/health')
  - location.py -> Indian village/town location search ('/api/location')
  - advisory/   -> Full agro-advisory, weather summary, pest card, what-to-do ('/api/advisory')
"""

from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.location import router as location_router, resolver as location_resolver
from app.api.advisory import router as advisory_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(location_router)
api_router.include_router(advisory_router)

__all__ = [
    "api_router",
    "health_router",
    "location_router",
    "location_resolver",
    "advisory_router",
]

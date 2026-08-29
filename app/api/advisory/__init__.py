"""
Advisory API package.
Aggregates all advisory-related feature routers:
  - overview.py        -> POST /api/advisory (canonical crop advisory)
  - weather_summary.py -> POST /api/advisory/weather-summary (24h weather summary)
  - pest_card.py       -> POST /api/advisory/pest-card (crop-specific pest & disease card)
  - what_to_do.py      -> POST /api/advisory/what-to-do (daily action recommendations)

To add a new feature:
  1. Create your endpoint in a new file (e.g. `irrigation.py`).
  2. Define `router = APIRouter()`.
  3. Include it below: `router.include_router(irrigation_router)`.

To remove a feature:
  Simply comment out or remove its `router.include_router(...)` line below.
"""

from fastapi import APIRouter

from app.api.advisory.overview import router as overview_router
from app.api.advisory.weather_summary import router as weather_summary_router
from app.api.advisory.pest_card import router as pest_card_router
from app.api.advisory.what_to_do import router as what_to_do_router
from app.api.advisory.dependencies import (
    context_builder,
    advisory_engine,
    translation_service,
    retriever,
    weather_service,
)

# Combined Advisory Router
router = APIRouter(prefix="/api/advisory", tags=["Advisory"])

# Mount individual feature routers
router.include_router(overview_router)
router.include_router(weather_summary_router)
router.include_router(pest_card_router)
router.include_router(what_to_do_router)

__all__ = [
    "router",
    "overview_router",
    "weather_summary_router",
    "pest_card_router",
    "what_to_do_router",
    "context_builder",
    "advisory_engine",
    "translation_service",
    "retriever",
    "weather_service",
]

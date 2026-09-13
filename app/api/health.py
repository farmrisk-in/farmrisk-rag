"""
Health & Root Endpoints.
GET / & HEAD /             - Root landing page and Hugging Face health checking
GET /health & HEAD /health - System health and environment check
"""

from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse
from app.core.config import settings

router = APIRouter(tags=["Health"])

_STATIC_INDEX = Path(__file__).resolve().parent.parent / "static" / "index.html"


@router.get("/")
@router.head("/")
async def root(request: Request):
    """Root landing page showing backend health, live docs links, and API catalog."""
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return JSONResponse(
            content={
                "message": "FarmRisk API is running. Visit /docs for documentation.",
                "status": "healthy",
                "environment": settings.APP_ENV,
            }
        )

    if _STATIC_INDEX.exists():
        return FileResponse(_STATIC_INDEX, media_type="text/html")

    return JSONResponse(
        content={
            "message": "FarmRisk API is running. Visit /docs for documentation.",
            "status": "healthy",
        }
    )


@router.get("/health")
@router.head("/health")
async def health_check():
    """Simple API health check endpoint."""
    return {"status": "healthy", "environment": settings.APP_ENV}

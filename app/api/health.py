"""
Health & Root Endpoints.
GET / & HEAD /             - Root endpoint for Hugging Face Spaces health checking
GET /health & HEAD /health - System health and environment check
"""

from fastapi import APIRouter
from app.core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/")
@router.head("/")
async def root():
    """Root endpoint for Hugging Face health checking."""
    return {"message": "FarmRisk API is running. Visit /docs for documentation.", "status": "healthy"}


@router.get("/health")
@router.head("/health")
async def health_check():
    """Simple API health check endpoint."""
    return {"status": "healthy", "environment": settings.APP_ENV}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.config import settings
from app.core.logging import logger
from app.api.location import router as location_router, resolver as location_resolver
from app.api.advisory import router as advisory_router, weather_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info(f"Starting FarmRisk Backend in {settings.APP_ENV} mode...")
    yield
    # Shutdown actions
    logger.info("Shutting down FarmRisk Backend...")
    await location_resolver.close()
    await weather_service.close()
    logger.info("FarmRisk Backend shutdown complete.")

app = FastAPI(
    title="FarmRisk AI Backend API",
    description="Production-grade agrometeorological advisory and village resolution system.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(location_router)
app.include_router(advisory_router)

@app.get("/health", tags=["Health"])
async def health_check():
    """Simple API health check endpoint."""
    return {"status": "healthy", "environment": settings.APP_ENV}

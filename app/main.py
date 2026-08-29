from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.config import settings
from app.core.logging import logger
from app.api import api_router, location_resolver

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting FarmRisk Backend in {settings.APP_ENV} mode...")
    yield
    logger.info("Shutting down FarmRisk Backend...")
    await location_resolver.close()
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

# Mount all modular feature routers
app.include_router(api_router)



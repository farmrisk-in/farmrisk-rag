from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from app.models.schemas import AdvisoryRequest, AdvisoryResponse
from app.services.location import NominatimLocationResolver
from app.services.weather import WeatherService
from app.services.season import MonthSeasonResolver
from app.rag.retriever import AdvisoryRetriever
from app.llm.advisory_engine import AdvisoryEngine
from app.services.translation import TranslationService
from app.core.caching import cache_manager
from app.core.logging import logger

router = APIRouter(prefix="/api/advisory", tags=["Advisory"])

# Services
location_resolver = NominatimLocationResolver()
weather_service = WeatherService()
season_resolver = MonthSeasonResolver()

# Try initializing vector retriever, fallback if Supabase config is missing/offline
try:
    retriever = AdvisoryRetriever()
except Exception as e:
    logger.warning(f"Failed to initialize AdvisoryRetriever: {e}. Falling back to empty search context.")
    retriever = None

advisory_engine = AdvisoryEngine()
translation_service = TranslationService()

@router.post("", response_model=Dict[str, Any])
async def generate_crop_advisory(request: AdvisoryRequest):
    """
    Generate agrometeorological advisory for a given coordinate grid location and crop.
    Retrieves weather context, performs Supabase RAG search, uses Gemini Flash, and translates.
    """
    logger.info(
        f"Processing advisory request: crop={request.crop}, lat={request.latitude}, lon={request.longitude}, lang={request.language}"
    )

    try:
        # Step 1: Geocode coordinates to get state and village details
        location_detail = await location_resolver.reverse_geocode(request.latitude, request.longitude)
        logger.info(
            f"Geocoded coordinates: state={location_detail.state}, district={location_detail.district}, village={location_detail.village}"
        )
        
        # Step 2: Fetch current weather and compute forecast hash
        weather_data = await weather_service.fetch_10day_forecast(request.latitude, request.longitude)
        weather_hash = weather_data["weather_hash"]
        
        # Step 3: Resolve current agricultural season
        season = season_resolver.resolve_season(request.latitude, request.longitude)
        logger.info(f"Resolved season: {season}")

        # Step 4: Spatial Cache Lookup (Check if advisory exists for this crop + location grid + weather forecast)
        english_advisory = cache_manager.get_advisory(request.crop, request.latitude, request.longitude, weather_hash)
        
        if english_advisory:
            logger.info("Advisory spatial cache HIT.")
            # Verify structure and convert back to schema if needed
            advisory_obj = AdvisoryResponse(**english_advisory)
        else:
            logger.info("Advisory spatial cache MISS. Initiating RAG pipeline.")
            
            # Step 5: Retrieve relevant RAG guidelines from Supabase
            rag_context = []
            if retriever:
                # Query vector database
                rag_context = retriever.retrieve(
                    crop=request.crop,
                    state=location_detail.state,
                    season=season
                )
            
            logger.info(f"Retrieved {len(rag_context)} documents from Supabase.")
            
            # Step 6: Generate crop advisory via Gemini Flash
            advisory_obj = await advisory_engine.generate_advisory(
                crop=request.crop,
                state=location_detail.state,
                district=location_detail.district,
                village=location_detail.village,
                season=season,
                weather_data=weather_data,
                rag_context=rag_context,
                crop_stage=request.crop_stage
            )
            
            # Save raw English advisory to spatial cache
            cache_manager.set_advisory(
                crop=request.crop,
                latitude=request.latitude,
                longitude=request.longitude,
                weather_hash=weather_hash,
                advisory_data=advisory_obj.model_dump()
            )
            logger.info("Cached raw English advisory.")

        # Step 7: Translation Cache Lookup
        english_dump = advisory_obj.model_dump()
        translated_advisory = cache_manager.get_translation(english_dump, request.language)
        
        if translated_advisory:
            logger.info(f"Translation cache HIT for language: {request.language}")
            return translated_advisory
            
        logger.info(f"Translation cache MISS for language: {request.language}. Triggering translator.")
        
        # Step 8: Translate response values using LLM provider pipeline
        translation_result = await translation_service.translate_advisory(advisory_obj, request.language)
        translated_advisory = translation_result.data
        
        # Cache translation result only if translation succeeded
        if translation_result.translated:
            cache_manager.set_translation(english_dump, request.language, translated_advisory)
            logger.info(f"Cached translation for language: {request.language}")
        else:
            logger.warning("Translation cache skipped because translation failed")
        
        # Inject resolved geographic location details into response metadata for UI display
        translated_advisory["location"] = location_detail.model_dump()
        
        return translated_advisory

    except Exception as e:
        logger.error(f"Failed to generate crop advisory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Advisory generation failed: {str(e)}")

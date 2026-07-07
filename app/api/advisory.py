"""
POST /api/advisory — thin orchestration layer.

Flow:
  AIAdvisoryRequest (full frontend payload)
    → Pydantic validation
    → AdvisoryContextBuilder.build()
    → cache lookup (English advisory)
    → AdvisoryRetriever.retrieve()
    → AdvisoryEngine.generate_advisory()
    → cache set
    → TranslationService.translate_advisory()
    → return AdvisoryResponse
"""

import time
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.models.schemas import AIAdvisoryRequest, AdvisoryResponse
from app.services.context_builder import AdvisoryContextBuilder
from app.rag.retriever import AdvisoryRetriever, RetrievalContext
from app.llm.advisory_engine import AdvisoryEngine, InsufficientKnowledgeError, AdvisoryGenerationError
from app.services.translation import TranslationService
from app.core.caching import cache_manager
from app.core.logging import logger

router = APIRouter(prefix="/api/advisory", tags=["Advisory"])

# Services (initialised once at import time)
context_builder = AdvisoryContextBuilder()
advisory_engine = AdvisoryEngine()
translation_service = TranslationService()

# Retriever: gracefully degrades if Supabase is not configured
try:
    retriever = AdvisoryRetriever()
except Exception as e:
    logger.warning(f"AdvisoryRetriever init failed: {e}. RAG will be skipped.")
    retriever = None

# Kept for main.py lifespan compatibility (no-op — no network client here)
weather_service = None


@router.post("", response_model=Dict[str, Any])
async def generate_crop_advisory(request: AIAdvisoryRequest):
    """
    Generate agrometeorological advisory from the full frontend data payload.
    """
    t_start = time.perf_counter()

    logger.info(
        f"Advisory request | crop={request.cropId} "
        f"lat={request.location.lat:.4f} lng={request.location.lng:.4f} "
        f"lang={request.language}"
    )

    try:
        # ----------------------------------------------------------------
        # Step 1: Build compact deterministic AdvisoryContext
        # ----------------------------------------------------------------
        context = context_builder.build(request)
        av = context.availability

        logger.info(
            f"Context built | crop={context.crop_context.crop_name} "
            f"season={context.crop_context.season} "
            f"stage={context.crop_context.crop_stage} "
            f"state={context.location.state} district={context.location.district} | "
            f"forecast={av.corrected_forecast_available} "
            f"soil={av.soil_moisture_available} "
            f"lightning={av.lightning_available}"
        )

        # ----------------------------------------------------------------
        # Step 2: Determine cache key from corrected forecast fingerprint
        # ----------------------------------------------------------------
        fingerprint = (
            context.forecast_summary.forecast_fingerprint
            if context.forecast_summary
            else "no_forecast"
        )

        # ----------------------------------------------------------------
        # Step 3: English advisory cache lookup
        # ----------------------------------------------------------------
        english_advisory = cache_manager.get_advisory(
            crop=context.crop_context.crop_name,
            latitude=request.location.lat,
            longitude=request.location.lng,
            weather_hash=fingerprint,
        )

        if english_advisory:
            logger.info("Advisory cache HIT (English).")
            advisory_obj = AdvisoryResponse(**english_advisory)
        else:
            logger.info("Advisory cache MISS. Running RAG pipeline.")

            # ------------------------------------------------------------
            # Step 4: Build retrieval context (skipped for general crop)
            # ------------------------------------------------------------
            is_general = request.cropId.lower().strip() == "general"

            ret_ctx = None
            if not is_general and (context.forecast_summary or context.soil_moisture_summary):
                sm_avg = (
                    context.soil_moisture_summary.forecast_average_percentile
                    if context.soil_moisture_summary else 50.0
                )
                sm_trend = (
                    context.soil_moisture_summary.soil_moisture_trend
                    if context.soil_moisture_summary else "unknown"
                )
                lg_cat = (
                    context.lightning_summary.category
                    if context.lightning_summary else "unknown"
                )
                fs = context.forecast_summary
                ret_ctx = RetrievalContext(
                    crop_stage=context.crop_context.crop_stage,
                    rainfall_pattern=fs.rainfall_pattern if fs else "unknown",
                    total_rainfall_mm=fs.total_rainfall_mm if fs else 0.0,
                    min_temp_c=fs.minimum_temperature_c if fs else context.current_weather.temperature_c,
                    max_temp_c=fs.maximum_temperature_c if fs else context.current_weather.temperature_c,
                    soil_moisture_trend=sm_trend,
                    soil_moisture_percentile_avg=sm_avg,
                    lightning_category=lg_cat,
                )

            # ------------------------------------------------------------
            # Step 5: RAG retrieval (skipped for general crop)
            # ------------------------------------------------------------
            rag_chunks = []
            if is_general:
                logger.info("General crop selected — skipping RAG, using direct LLM generation.")
            elif retriever:
                t_rag = time.perf_counter()
                rag_chunks = retriever.retrieve(
                    crop=context.crop_context.crop_name,
                    state=context.location.state,
                    season=context.crop_context.season,
                    retrieval_context=ret_ctx,
                )
                logger.info(
                    f"RAG retrieval done in {time.perf_counter() - t_rag:.2f}s | "
                    f"{len(rag_chunks)} chunks"
                )
            else:
                logger.warning("RAG retriever unavailable — generating without ICAR context.")

            # ------------------------------------------------------------
            # Step 6: Generate English advisory
            # ------------------------------------------------------------
            t_gen = time.perf_counter()
            advisory_obj = await advisory_engine.generate_advisory(
                context=context,
                rag_chunks=rag_chunks,
            )
            logger.info(f"Advisory generation done in {time.perf_counter() - t_gen:.2f}s")

            # ------------------------------------------------------------
            # Step 7: Cache English advisory
            # ------------------------------------------------------------
            cache_manager.set_advisory(
                crop=context.crop_context.crop_name,
                latitude=request.location.lat,
                longitude=request.location.lng,
                weather_hash=fingerprint,
                advisory_data=advisory_obj.model_dump(),
            )
            logger.info("Cached English advisory.")

        # ----------------------------------------------------------------
        # Step 8: Translation cache lookup / translate
        # ----------------------------------------------------------------
        english_dump = advisory_obj.model_dump()
        translated = cache_manager.get_translation(english_dump, request.language)

        if translated:
            logger.info(f"Translation cache HIT for lang={request.language}")
        else:
            logger.info(f"Translation cache MISS for lang={request.language}. Translating...")
            t_tr = time.perf_counter()
            result = await translation_service.translate_advisory(advisory_obj, request.language)
            logger.info(f"Translation done in {time.perf_counter() - t_tr:.2f}s | translated={result.translated}")
            translated = result.data
            if result.translated:
                cache_manager.set_translation(english_dump, request.language, translated)

        total = time.perf_counter() - t_start
        logger.info(f"Advisory request complete in {total:.2f}s")
        return translated

    except AdvisoryGenerationError as e:
        logger.error(f"Advisory generation exhausted all providers: {e}")
        raise HTTPException(
            status_code=502,
            detail="Advisory generation failed: the AI service was unable to produce a valid response.",
        )
    except Exception as e:
        logger.error(f"Unexpected advisory error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during advisory generation.")

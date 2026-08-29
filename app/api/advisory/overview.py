"""
Overview / Full Advisory Endpoint.
POST /api/advisory
Generates canonical multi-paragraph agrometeorological advisory, irrigation insight,
and attaches RAG sources and translations.
"""

import time
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.models.schemas import AIAdvisoryRequest, AdvisoryResponse
from app.rag.retriever import RetrievalContext
from app.llm.advisory_engine import AdvisoryGenerationError
from app.services.irrigation import compute_irrigation_insight_for_request
from app.core.caching import advisory_cache, translation_cache, lock_manager
from app.core.logging import logger
from app.api.advisory.dependencies import (
    context_builder,
    advisory_engine,
    translation_service,
    retriever,
)

router = APIRouter()


@router.post("", response_model=Dict[str, Any])
@router.post("/", response_model=Dict[str, Any], include_in_schema=False)
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
        # Irrigation insight (deterministic, independent of the LLM)
        # Uses the existing decision engine on the request's daily series.
        # ----------------------------------------------------------------
        irrigation_insight = compute_irrigation_insight_for_request(
            request, crop_stage=context.crop_context.crop_stage
        )
        logger.info(f"Irrigation insight: {'generated' if irrigation_insight else 'not available'}")

        # ----------------------------------------------------------------
        # Step 2: Determine cache key parameters
        # ----------------------------------------------------------------
        fingerprint = (
            context.forecast_summary.forecast_fingerprint
            if context.forecast_summary
            else "no_forecast"
        )
        village_id = request.forecastData.village_id if request.forecastData else None

        # Build weather hash incorporating forecast fingerprint and soil moisture availability
        sm_status = "sm_yes" if context.availability.soil_moisture_available else "sm_no"
        weather_hash = f"{fingerprint}_{sm_status}"

        # Build location identifier and key
        advisory_key = advisory_cache.get_key(
            crop=context.crop_context.crop_name,
            latitude=request.location.lat,
            longitude=request.location.lng,
            weather_hash=weather_hash,
            village_id=village_id,
        )

        # ----------------------------------------------------------------
        # Step 3: English advisory cache lookup with Request Deduplication Lock
        # ----------------------------------------------------------------
        advisory_lock = await lock_manager.get_lock(advisory_key)
        if advisory_lock.locked():
            logger.info("Waiting for existing generation")

        async with advisory_lock:
            # Double check cache inside lock
            english_advisory = advisory_cache.get(advisory_key)
            if english_advisory:
                logger.info("Advisory Cache HIT")
                remaining_ttl = advisory_cache.ttl(advisory_key)
                logger.info(f"Cache TTL remaining: {remaining_ttl}")
                advisory_obj = AdvisoryResponse(**english_advisory)
                advisory_obj.irrigation_insight = irrigation_insight
            else:
                logger.info("Advisory Cache MISS")
                logger.info("Lock acquired")

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
                advisory_obj = await advisory_engine.generate_advisory(
                    context=context,
                    rag_chunks=rag_chunks,
                )
                advisory_obj.sources = rag_chunks
                advisory_obj.irrigation_insight = irrigation_insight
                logger.info("Generation finished")

                # ------------------------------------------------------------
                # Step 7: Cache English advisory
                # ------------------------------------------------------------
                advisory_cache.set(advisory_key, advisory_obj.model_dump())
                logger.info("Cache stored")

        # ----------------------------------------------------------------
        # Step 8: Translation cache lookup / translate with Deduplication Lock
        # ----------------------------------------------------------------
        english_dump = advisory_obj.model_dump()
        is_english = request.language.lower().strip() in ("en", "english")

        if is_english:
            translated = english_dump
        else:
            trans_key = translation_cache.get_key(english_dump, request.language)
            trans_lock = await lock_manager.get_lock(trans_key)
            if trans_lock.locked():
                logger.info("Waiting for existing generation")

            async with trans_lock:
                # Double check translation cache inside lock
                translated = translation_cache.get(trans_key)
                if translated:
                    logger.info("Translation Cache HIT")
                    remaining_ttl = translation_cache.ttl(trans_key)
                    logger.info(f"Cache TTL remaining: {remaining_ttl}")
                else:
                    logger.info("Translation Cache MISS")
                    logger.info("Lock acquired")

                    t_tr = time.perf_counter()
                    result = await translation_service.translate_advisory(advisory_obj, request.language)
                    logger.info(f"Translation done in {time.perf_counter() - t_tr:.2f}s | translated={result.translated}")
                    translated = result.data
                    logger.info("Generation finished")

                    if result.translated:
                        translation_cache.set(trans_key, translated)
                        logger.info("Cache stored")

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

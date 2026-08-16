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

import asyncio
import time
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional

from app.models.schemas import AIAdvisoryRequest, AdvisoryResponse
from app.services.context_builder import AdvisoryContextBuilder
from app.rag.retriever import AdvisoryRetriever, RetrievalContext
from app.llm.advisory_engine import AdvisoryEngine, InsufficientKnowledgeError, AdvisoryGenerationError
from app.llm.providers import get_primary_provider, get_fallback_provider
from app.services.translation import TranslationService
from app.services.irrigation import compute_irrigation_insight_for_request, compute_irrigation_decision_and_insight
from app.services.what_to_do import build_weather_todos_from_request, select_what_to_do
from app.services.pest_disease_card import (
    build_pest_disease_card,
    call_llm_text,
)
from app.core.caching import cache_manager, advisory_cache, translation_cache, lock_manager
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
                t_gen = time.perf_counter()
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


@router.post("/weather-summary", response_model=Dict[str, Any])
async def generate_weather_summary(request: AIAdvisoryRequest):
    """
    Generate a simple 1-2 sentence weather summary using the primary/fallback LLM providers.
    """
    t_start = time.perf_counter()

    logger.info(
        f"Weather summary request | "
        f"lat={request.location.lat:.4f} lng={request.location.lng:.4f} "
        f"lang={request.language}"
    )

    try:
        # Build compact deterministic AdvisoryContext
        context = context_builder.build(request)
        loc = context.location
        cw = context.current_weather

        # Extract hourly data for the next 24 hours
        h = request.weatherData.hourly
        temps_24h = h.temperature_2m[:24] if h.temperature_2m else []
        precip_prob_24h = h.precipitation_probability[:24] if h.precipitation_probability else []
        wind_speeds_24h = h.wind_speed_10m[:24] if h.wind_speed_10m else []

        max_temp_24h = max(temps_24h) if temps_24h else cw.temperature_c
        min_temp_24h = min(temps_24h) if temps_24h else cw.temperature_c
        max_precip_prob = max(precip_prob_24h) if precip_prob_24h else 0.0
        max_wind_speed = max(wind_speeds_24h) if wind_speeds_24h else cw.wind_speed_kmh

        d = request.weatherData.daily
        precip_today = d.precipitation_sum[0] if d.precipitation_sum else 0.0

        # Construct a timeline of how weather changes over the next 24 hours
        # Pick 5 representative intervals (current hour, +6h, +12h, +18h, +23h)
        timeline = []
        for offset in [0, 6, 12, 18, 23]:
            if offset < len(h.time):
                t_str = h.time[offset]
                try:
                    time_display = t_str.split("T")[-1][:5]
                except Exception:
                    time_display = f"+{offset}h"
                temp = h.temperature_2m[offset] if offset < len(h.temperature_2m) else cw.temperature_c
                prob = h.precipitation_probability[offset] if offset < len(h.precipitation_probability) else 0.0
                wind = h.wind_speed_10m[offset] if offset < len(h.wind_speed_10m) else cw.wind_speed_kmh
                timeline.append(f"At {time_display}: Temp {temp:.1f}°C, Rain Prob {prob:.0f}%, Wind Speed {wind:.1f} km/h")
        timeline_str = "\n".join(timeline)

        # Construct prompt for the 1-2 sentence weather summary
        location_name = loc.name
        state_name = loc.state
        current_temp = cw.temperature_c
        current_condition = cw.weather_condition
        humidity = cw.relative_humidity_percent
        wind_speed = cw.wind_speed_kmh
        wind_gusts = cw.wind_gusts_kmh

        # Determine cache key parameters
        fingerprint = (
            context.forecast_summary.forecast_fingerprint
            if context.forecast_summary
            else "no_forecast"
        )
        village_id = request.forecastData.village_id if request.forecastData else None

        if village_id is not None:
            location_identifier = str(village_id)
        else:
            lat_grid = f"{request.location.lat:.3f}"
            lon_grid = f"{request.location.lng:.3f}"
            location_identifier = f"{lat_grid}_{lon_grid}"
        
        summary_key = f"weather_summary:{location_identifier}:{fingerprint}"

        # Acquire lock and check cache
        summary_lock = await lock_manager.get_lock(summary_key)
        if summary_lock.locked():
            logger.info("Waiting for existing weather summary generation")

        async with summary_lock:
            # Check cache
            cached_summary = cache_manager.get(summary_key)
            if cached_summary:
                logger.info("Weather Summary Cache HIT")
                weather_summary_text = cached_summary["advisory_summary"]
            else:
                logger.info("Weather Summary Cache MISS")
                logger.info("Lock acquired for weather summary generation")

                prompt = (
                    f"You are a professional weather assistant. Generate a crisp, actionable weather summary of exactly one to two sentences "
                    f"for the NEXT 24 HOURS in {location_name}, {state_name}.\n\n"
                    f"Facts to use:\n"
                    f"- Current Temperature: {current_temp:.1f}°C\n"
                    f"- Current Weather Condition: {current_condition}\n"
                    f"- Current Relative Humidity: {humidity:.0f}%\n"
                    f"- Current Wind: {wind_speed:.1f} km/h (gusts up to {wind_gusts:.1f} km/h)\n"
                    f"- Next 24 Hours Minimum Temperature: {min_temp_24h:.1f}°C\n"
                    f"- Next 24 Hours Maximum Temperature: {max_temp_24h:.1f}°C\n"
                    f"- Next 24 Hours Maximum Probability of Precipitation: {max_precip_prob:.0f}%\n"
                    f"- Expected precipitation for the first day: {precip_today:.1f} mm\n"
                    f"- Next 24 Hours Maximum Wind Speed: {max_wind_speed:.1f} km/h\n\n"
                    f"Hourly forecast timeline over the next 24 hours:\n"
                    f"{timeline_str}\n\n"
                    f"WRITING RULES:\n"
                    f"1. Lead with the single most important actionable takeaway (the thing that changes what a person does today).\n"
                    f"2. Use relative time cues ('in about 4 hours', 'after midnight') OR clock intervals ('between 2 pm and 4 pm'), not vague phrases like 'later today'.\n"
                    f"3. Be specific and quantitative: name the peak value and when it occurs (e.g. 'peaks at 42°C around 3 pm', 'gusts up to 45 km/h after 6 pm').\n"
                    f"4. Do NOT list every fact. Mention only what is notable or actionable. Skip mild, unremarkable conditions.\n"
                    f"5. No filler ('will experience', 'featuring'). Write direct, punchy statements.\n\n"
                    f"DECISION LOGIC (pick the dominant hazard; don't dilute the summary by covering everything):\n"
                    f"- If significant rain is expected (precip prob high or precip > ~2 mm), focus on rainfall timing and intensity (light/moderate/heavy downpour) and DE-EMPHASIZE temperature, since rain suppresses peak heat.\n"
                    f"- If it is a hot, dry day (max temp high, low precip prob), focus on peak temperature, when it hits, and heat-stress caution; mention UV if it maps to a clear-sky midday window.\n"
                    f"- If winds/gusts are strong (high max wind or gusts), call out the windy interval as the headline instead.\n"
                    f"- If humidity is high AND temperature is high, flag the 'feels hotter / muggy' discomfort rather than the raw number.\n"
                    f"- If nothing is extreme, give a brief reassuring summary with the min/max and the general condition.\n"
                    f"- Only mention UV index when skies are clear/mostly clear during daylight (UV is irrelevant under heavy cloud or at night).\n"
                    f"- If a sharp transition occurs (e.g. clear → thunderstorm), lead with the transition and its onset time.\n"
                    f"- Never warn about heat and heavy rain in the same breath as if both peak together; choose the one the data supports.\n"
                )

                primary = get_primary_provider()
                fallback = get_fallback_provider()

                raw = None
                try:
                    raw = await primary.generate_text(prompt=prompt, temperature=0.2)
                except Exception as e:
                    logger.warning(f"Primary provider failed for weather summary: {e}")
                    if fallback:
                        try:
                            raw = await fallback.generate_text(prompt=prompt, temperature=0.2)
                        except Exception as fe:
                            logger.error(f"Fallback provider failed for weather summary: {fe}")

                if raw is None:
                    raise HTTPException(status_code=502, detail="Failed to generate weather summary from LLM providers.")

                weather_summary_text = raw.strip()
                # Cache English version
                cache_manager.set(summary_key, {"advisory_summary": weather_summary_text}, ttl_seconds=43200)
                logger.info("Weather Summary Cache stored")

        # Wrap in AdvisoryResponse for translation pipeline compat
        advisory_obj = AdvisoryResponse(advisory_summary=weather_summary_text)

        # Translate if target language is not English
        is_english = request.language.lower().strip() in ("en", "english")
        if is_english:
            translated = advisory_obj.model_dump()
        else:
            trans_key = translation_cache.get_key(advisory_obj.model_dump(), request.language)
            trans_lock = await lock_manager.get_lock(trans_key)
            if trans_lock.locked():
                logger.info("Waiting for existing translation generation")

            async with trans_lock:
                translated = translation_cache.get(trans_key)
                if translated:
                    logger.info("Weather Summary Translation Cache HIT")
                else:
                    logger.info("Weather Summary Translation Cache MISS")
                    result = await translation_service.translate_advisory(advisory_obj, request.language)
                    translated = result.data
                    if result.translated:
                        translation_cache.set(trans_key, translated)
                        logger.info("Weather Summary Translation Cache stored")

        total = time.perf_counter() - t_start
        logger.info(f"Weather summary request complete in {total:.2f}s")
        return translated

    except Exception as e:
        logger.error(f"Unexpected weather summary error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Pest & Disease card — crop-specific, RAG-grounded, deterministic risk band
# ---------------------------------------------------------------------------

def _pest_inputs_from_request(request: "AIAdvisoryRequest"):
    """Derive the exact pest-index inputs from the frontend payload.

    Mirrors the frontend `useRisk` computation so the card's deterministic
    score/band always agrees with the Weather Risk pest score:
      - avg_max_temp : mean of the next 5 bias-corrected max temps
      - humidity     : current relative humidity
      - soil_percentile : latest historical soil-moisture percentile
      - rainy_days   : count of next-5 days with rain >= 1 mm
      - total_rainfall : sum of next-5 day rain
    """
    daily = request.weatherData.daily

    # Next-5-day bias-corrected forecast (preferred), else weather API daily
    forecast_days = []
    if request.forecastData and request.forecastData.forecast and request.forecastData.forecast.forecast:
        forecast_days = sorted(
            request.forecastData.forecast.forecast, key=lambda d: d.date
        )[:5]

    avg_max_temp = None
    if forecast_days:
        avg_max_temp = sum(d.tmax_corrected for d in forecast_days) / len(forecast_days)
    elif daily.temperature_2m_max:
        vals = daily.temperature_2m_max[:5]
        avg_max_temp = sum(vals) / len(vals)

    humidity = request.weatherData.current.relative_humidity_2m

    rainy_days = None
    total_rainfall = None
    if forecast_days:
        rainy_days = sum(1 for d in forecast_days if d.pcp_corrected >= 1.0)
        total_rainfall = sum(d.pcp_corrected for d in forecast_days)
    elif daily.precipitation_sum:
        vals = daily.precipitation_sum[:5]
        total_rainfall = sum(vals)
        rainy_days = sum(1 for v in vals if v >= 1.0)

    soil_percentile = None
    if (
        request.forecastData
        and request.forecastData.soil_moisture
        and request.forecastData.soil_moisture.soil_moisture
    ):
        historical = [
            r for r in request.forecastData.soil_moisture.soil_moisture
            if r.is_forecast == 0
        ]
        if historical:
            soil_percentile = historical[-1].sm_percentile

    return avg_max_temp, humidity, soil_percentile, rainy_days, total_rainfall


async def _get_pest_card(request: "AIAdvisoryRequest", context) -> Dict[str, Any]:
    """Build the crop-specific English Pest & Disease card with cache + dedup lock.

    Shared by POST /api/advisory/pest-card and POST /api/advisory/what-to-do so
    both reuse the SAME cache key — a card generated by either endpoint is a
    cache HIT for the other (no duplicate RAG retrieval or LLM call).
    """
    is_general = request.cropId.lower().strip() == "general"

    # Cache key (crop + location + weather/soil fingerprint)
    fingerprint = (
        context.forecast_summary.forecast_fingerprint
        if context.forecast_summary else "no_forecast"
    )
    village_id = request.forecastData.village_id if request.forecastData else None
    sm_status = "sm_yes" if context.availability.soil_moisture_available else "sm_no"
    weather_hash = f"{fingerprint}_{sm_status}"
    card_key = (
        "pest_card:"
        + advisory_cache.get_key(
            crop=context.crop_context.crop_name,
            latitude=request.location.lat,
            longitude=request.location.lng,
            weather_hash=weather_hash,
            village_id=village_id,
        )
    )

    # Cache lookup FIRST — RAG + LLM only run on a MISS.
    card_lock = await lock_manager.get_lock(card_key)
    async with card_lock:
        card = cache_manager.get(card_key)
        if card:
            logger.info("Pest card Cache HIT")
            return card

        logger.info("Pest card Cache MISS")

        # Crop-filtered RAG retrieval (skipped for general crop), offloaded to
        # a worker thread (embedding encode + pgvector RPCs are sync).
        rag_chunks: List[Dict[str, Any]] = []
        if is_general:
            logger.info("General crop selected — skipping RAG for pest & disease card.")
        elif retriever:
            ret_ctx = _build_pest_retrieval_context(context)
            t_rag = time.perf_counter()
            rag_chunks = await asyncio.to_thread(
                retriever.retrieve,
                crop=context.crop_context.crop_name,
                state=context.location.state,
                season=context.crop_context.season,
                retrieval_context=ret_ctx,
            )
            logger.info(
                f"Pest card RAG retrieval done in {time.perf_counter() - t_rag:.2f}s | "
                f"{len(rag_chunks)} chunks for crop='{context.crop_context.crop_name}'"
            )
        else:
            logger.warning("RAG retriever unavailable — pest card generated without ICAR context.")

        # Deterministic pest inputs + score + band
        avg_max_temp, humidity, soil_percentile, rainy_days, total_rainfall = (
            _pest_inputs_from_request(request)
        )

        card = await build_pest_disease_card(
            avg_max_temp=avg_max_temp,
            humidity=humidity,
            soil_percentile=soil_percentile,
            rainy_days=rainy_days,
            total_rainfall=total_rainfall,
            crop_name=context.crop_context.crop_name,
            crop_stage=context.crop_context.crop_stage,
            rag_chunks=rag_chunks,
            llm_call=call_llm_text,
            api_key="",  # providers are configured app-wide
            crop_id=request.cropId,
            season=context.crop_context.season,
        )
        card["crop_id"] = request.cropId
        card["crop_name"] = context.crop_context.crop_name
        card["crop_stage"] = context.crop_context.crop_stage
        card["season"] = context.crop_context.season
        card["is_general"] = is_general
        cache_manager.set(card_key, card)
        logger.info("Pest card Cache stored")
        return card


@router.post("/pest-card", response_model=Dict[str, Any])
async def generate_pest_disease_card(request: AIAdvisoryRequest):
    """
    Build the crop-specific "Pest & Disease" card.

    Flow:
      1. Build the deterministic AdvisoryContext (crop name/stage/season).
      2. RAG-retrieve ICAR chunks filtered by the selected crop.
      3. Compute the deterministic pest score + band (score_pest / pest_band).
      4. LLM phrases summary / potential pests / actions grounded ONLY in the
         retrieved chunks; the band can never be overridden by the model.
      5. General crop ("general") skips RAG and keeps the crop-agnostic card.
    """
    t_start = time.perf_counter()

    logger.info(
        f"Pest & Disease card request | crop={request.cropId} "
        f"lat={request.location.lat:.4f} lng={request.location.lng:.4f} "
        f"lang={request.language}"
    )

    # ------------------------------------------------------------------
    # Step 1: Deterministic context (crop name, stage, season)
    # ------------------------------------------------------------------
    context = context_builder.build(request)

    # ------------------------------------------------------------------
    # Step 2: Card (shared helper — cache + RAG + LLM, English)
    # ------------------------------------------------------------------
    card = await _get_pest_card(request, context)

    # ------------------------------------------------------------------
    # Step 3: Translation cache lookup / translate with Deduplication Lock
    #   (mirrors the AI Advisory translation step in POST /api/advisory)
    # ------------------------------------------------------------------
    is_english = request.language.lower().strip() in ("en", "english")

    if is_english:
        translated = card
    else:
        trans_key = translation_cache.get_key(card, request.language)
        trans_lock = await lock_manager.get_lock(trans_key)
        if trans_lock.locked():
            logger.info("Waiting for existing pest card translation")

        async with trans_lock:
            # Double check translation cache inside lock
            translated = translation_cache.get(trans_key)
            if translated:
                logger.info("Pest card Translation Cache HIT")
            else:
                logger.info("Pest card Translation Cache MISS")
                t_tr = time.perf_counter()
                result = await translation_service.translate_pest_disease_card(card, request.language)
                logger.info(
                    f"Pest card translation done in {time.perf_counter() - t_tr:.2f}s | "
                    f"translated={result.translated}"
                )
                translated = result.data

                if result.translated:
                    translation_cache.set(trans_key, translated)
                    logger.info("Pest card translation cache stored")

    total = time.perf_counter() - t_start
    logger.info(f"Pest & Disease card request complete in {total:.2f}s")
    return translated


@router.post("/what-to-do", response_model=Dict[str, Any])
async def generate_what_to_do(request: AIAdvisoryRequest):
    """
    Build the "What To Do Today" card — an aggregation/display layer over the
    EXISTING recommendation systems (NOT a new engine).

    Flow:
      1. Deterministic irrigation decision + insight (one calculation).
      2. Weather to-dos (rain/temp/wind engine) — deterministic phrasing.
      3. Crop-specific Pest & Disease card (shared cache with /pest-card).
      4. Deterministic selection: best pest action + best irrigation
         recommendation, weather to-dos as fallback, at most TWO items.
      5. Translate the selected items' title/hint into the request language.

    The card does NOT wait for the AI Overview: the advisory LLM never runs here.
    """
    t_start = time.perf_counter()

    logger.info(
        f"What-to-do request | crop={request.cropId} "
        f"lat={request.location.lat:.4f} lng={request.location.lng:.4f} "
        f"lang={request.language}"
    )

    try:
        # ------------------------------------------------------------------
        # Step 1: Deterministic context (crop name/stage/season)
        # ------------------------------------------------------------------
        context = context_builder.build(request)

        # ------------------------------------------------------------------
        # Step 2: Irrigation decision + insight (single deterministic walk)
        # ------------------------------------------------------------------
        irr_result = compute_irrigation_decision_and_insight(
            request, crop_stage=context.crop_context.crop_stage
        )
        logger.info(
            "What-to-do irrigation | %s",
            irr_result["decision"].get("action") if irr_result else "not available",
        )

        # ------------------------------------------------------------------
        # Step 3: Weather to-dos (deterministic, no LLM)
        # ------------------------------------------------------------------
        weather_todos = build_weather_todos_from_request(request)

        # ------------------------------------------------------------------
        # Step 4: Pest & Disease card (shared cache — runs independently of the
        #   AI Overview generation)
        # ------------------------------------------------------------------
        card = await _get_pest_card(request, context)

        # ------------------------------------------------------------------
        # Step 5: Deterministic selection (<= 2 items)
        # ------------------------------------------------------------------
        items = select_what_to_do(card, irr_result, weather_todos)

        # ------------------------------------------------------------------
        # Step 6: Translate selected items (title/hint) with dedup lock
        # ------------------------------------------------------------------
        is_english = request.language.lower().strip() in ("en", "english")
        if not is_english and items:
            trans_key = translation_cache.get_key(
                {"what_to_do": items}, request.language
            )
            trans_lock = await lock_manager.get_lock(trans_key)
            async with trans_lock:
                translated = translation_cache.get(trans_key)
                if translated:
                    logger.info("What-to-do Translation Cache HIT")
                    items = translated
                else:
                    logger.info("What-to-do Translation Cache MISS")
                    translated_items, was_translated = await translation_service.translate_what_to_do(
                        items, request.language
                    )
                    if was_translated:
                        items = translated_items
                        translation_cache.set(trans_key, items)
                        logger.info("What-to-do translation cache stored")
                    else:
                        logger.warning("What-to-do translation failed; returning English items.")

        response = {
            "success": True,
            "crop_id": request.cropId,
            "crop_name": context.crop_context.crop_name,
            "is_general": request.cropId.lower().strip() == "general",
            "language": request.language,
            "recommendations": items,
        }

        total = time.perf_counter() - t_start
        logger.info(f"What-to-do request complete in {total:.2f}s | {len(items)} item(s)")
        return response

    except Exception as e:
        logger.error(f"Unexpected what-to-do error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during what-to-do generation.")


def _build_pest_retrieval_context(context) -> Optional["RetrievalContext"]:
    """Deterministic weather/soil facts used to enrich the RAG query.

    Returns None when there is no forecast/soil context (e.g. a weather-only
    payload), in which case retrieval falls back to the plain crop+state+season
    query — the crop filter is what matters for crop-specificity.
    """
    if not (context.forecast_summary or context.soil_moisture_summary):
        return None
    sm_avg = (
        context.soil_moisture_summary.forecast_average_percentile
        if context.soil_moisture_summary else 50.0
    )
    sm_trend = (
        context.soil_moisture_summary.soil_moisture_trend
        if context.soil_moisture_summary else "unknown"
    )
    fs = context.forecast_summary
    return RetrievalContext(
        crop_stage=context.crop_context.crop_stage,
        rainfall_pattern=fs.rainfall_pattern if fs else "unknown",
        total_rainfall_mm=fs.total_rainfall_mm if fs else 0.0,
        min_temp_c=fs.minimum_temperature_c if fs else context.current_weather.temperature_c,
        max_temp_c=fs.maximum_temperature_c if fs else context.current_weather.temperature_c,
        soil_moisture_trend=sm_trend,
        soil_moisture_percentile_avg=sm_avg,
        lightning_category=(
            context.lightning_summary.category if context.lightning_summary else "unknown"
        ),
    )


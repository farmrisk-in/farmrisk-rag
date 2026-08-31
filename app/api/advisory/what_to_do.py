"""
What To Do Today Endpoint.
POST /api/advisory/what-to-do
Aggregates top daily actions across Pest & Disease, Irrigation, and IMD Weather To-Dos.
"""

import time
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.models.schemas import AIAdvisoryRequest
from app.services.irrigation import compute_irrigation_decision_and_insight
from app.services.what_to_do import build_weather_todos_from_request, select_what_to_do
from app.core.caching import translation_cache, lock_manager
from app.core.logging import logger
from app.api.advisory.dependencies import (
    context_builder,
    translation_service,
    get_or_create_pest_card,
)

router = APIRouter(prefix="/api/advisory", tags=["Advisory"])


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
        card = await get_or_create_pest_card(request, context)

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

"""
Pest & Disease Card Endpoint.
POST /api/advisory/pest-card
Builds the crop-specific "Pest & Disease" card with deterministic risk scoring,
RAG retrieval of ICAR advisory chunks, LLM phrasing, and translation.
"""

import time
from fastapi import APIRouter
from typing import Dict, Any

from app.models.schemas import AIAdvisoryRequest
from app.core.caching import translation_cache, lock_manager
from app.core.logging import logger
from app.api.advisory.dependencies import (
    context_builder,
    translation_service,
    get_or_create_pest_card,
)

router = APIRouter(prefix="/api/advisory", tags=["Advisory"])


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
    card = await get_or_create_pest_card(request, context)

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

"""
Pre-Sowing Advisory Endpoint.
POST /api/advisory/pre-sowing

Returns all 7 crop-calendar sections as Markdown strings in a single JSON response.
Each string can be directly parsed and rendered by the frontend Markdown component.

Sections:
  sowing_window    — Optimal dates, sowing type, and trigger conditions
  seed_selection   — Variety filters, seed rates, spacing, and refuge rules
  field_preparation — Operations, timing, and soil-specific details
  fertilizer_plan  — Baseline NPK, split schedule table, add-ons, drip fertigations
  irrigation       — Stage-wise schedule, criticality, waterlogging warnings
  weed_management  — Critical window, herbicide schedule, interculture ops
  pest_disease     — Stage-grouped pest/disease tables + spray discipline rules

Input: PresowingRequest (crop, state, soil_type, season, irrigation_type)
"""

import json
import time
from typing import Dict, Any, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.logging import logger
from app.core.config import settings
from app.llm.presowing_engine import PresowingEngine, PresowingGenerationError
from app.rag.retriever import AdvisoryRetriever

# Load canonical crop and state lists at startup for validation
import json as _json
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent.parent.parent
_CROPS: List[str] = _json.loads((_BASE / "config" / "crops.json").read_text())
_STATES: List[str] = _json.loads((_BASE / "config" / "states.json").read_text())

_CROPS_LOWER = {c.lower(): c for c in _CROPS}
_STATES_LOWER = {s.lower(): s for s in _STATES}

LANGUAGE_MAP: Dict[str, str] = {
    "en": "English",
    "english": "English",
    "hi": "Hindi",
    "hindi": "Hindi",
    "gu": "Gujarati",
    "gujarati": "Gujarati",
    "pa": "Punjabi",
    "punjabi": "Punjabi",
    "ta": "Tamil",
    "tamil": "Tamil",
    "te": "Telugu",
    "telugu": "Telugu",
    "mr": "Marathi",
    "marathi": "Marathi",
    "bn": "Bengali",
    "bengali": "Bengali",
    "kn": "Kannada",
    "kannada": "Kannada",
    "ml": "Malayalam",
    "malayalam": "Malayalam",
    "ur": "Urdu",
    "urdu": "Urdu",
    "or": "Odia",
    "odia": "Odia",
}

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/advisory", tags=["Pre-Sowing Advisory"])

# Singletons — initialized once at module load
_engine = PresowingEngine()
_retriever = AdvisoryRetriever()


# ---------------------------------------------------------------------------
# Input Schema
# ---------------------------------------------------------------------------

class PresowingRequest(BaseModel):
    """
    Input for the Pre-Sowing Advisory endpoint.

    crop           : Canonical crop name (must match config/crops.json, case-insensitive)
    state          : Indian state name (must match config/states.json, case-insensitive)
    soil_type      : Soil classification — drives fertilizer, irrigation, and field prep advice
    season         : Kharif | Rabi | Zaid (defaults to Kharif if not provided)
    irrigation_type: flood | drip | sprinkler | rainfed (defaults to flood)
    language       : Target language code ('en', 'hi', 'gu', 'pa', 'ta', 'te', 'mr') or name ('English', 'Hindi', 'Gujarati', etc.)
    """

    crop: str = Field(
        ...,
        min_length=2,
        max_length=60,
        description="Crop name — must match one of the canonical crops in config/crops.json",
        examples=["Cotton", "Wheat", "Rice"],
    )
    state: str = Field(
        ...,
        min_length=2,
        max_length=60,
        description="Indian state name — must match config/states.json",
        examples=["Gujarat", "Punjab", "Tamil Nadu"],
    )
    soil_type: Literal[
        "black cotton soil",
        "sandy loam",
        "clay",
        "loam",
        "laterite",
        "alluvial",
        "red soil",
        "saline",
        "alkaline",
    ] = Field(
        ...,
        description="Soil classification — adjusts fertilizer splits, irrigation counts, and field prep",
        examples=["black cotton soil"],
    )
    season: Literal["Kharif", "Rabi", "Zaid"] = Field(
        default="Kharif",
        description="Crop season",
    )
    irrigation_type: Literal["flood", "drip", "sprinkler", "rainfed"] = Field(
        default="flood",
        description="Primary irrigation method — adjusts fertilizer scheduling and water amounts",
    )
    language: str = Field(
        default="en",
        description="Target language code (e.g. 'en', 'hi', 'gu', 'pa', 'ta', 'te', 'mr') or language name ('English', 'Hindi', 'Gujarati', etc.)",
        examples=["en", "gu", "hi", "pa"],
    )


# ---------------------------------------------------------------------------
# Response Schema
# ---------------------------------------------------------------------------

class PresowingResponse(BaseModel):
    crop: str
    state: str
    season: str
    soil_type: str
    irrigation_type: str
    language: str
    generated_at: str
    rag_sources_used: int
    runtime_seconds: float
    sections: Dict[str, str] = Field(
        description="7 Markdown strings keyed by section name, ready for frontend rendering"
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/pre-sowing", response_model=PresowingResponse)
async def generate_presowing_advisory(request: PresowingRequest) -> PresowingResponse:
    """
    Generate all 7 pre-sowing advisory sections in one call.

    Returns a JSON object where each section value is a Markdown string
    that can be passed directly to your frontend Markdown parser/renderer.

    **Fallback behavior**: If no state-specific knowledge exists in the vector DB,
    automatically falls back to All-India ICAR data without any error.
    """
    t_start = time.perf_counter()

    # --- Normalize and validate crop/state against canonical lists ---
    crop_canonical = _CROPS_LOWER.get(request.crop.lower())
    if not crop_canonical:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Crop '{request.crop}' is not in the supported crop list. "
                f"Supported crops: {', '.join(_CROPS[:10])}... (see /docs for full list)"
            ),
        )

    state_canonical = _STATES_LOWER.get(request.state.lower())
    if not state_canonical:
        raise HTTPException(
            status_code=422,
            detail=(
                f"State '{request.state}' is not in the supported state list. "
                f"Supported states: {', '.join(_STATES[:8])}... (see /docs for full list)"
            ),
        )

    raw_lang = (request.language or "en").strip().lower()
    target_language = LANGUAGE_MAP.get(raw_lang, request.language.strip().title())

    logger.info(
        f"Presowing request | crop={crop_canonical} state={state_canonical} "
        f"soil={request.soil_type} season={request.season} irrig={request.irrigation_type} "
        f"lang={target_language}"
    )

    # --- RAG: Retrieve knowledge chunks for this crop+state combo ---
    # Fetch more chunks than usual (top 12) to cover all 7 sections
    rag_chunks = []
    try:
        rag_chunks = _retriever.retrieve(
            crop=crop_canonical,
            state=state_canonical,
            season=request.season,
            top_k=12,
        )

        # If we got fewer than 3 state-specific chunks, also retrieve All-India chunks
        if len(rag_chunks) < 3:
            india_chunks = _retriever.retrieve(
                crop=crop_canonical,
                state="All India",
                season=request.season,
                top_k=8,
            )
            # Merge, dedup by chunk id
            existing_ids = {c.get("id") for c in rag_chunks}
            for c in india_chunks:
                if c.get("id") not in existing_ids:
                    rag_chunks.append(c)

        logger.info(f"Presowing RAG: {len(rag_chunks)} chunks retrieved for {crop_canonical}/{state_canonical}")
    except Exception as e:
        logger.warning(f"Presowing RAG retrieval failed ({e}), proceeding without RAG chunks")

    # --- LLM: Generate all 7 sections in one call ---
    try:
        sections = await _engine.generate(
            crop=crop_canonical,
            state=state_canonical,
            soil_type=request.soil_type,
            season=request.season,
            irrigation_type=request.irrigation_type,
            target_language=target_language,
            rag_chunks=rag_chunks,
        )
    except PresowingGenerationError as e:
        logger.error(f"Presowing LLM generation failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

    runtime = round(time.perf_counter() - t_start, 3)
    from datetime import datetime, timezone

    return PresowingResponse(
        crop=crop_canonical,
        state=state_canonical,
        season=request.season,
        soil_type=request.soil_type,
        irrigation_type=request.irrigation_type,
        language=target_language,
        generated_at=datetime.now(timezone.utc).isoformat(),
        rag_sources_used=len(rag_chunks),
        runtime_seconds=runtime,
        sections={
            "sowing_window": sections.sowing_window,
            "seed_selection": sections.seed_selection,
            "field_preparation": sections.field_preparation,
            "fertilizer_plan": sections.fertilizer_plan,
            "irrigation": sections.irrigation,
            "weed_management": sections.weed_management,
            "pest_disease": sections.pest_disease,
        },
    )

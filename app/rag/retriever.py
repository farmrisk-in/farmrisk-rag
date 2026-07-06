"""
AdvisoryRetriever — weather-aware, 3-stage RAG retrieval from Supabase pgvector.

Strategy:
  Stage 1: crop + state + season          (strict local match)
  Stage 2: crop + season, state=None      (regional/national)
  Stage 3: crop only, state+season=None   (general crop guidance)

Deduplication by chunk id at each stage.
Stops when top_k unique chunks are collected.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from app.database.pgvector import PgVectorStore
from app.core.config import settings
from app.core.logging import logger


@dataclass
class RetrievalContext:
    """Compact weather/agronomic facts used to enrich the semantic query."""
    crop_stage: str
    rainfall_pattern: str
    total_rainfall_mm: float
    min_temp_c: float
    max_temp_c: float
    soil_moisture_trend: str
    soil_moisture_percentile_avg: float
    lightning_category: str


class AdvisoryRetriever:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.store = PgVectorStore()

    def retrieve(
        self,
        crop: str,
        state: str,
        season: Optional[str],
        retrieval_context: Optional[RetrievalContext] = None,
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve up to top_k unique ICAR advisory chunks via 3-stage fallback.
        Returns empty list if nothing found (caller must handle safe failure).
        """
        if top_k is None:
            top_k = settings.RAG_TOP_K

        query_text = self._build_query(crop, state, season, retrieval_context)
        logger.debug(f"RAG query: {query_text[:200]}")

        query_vector = self.model.encode(query_text, normalize_embeddings=True).tolist()

        collected: Dict[str, Dict[str, Any]] = {}  # id -> chunk

        # Stage 1: strict — crop + state + season
        stage1 = self._search(query_vector, crop, state, season, top_k)
        for chunk in stage1:
            cid = str(chunk.get("id", ""))
            if cid and cid not in collected:
                collected[cid] = self._normalise(chunk, stage=1)

        logger.info(f"RAG Stage 1 (strict): {len(stage1)} raw → {len(collected)} unique")

        # Stage 2: crop + season, no state filter
        if len(collected) < top_k:
            need = top_k - len(collected)
            stage2 = self._search(query_vector, crop, None, season, top_k)
            added = 0
            for chunk in stage2:
                cid = str(chunk.get("id", ""))
                if cid and cid not in collected:
                    collected[cid] = self._normalise(chunk, stage=2)
                    added += 1
                    if len(collected) >= top_k:
                        break
            logger.info(f"RAG Stage 2 (crop+season): +{added} unique")

        # Stage 3: crop only
        if len(collected) < top_k:
            stage3 = self._search(query_vector, crop, None, None, top_k)
            added = 0
            for chunk in stage3:
                cid = str(chunk.get("id", ""))
                if cid and cid not in collected:
                    collected[cid] = self._normalise(chunk, stage=3)
                    added += 1
                    if len(collected) >= top_k:
                        break
            logger.info(f"RAG Stage 3 (crop only): +{added} unique")

        results = list(collected.values())
        logger.info(
            f"RAG final: {len(results)} unique chunks | "
            f"IDs={[r['id'] for r in results]} | "
            f"scores={[round(r['score'] or 0, 3) for r in results]}"
        )
        return results

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    def _build_query(
        self,
        crop: str,
        state: str,
        season: Optional[str],
        ctx: Optional[RetrievalContext],
    ) -> str:
        season_str = season or "current"
        base = (
            f"Agricultural advisory recommendations for {crop} in {state} "
            f"during {season_str} season."
        )
        if ctx is None:
            return base

        details = (
            f" Crop stage: {ctx.crop_stage}."
            f" Expected 10-day conditions:"
            f" Total corrected rainfall: {ctx.total_rainfall_mm:.1f} mm."
            f" Rainfall pattern: {ctx.rainfall_pattern}."
            f" Temperature range: {ctx.min_temp_c:.1f} C to {ctx.max_temp_c:.1f} C."
            f" Soil moisture trend: {ctx.soil_moisture_trend}."
            f" Average soil moisture percentile: {ctx.soil_moisture_percentile_avg:.0f}."
            f" Lightning risk: {ctx.lightning_category}."
            f" Retrieve ICAR guidance relevant to current conditions"
            f" concerning irrigation, drainage, sowing, fertilizer timing,"
            f" pesticide spraying, pest/disease monitoring, and harvesting."
        )
        return base + details

    def _search(
        self,
        query_vector: List[float],
        crop: str,
        state: Optional[str],
        season: Optional[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        try:
            return self.store.similarity_search(
                query_vector,
                {"crop": crop, "state": state, "season": season},
                top_k,
            )
        except Exception as e:
            logger.error(f"Supabase similarity_search error: {e}")
            return []

    def _normalise(self, match: Dict[str, Any], stage: int) -> Dict[str, Any]:
        return {
            "id": match.get("id"),
            "score": match.get("similarity"),
            "content": match.get("content", ""),
            "state": match.get("state"),
            "crop": match.get("crop"),
            "season": match.get("season"),
            "page": match.get("page"),
            "source": match.get("source"),
            "retrieval_stage": stage,
        }

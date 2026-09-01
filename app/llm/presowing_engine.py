"""
presowing_engine.py — LLM engine for the Pre-Sowing Advisory endpoint.

Responsibilities:
  1. Fetch RAG chunks relevant to all 7 pre-sowing topics for the crop+state combo.
  2. Build the final prompt by injecting chunks + user inputs into the template.
  3. Call the LLM (Gemini primary, Groq fallback) for a structured JSON response.
  4. Validate and return PresowingResponse.

The actual prompt text lives in presowing_prompts.py — edit that file to change output.
"""

import json
from typing import List, Dict, Any
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import logger
from app.llm.providers import get_primary_provider, get_fallback_provider
from app.llm.presowing_prompts import PRESOWING_SYSTEM_PROMPT, PRESOWING_USER_TEMPLATE


# ---------------------------------------------------------------------------
# Response Schema — 7 Markdown strings
# ---------------------------------------------------------------------------

class PresowingSections(BaseModel):
    sowing_window: str
    seed_selection: str
    field_preparation: str
    fertilizer_plan: str
    irrigation: str
    weed_management: str
    pest_disease: str


class PresowingGenerationError(Exception):
    """Raised when all LLM providers fail."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PresowingEngine:
    """Generates all 7 pre-sowing advisory sections in a single LLM call."""

    def _build_prompt(
        self,
        crop: str,
        state: str,
        soil_type: str,
        season: str,
        irrigation_type: str,
        target_language: str,
        rag_chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Combine the RAG-retrieved knowledge chunks with the user template.
        The chunk content is injected between SYSTEM and USER sections so the
        LLM can ground each section in real agronomic data.
        """
        # Build the knowledge block from RAG chunks
        if rag_chunks:
            chunk_lines = []
            for i, chunk in enumerate(rag_chunks, 1):
                src = chunk.get("source", "ICAR")
                pg = chunk.get("page", "?")
                content = chunk.get("content", "").strip()
                chunk_lines.append(f"[{i}] ({src}, p.{pg}):\n{content}")
            knowledge_block = (
                "=== RETRIEVED KNOWLEDGE (from official Indian agronomy manuals) ===\n"
                + "\n\n".join(chunk_lines)
                + "\n=== END OF RETRIEVED KNOWLEDGE ===\n\n"
                "Use the above knowledge to ground your response. If a section topic is not "
                "covered in the retrieved chunks, use your training knowledge with ICAR fallback.\n\n"
            )
        else:
            knowledge_block = (
                "No pre-retrieved knowledge available. Use All-India ICAR guidelines.\n\n"
            )

        user_prompt = knowledge_block + PRESOWING_USER_TEMPLATE.format(
            crop=crop,
            state=state,
            soil_type=soil_type,
            season=season,
            irrigation_type=irrigation_type,
            target_language=target_language,
        )

        return user_prompt

    async def generate(
        self,
        crop: str,
        state: str,
        soil_type: str,
        season: str,
        irrigation_type: str,
        target_language: str,
        rag_chunks: List[Dict[str, Any]],
    ) -> PresowingSections:
        """
        Generate all 7 sections in one LLM call in the target language.
        Primary: Gemini | Fallback: Groq
        """
        prompt = self._build_prompt(
            crop=crop,
            state=state,
            soil_type=soil_type,
            season=season,
            irrigation_type=irrigation_type,
            target_language=target_language,
            rag_chunks=rag_chunks,
        )

        full_prompt = PRESOWING_SYSTEM_PROMPT + "\n\n" + prompt

        # Try primary provider (Gemini)
        primary = get_primary_provider()
        if primary:
            try:
                result = await primary.generate_json(
                    prompt=full_prompt,
                    schema=PresowingSections,
                    temperature=0.3,
                )
                logger.info(f"Presowing: Gemini generated all 7 sections for {crop}/{state}")
                return result
            except Exception as e:
                logger.warning(f"Presowing: Gemini failed ({e}), trying Groq fallback...")

        # Try fallback provider (Groq)
        fallback = get_fallback_provider()
        if fallback:
            try:
                result = await fallback.generate_json(
                    prompt=full_prompt,
                    schema=PresowingSections,
                    temperature=0.3,
                )
                logger.info(f"Presowing: Groq generated all 7 sections for {crop}/{state}")
                return result
            except Exception as e:
                logger.error(f"Presowing: Groq also failed ({e})")

        raise PresowingGenerationError(
            f"All LLM providers failed to generate presowing advisory for {crop}/{state}."
        )

import json
from typing import Dict, Any, List, Optional, Tuple
from app.core.config import settings
from app.models.schemas import AdvisoryResponse, TranslationResult
from pydantic import BaseModel

class TranslationResponse(BaseModel):
    translations: List[str]

from app.llm.providers import get_primary_provider, get_fallback_provider
from app.core.logging import logger

class TranslationService:
    def __init__(self):
        pass

    async def translate_advisory(self, advisory: AdvisoryResponse, target_language: str) -> TranslationResult:
        """Translate the values of the AdvisoryResponse into the target language."""
        if not target_language or target_language.lower() == "english":
            return TranslationResult(
                data=advisory.model_dump(),
                translated=True,
                provider=None
            )

        # Step 1: Extract and flatten text fields to translate
        texts_to_translate = [advisory.advisory_summary]
        
        # Step 2: Batch translate using providers
        translated_texts, provider = await self._batch_translate(texts_to_translate, target_language)
        
        if translated_texts is None or not translated_texts:
            # Translation failed completely
            logger.warning("Translation failed. Returning original English advisory.")
            return TranslationResult(
                data=advisory.model_dump(),
                translated=False,
                provider=None
            )
            
        # Step 3: Reconstruct the advisory dictionary
        try:
            translated_data = {
                "advisory_summary": translated_texts[0]
            }
            return TranslationResult(
                data=translated_data,
                translated=True,
                provider=provider
            )
        except Exception as e:
            logger.error(f"Error reconstructing translation data: {e}")
            return TranslationResult(
                data=advisory.model_dump(),
                translated=False,
                provider=None
            )


    async def _batch_translate(self, texts: List[str], target_language: str) -> Tuple[Optional[List[str]], Optional[str]]:
        # Check if settings allow actual provider runs
        # If no keys are set, fallback to mock translator
        has_gemini = bool(settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY != "your_google_api_key")
        has_groq = bool(settings.GROQ_API_KEY)
        
        if not has_gemini and not has_groq:
            # Mock translator: just appends language suffix
            logger.info("Running translation in mock mode (no provider keys configured)")
            mocked = [f"{text} [{target_language}]" for text in texts]
            return mocked, None
            
        prompt = f"""
You are a precise translator. Translate the following list of strings from English into {target_language}.

RULES:
1. Maintain the exact order and number of elements in the list.
2. Return a JSON object with a key "translations" containing the array of translated strings of the exact same length.
3. Translate the meaning accurately. Do not summarize, rephrase, rewrite, or add any formatting.
4. Keep technical agricultural terms accurate in the target language.
5. Crucially, preserve paragraph separation (e.g. double newlines), exact wording, meaning, and sentence order. Do not regenerate the advisory, do not summarize, and do not rewrite.


Input list:
{json.dumps(texts, ensure_ascii=False)}
"""

        primary_provider = get_primary_provider()
        fallback_provider = get_fallback_provider()
        
        # Try primary provider
        try:
            result = await primary_provider.generate_json(
                prompt=prompt,
                schema=TranslationResponse,
                temperature=0.1
            )
            logger.info(f"{settings.LLM_PROVIDER} translation succeeded")
            return result.translations, settings.LLM_PROVIDER.lower()
        except Exception as primary_exc:
            logger.warning(f"Primary provider {settings.LLM_PROVIDER} failed translation: {primary_exc}")
            
            # Switch to Groq fallback
            if fallback_provider:
                logger.info("Switching to Groq fallback for translation")
                try:
                    result = await fallback_provider.generate_json(
                        prompt=prompt,
                        schema=TranslationResponse,
                        temperature=0.1
                    )
                    logger.info("Groq translation succeeded")
                    return result.translations, "groq"
                except Exception as fallback_exc:
                    logger.error(f"Fallback provider Groq failed translation: {fallback_exc}")
                    
        return None, None

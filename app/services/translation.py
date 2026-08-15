import json
import re
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
        if not target_language or target_language.lower().strip() in ("en", "english"):
            return TranslationResult(
                data=advisory.model_dump(),
                translated=True,
                provider=None
            )

        # Step 1: Extract and flatten text fields to translate
        texts_to_translate = [advisory.advisory_summary]
        has_insight = bool(advisory.irrigation_insight)
        if has_insight:
            texts_to_translate.append(advisory.irrigation_insight)
        
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
            if has_insight and len(translated_texts) > 1:
                translated_data["irrigation_insight"] = translated_texts[1]
            if advisory.sources is not None:
                translated_data["sources"] = advisory.sources
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


    async def translate_pest_disease_card(self, card: Dict[str, Any], target_language: str) -> TranslationResult:
        """Translate the human-readable fields of a Pest & Disease card dict.

        Only the display strings are translated (risk band, summary, crop
        identity, potential pest/disease names, action titles/details).
        Structured data (score, driver, sources, crop_id, is_general, internal
        warnings) is preserved as-is, matching the advisory flow.

        Highlight terms (crop name/stage/season, potential names) are embedded
        into the summary/details as @@N@@ placeholders *before* translation so
        the translated text keeps them verbatim; the standalone translated term
        is then substituted back in. This guarantees the exact translated term
        the frontend highlights appears inside the translated sentence, despite
        the model otherwise inflecting it (e.g. ફૂગના vs ફૂગનો).
        """
        if not target_language or target_language.lower().strip() in ("en", "english"):
            return TranslationResult(data=card, translated=True, provider=None)

        # Step 1: Ordered list of highlight terms (must match what the frontend
        #   highlights: crop_name, crop_stage, season, then potential items).
        terms: List[str] = []
        for key in ("crop_name", "crop_stage", "season"):
            val = card.get(key)
            if val:
                terms.append(str(val))
        potential = [str(p) for p in (card.get("potential") or []) if str(p).strip()]
        terms += potential

        actions = card.get("actions") or []
        action_dicts = [a for a in actions if isinstance(a, dict)]

        # Step 2: Embed @@N@@ placeholders into summary and action details.
        summary_embedded = self._embed_highlight_terms(str(card.get("summary", "")), terms)
        detail_embedded = [
            self._embed_highlight_terms(str(a.get("detail", "")), terms)
            for a in action_dicts
        ]
        titles = [str(a.get("title", "")) for a in action_dicts]

        # Step 3: Flatten the texts to translate (stable order).
        texts_to_translate: List[str] = [str(card.get("risk", "")), summary_embedded]
        texts_to_translate += terms
        texts_to_translate += titles
        texts_to_translate += detail_embedded

        # Step 4: Batch translate using providers
        translated_texts, provider = await self._batch_translate(texts_to_translate, target_language)
        if translated_texts is None or len(translated_texts) != len(texts_to_translate):
            logger.warning("Pest & Disease card translation failed. Returning original English card.")
            return TranslationResult(data=card, translated=False, provider=None)

        # Step 5: Reconstruct the card dict, replacing only the text fields.
        try:
            idx = 0
            trans_risk = translated_texts[idx]; idx += 1
            trans_summary = translated_texts[idx]; idx += 1
            trans_terms = translated_texts[idx:idx + len(terms)]; idx += len(terms)
            trans_titles = translated_texts[idx:idx + len(titles)]; idx += len(titles)
            trans_details = translated_texts[idx:idx + len(detail_embedded)]; idx += len(detail_embedded)

            def substitute(text: str) -> str:
                for i, term in enumerate(terms):
                    text = text.replace(f"@@{i}@@", trans_terms[i])
                return text

            translated_data = dict(card)
            translated_data["risk"] = trans_risk
            translated_data["summary"] = substitute(trans_summary)

            pos = 0
            for key in ("crop_name", "crop_stage", "season"):
                if card.get(key):
                    translated_data[key] = trans_terms[pos]; pos += 1
            translated_data["potential"] = trans_terms[pos:pos + len(potential)]

            new_actions = []
            ai = 0
            for a in actions:
                if not isinstance(a, dict):
                    new_actions.append(a)
                    continue
                na = dict(a)
                na["title"] = trans_titles[ai]
                na["detail"] = substitute(trans_details[ai])
                ai += 1
                new_actions.append(na)
            translated_data["actions"] = new_actions
            return TranslationResult(data=translated_data, translated=True, provider=provider)
        except Exception as e:
            logger.error(f"Error reconstructing pest card translation: {e}")
            return TranslationResult(data=card, translated=False, provider=None)

    def _embed_highlight_terms(self, text: str, terms: List[str]) -> str:
        """Replace English highlight terms with @@N@@ placeholders.

        Longest terms match first and matching is case-insensitive and
        whole-word (ASCII), so an inflected English form in the summary still
        maps to its placeholder. The frontend highlights these exact translated
        strings, so preserving them verbatim is what keeps highlighting working
        in every language.
        """
        if not text or not terms:
            return text
        ordered = sorted(terms, key=len, reverse=True)
        pattern = (
            r"(?<![A-Za-z0-9_])(?:"
            + "|".join(re.escape(t) for t in ordered)
            + r")(?![A-Za-z0-9_])"
        )

        def repl(match: re.Match) -> str:
            word = match.group(0)
            for i, term in enumerate(terms):
                if term.lower() == word.lower():
                    return f"@@{i}@@"
            return word

        return re.sub(pattern, repl, text, flags=re.IGNORECASE)


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
6. Preserve every placeholder token of the form @@N@@ (where N is a number) EXACTLY as-is — same position, same characters, inside the translated string. Never translate, drop, reorder, split, or add spaces around them. They are markers the caller substitutes afterwards.


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

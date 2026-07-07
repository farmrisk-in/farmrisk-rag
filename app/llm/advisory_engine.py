"""
AdvisoryEngine — generates the canonical English AI Advisory Summary.

Input:  AdvisoryContext (compact deterministic facts) + ICAR RAG chunks
Output: AdvisoryResponse (plain-text, two paragraphs, 120-180 words)

Rules:
- Gemini receives structured text facts, not raw JSON arrays.
- All arithmetic is already done in ContextBuilder.
- RAG chunks MUST exist — raises InsufficientKnowledgeError otherwise.
- Output is validated; retries preserve the full original base prompt.
- _get_mock_advisory() is available in dev mode only.
"""

import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import AdvisoryContext, AdvisoryResponse
from app.llm.providers import get_primary_provider, get_fallback_provider


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------

class InsufficientKnowledgeError(Exception):
    """Raised when RAG returned no chunks — prevents hallucinated advisory."""

class AdvisoryGenerationError(Exception):
    """Raised when all LLM providers fail after all retries."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AdvisoryEngine:
    def __init__(self):
        pass

    async def generate_advisory(
        self,
        context: AdvisoryContext,
        rag_chunks: List[Dict[str, Any]],
    ) -> AdvisoryResponse:
        """
        Generate a two-paragraph plain-text advisory.

        If rag_chunks is empty, generation continues with a weather-only prompt
        (no ICAR recommendations invented). This avoids a hard 503 failure for
        general/unknown crops where the RAG index has no matching chunks.

        Raises:
            AdvisoryGenerationError: if all LLM providers fail on every attempt.
        """
        if not rag_chunks:
            logger.warning(
                "No ICAR RAG chunks retrieved. Generating weather-only advisory "
                "without crop-specific ICAR recommendations."
            )

        # Build the base prompt once — preserved for all retry correction prompts
        base_prompt = self._build_prompt(context, rag_chunks)
        prompt = base_prompt

        primary = get_primary_provider()
        fallback = get_fallback_provider()

        for attempt in range(1, 4):
            # Try primary provider
            raw = None
            try:
                raw = await primary.generate_text(prompt=prompt, temperature=settings.TEMPERATURE)
            except Exception as e:
                logger.warning(f"Primary provider attempt {attempt} failed: {e}")

            # If primary failed, try fallback
            if raw is None and fallback:
                try:
                    raw = await fallback.generate_text(prompt=prompt, temperature=settings.TEMPERATURE)
                    logger.info(f"Fallback provider succeeded on attempt {attempt}")
                except Exception as fe:
                    logger.error(f"Fallback provider attempt {attempt} failed: {fe}")

            if raw is None:
                logger.warning(f"Both providers failed on attempt {attempt}. No raw output.")
                continue

            errors = self._validate(raw, context)
            if not errors:
                logger.info(f"Advisory validated on attempt {attempt}.")
                return AdvisoryResponse(advisory_summary=raw.strip())

            logger.warning(f"Advisory attempt {attempt} failed validation: {errors}")

            # Correction prompt preserves the original base prompt (never loses context)
            prompt = self._build_correction_prompt(
                base_prompt=base_prompt,
                previous=raw,
                errors=errors,
            )

        logger.error("All advisory generation attempts exhausted. Raising AdvisoryGenerationError.")
        raise AdvisoryGenerationError(
            "Advisory generation failed: all LLM provider attempts were exhausted."
        )

    # ------------------------------------------------------------------
    # PROMPT CONSTRUCTION
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        ctx: AdvisoryContext,
        rag_chunks: List[Dict[str, Any]],
    ) -> str:
        loc = ctx.location
        crop = ctx.crop_context
        cw = ctx.current_weather
        av = ctx.availability

        # ---- Forecast section ----
        wa = ctx.weather_api_summary
        if ctx.forecast_summary:
            fs = ctx.forecast_summary
            start_fmt = self._fmt_date(fs.forecast_start_date)
            end_fmt = self._fmt_date(fs.forecast_end_date)
            forecast_section = (
                f"CORRECTED FORECAST SUMMARY (Primary — Bias-Corrected Model)\n"
                f"Forecast period  : {start_fmt} to {end_fmt} ({fs.forecast_days} days)\n"
                f"Total rainfall   : {fs.total_rainfall_mm:.1f} mm\n"
                f"Rainy days       : {fs.rainy_days} of {fs.forecast_days}\n"
                f"Max daily rain   : {fs.maximum_daily_rainfall_mm:.1f} mm\n"
                f"Rainfall pattern : {fs.rainfall_pattern}\n"
                f"Temp range       : {fs.minimum_temperature_c:.1f} °C to {fs.maximum_temperature_c:.1f} °C\n"
                f"Avg min/max temp : {fs.average_min_temperature_c:.1f} °C / {fs.average_max_temperature_c:.1f} °C"
            )
        else:
            # Fall back to weather API daily dates for the "From ... to ..." prefix
            start_fmt = self._fmt_date(wa.api_start_date) if wa.api_start_date else ""
            end_fmt = self._fmt_date(wa.api_end_date) if wa.api_end_date else ""
            forecast_section = "CORRECTED FORECAST SUMMARY: Not available."

        # ---- Weather API secondary ----
        api_section = (
            f"WEATHER API DAILY ({wa.api_start_date} to {wa.api_end_date})\n"
            f"Total rainfall: {wa.api_total_rainfall_mm:.1f} mm | "
            f"Temp range: {wa.api_min_temp_c:.1f} °C – {wa.api_max_temp_c:.1f} °C"
        )

        # ---- Soil moisture section — explicit start/end/min/max ----
        if ctx.soil_moisture_summary and av.soil_moisture_available:
            sm = ctx.soil_moisture_summary
            sm_section = (
                f"SOIL MOISTURE SUMMARY\n"
                f"Latest known percentile       : {sm.latest_sm_percentile:.1f}\n"
                f"Forecast start percentile     : {sm.start_percentile:.1f}\n"
                f"Forecast end percentile       : {sm.end_percentile:.1f}\n"
                f"Forecast minimum percentile   : {sm.forecast_min_percentile:.1f}\n"
                f"Forecast maximum percentile   : {sm.forecast_max_percentile:.1f}\n"
                f"Forecast average percentile   : {sm.forecast_average_percentile:.1f}\n"
                f"Forecast w_frac range         : {sm.forecast_min_w_frac:.3f} to {sm.forecast_max_w_frac:.3f}\n"
                f"Overall forecast trend        : {sm.soil_moisture_trend}\n"
                f"\n"
                f"IMPORTANT SOIL MOISTURE INTERPRETATION RULES:\n"
                f"- Forecast start percentile ({sm.start_percentile:.1f}) and forecast end percentile "
                f"({sm.end_percentile:.1f}) describe the change over the forecast period.\n"
                f"- Forecast minimum ({sm.forecast_min_percentile:.1f}) and maximum "
                f"({sm.forecast_max_percentile:.1f}) describe only the extremes reached during the period.\n"
                f"- NEVER describe minimum/maximum values as the starting and ending values.\n"
                f"- If describing the soil moisture trend numerically, use ONLY:\n"
                f"  {sm.start_percentile:.1f} → {sm.end_percentile:.1f}\n"
                f"- The maximum percentile ({sm.forecast_max_percentile:.1f}) may be mentioned separately "
                f"as the peak value reached during the period.\n"
                f"- Do NOT describe soil as saturated, waterlogged, drought-stressed, critically dry, "
                f"or use any other categorical soil condition — the AdvisoryContext does not provide "
                f"a deterministic classification.\n"
                f"- Do NOT infer soil-condition categories from percentile or w_frac values."
            )
        else:
            sm_section = "SOIL MOISTURE: No data available. Do NOT mention soil moisture at all in the advisory."

        # ---- Lightning section ----
        if ctx.lightning_summary and av.lightning_available:
            lg = ctx.lightning_summary
            lg_section = f"LIGHTNING RISK\nScore: {lg.score:.1f} | Category: {lg.category}"
        else:
            lg_section = "LIGHTNING RISK: Not available."

        # ---- Availability flags ----
        avail_section = (
            f"SOURCE AVAILABILITY\n"
            f"Corrected forecast : {'Yes' if av.corrected_forecast_available else 'No'}\n"
            f"Soil moisture      : {'Yes' if av.soil_moisture_available else 'No'}\n"
            f"Lightning          : {'Yes' if av.lightning_available else 'No'}\n"
            f"Crop calendar      : {'Yes' if av.calendar_available else 'No'}"
        )

        # ---- RAG chunks ----
        rag_lines = []
        if rag_chunks:
            for i, chunk in enumerate(rag_chunks, 1):
                rag_lines.append(
                    f"[Chunk {i}]\n"
                    f"Source    : {chunk.get('source', 'ICAR')}\n"
                    f"Page      : {chunk.get('page', 'N/A')}\n"
                    f"Crop      : {chunk.get('crop', 'N/A')}\n"
                    f"Season    : {chunk.get('season', 'N/A')}\n"
                    f"Similarity: {round(chunk.get('score') or 0, 3)}\n"
                    f"Content   :\n{chunk.get('content', '')}\n"
                )
            rag_section = "\n".join(rag_lines)
            rag_instruction = (
                "Provide practical recommendations based ONLY on retrieved ICAR knowledge and forecast conditions.\n"
                "- Cover where relevant: irrigation, sowing/transplanting, fertilizer, pesticide, drainage, pest/disease, harvesting.\n"
                "- Do NOT invent recommendations not supported by retrieved ICAR context."
            )
        else:
            rag_section = "[No ICAR knowledge retrieved]"
            rag_instruction = (
                "No crop-specific ICAR data is available. "
                "Generate the advisory based solely on the observed weather and forecast data above. "
                "Do NOT mention ICAR, do NOT mention that knowledge is unavailable, do NOT mention RAG. "
                "Do NOT invent any pesticide names, fertilizer rates, or specific sowing/harvesting dates. "
                "Instead, provide practical general weather-based guidance for farmers and end with the outlook sentence."
            )

        prompt = f"""You are an expert agrometeorologist at FarmRisk. Generate a professional 10-Day Crop-Specific Advisory Summary.

=== LOCATION ===
Village     : {loc.name}
District    : {loc.district}
State       : {loc.state}
Coordinates : {loc.lat:.4f} N, {loc.lng:.4f} E

=== CROP CONTEXT ===
Crop        : {crop.crop_name}
Season      : {crop.season}
Crop stage  : {crop.crop_stage}

=== CURRENT WEATHER (Observed) ===
Time        : {cw.observation_time}
Temperature : {cw.temperature_c:.1f} °C (feels like {cw.apparent_temperature_c:.1f} °C)
Humidity    : {cw.relative_humidity_percent:.0f}%
Precipitation: {cw.precipitation_mm:.1f} mm
Wind        : {cw.wind_speed_kmh:.1f} km/h (gusts {cw.wind_gusts_kmh:.1f} km/h)
Cloud cover : {cw.cloud_cover_percent:.0f}%
Condition   : {cw.weather_condition}

=== {forecast_section} ===

=== {api_section} ===

=== {sm_section} ===

=== {lg_section} ===

=== {avail_section} ===

=== RETRIEVED ICAR KNOWLEDGE ===
{rag_section}

=== OUTPUT INSTRUCTIONS ===
Generate exactly THREE paragraphs of plain text, separated by a blank line.
Total word count MUST be between {settings.ADVISORY_MIN_WORDS} and {settings.ADVISORY_MAX_WORDS} words. Write in full, detailed sentences — do NOT be brief.
Use Indian date format DD/MM/YYYY for any dates.
Highlight important words/numbers with single asterisks: *word* or *number*.

FORBIDDEN PHRASES — never use these:
  "Based on the retrieved ICAR knowledge"
  "Based on ICAR knowledge"
  "According to ICAR"
  "ICAR recommends"
  "No ICAR"
  "RAG"
  "knowledge base"
  "retrieved knowledge"
  "not available"
  "unavailable"

NOT ALLOWED — reject these formatting elements:
  Markdown headings (#, ##, ###)
  Bullet lists (-, +, •)
  Numbered lists (1., 2., 3.)
  Code blocks (```)
  Tables (|---|)
  Double-asterisk bold (**word**)
  Double-underscore bold (__word__)

ALLOWED single-asterisk inline emphasis examples:
  *Groundnut*    *131.1 mm*    *24.1 °C*    *moderate*    *increasing*

PARAGRAPH 1 — Detailed Weather & Climate Analysis (70–120 words):
- MUST begin exactly with: "From {start_fmt} to {end_fmt}, "
- Cover ALL of the following in detail:
  • Corrected total rainfall, rainy day count, max single-day rainfall, and rainfall pattern classification
  • Temperature range (min/max), average temperatures, and heat stress implications for crops
  • Current observed conditions: humidity, wind speed, wind gusts, cloud cover, precipitation, weather condition
  • Lightning risk score and category with agricultural implication
  • Soil moisture trend (start → end percentile) only if data is available — otherwise skip silently
- Write in flowing sentences, not a list.
- Do NOT include any recommendations or instructions in Paragraph 1.

PARAGRAPH 2 — Crop-Specific Advisory & Field Management (80–140 words):
- {rag_instruction}
- Cover ALL relevant aspects: sowing/transplanting timing, irrigation scheduling, fertilizer application (type, rate, timing), pest and disease watch, drainage management, field preparation.
- Mention the specific crop stage and tailor recommendations to current growth phase.
- Reference specific weather risks (excess rain, heat stress, wind) and how farmers should respond.
- Write detailed, actionable sentences — do NOT be vague.
- Do NOT make claims about data sources.

PARAGRAPH 3 — Outlook & Risk Summary (40–60 words):
- Summarise the overall risk level for the coming period.
- Mention the single biggest weather risk farmers should watch.
- Advise farmers on monitoring frequency or key action to take this week.
- MUST end with exactly one of:
  "Overall, the agricultural outlook for this period is Favorable."
  "Overall, the agricultural outlook for this period is Cautionary."
  "Overall, the agricultural outlook for this period is Unfavorable."

HALLUCINATION RULES:
- Use ONLY the numerical values provided above. Do not invent any values.
- Do not invent crop stage, pesticide names, fertilizer rates, or sowing/harvesting dates not present in the knowledge chunks.
- If soil moisture data is absent, skip it entirely — do not say it is unavailable.
"""
        return prompt

    def _build_correction_prompt(
        self,
        base_prompt: str,
        previous: str,
        errors: List[str],
    ) -> str:
        """
        Build a correction prompt that preserves the complete original factual context.
        The previous invalid output is shown so the model can diff against it,
        but the model is explicitly told to regenerate from the original facts — not
        to copy unsupported statements from the invalid output.
        """
        error_list = "\n".join(f"  - {e}" for e in errors)
        return (
            f"=== ORIGINAL FACTUAL CONTEXT AND RETRIEVED ICAR KNOWLEDGE ===\n"
            f"{base_prompt}\n\n"
            f"=== PREVIOUS INVALID OUTPUT (DO NOT COPY) ===\n"
            f"{previous}\n\n"
            f"=== VALIDATION ERRORS THAT MUST BE CORRECTED ===\n"
            f"{error_list}\n\n"
            f"=== CORRECTION INSTRUCTIONS ===\n"
            f"Regenerate the advisory from the ORIGINAL FACTUAL CONTEXT above.\n"
            f"Correct every validation error listed above.\n"
            f"Do NOT copy unsupported statements from the previous invalid output.\n"
            f"All numerical claims must remain grounded in the AdvisoryContext values above.\n"
            f"Agricultural recommendations must remain grounded in the retrieved ICAR chunks above.\n"
            f"Return ONLY the corrected advisory — no explanation, no preamble.\n"
        )

    # ------------------------------------------------------------------
    # OUTPUT VALIDATION
    # ------------------------------------------------------------------

    def _validate(self, text: str, context: AdvisoryContext) -> List[str]:
        errors = []
        text = text.strip()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # ---- Structure: exactly 3 paragraphs ----
        if len(paragraphs) != 3:
            errors.append(
                f"Expected exactly 3 paragraphs separated by a blank line, got {len(paragraphs)}."
            )

        # ---- Word count ----
        word_count = len(text.split())
        if word_count < settings.ADVISORY_MIN_WORDS:
            errors.append(f"Too short: {word_count} words (min {settings.ADVISORY_MIN_WORDS}).")
        if word_count > settings.ADVISORY_MAX_WORDS:
            errors.append(f"Too long: {word_count} words (max {settings.ADVISORY_MAX_WORDS}).")

        # ---- Paragraph 1 date prefix ----
        if paragraphs:
            if context.forecast_summary:
                start_dd = self._fmt_date(context.forecast_summary.forecast_start_date)
                end_dd = self._fmt_date(context.forecast_summary.forecast_end_date)
            elif context.weather_api_summary.api_start_date:
                start_dd = self._fmt_date(context.weather_api_summary.api_start_date)
                end_dd = self._fmt_date(context.weather_api_summary.api_end_date)
            else:
                start_dd = end_dd = None

            if start_dd and not paragraphs[0].startswith(f"From {start_dd}"):
                errors.append(
                    f"Paragraph 1 must start with 'From {start_dd} to {end_dd},'. "
                )

        # ---- Last paragraph (Paragraph 3) must end with outlook sentence ----
        valid_endings = [
            "Overall, the agricultural outlook for this period is Favorable.",
            "Overall, the agricultural outlook for this period is Cautionary.",
            "Overall, the agricultural outlook for this period is Unfavorable.",
        ]
        if paragraphs:
            if not any(paragraphs[-1].endswith(e) for e in valid_endings):
                errors.append("Last paragraph must end with a valid outlook sentence.")

        # ---- Paragraph 1 must not contain recommendation language ----
        if paragraphs:
            p1_lower = paragraphs[0].lower()
            forbidden_in_p1 = [
                "farmers should", "farmers must",
                "apply fertilizer", "apply pesticide", "apply fungicide", "apply insecticide",
                "spray pesticide", "spray fungicide",
                "sow now", "harvest now",
                "irrigate", "ensure drainage",
                "use fungicide", "use insecticide",
            ]
            for phrase in forbidden_in_p1:
                if phrase in p1_lower:
                    errors.append(
                        f"Paragraph 1 contains recommendation language ('{phrase}'). "
                        f"Recommendations must appear only in Paragraph 2."
                    )
                    break

        # ---- Structural markdown detection ----
        for line in text.split("\n"):
            stripped = line.lstrip()
            # Headings
            if stripped.startswith("#"):
                errors.append("Response must not contain markdown headings (lines starting with #).")
                break
        for line in text.split("\n"):
            stripped = line.lstrip()
            # Bullet lists
            if re.match(r'^[-+•]\s', stripped):
                errors.append("Response must not contain markdown bullet lists (-, +, •).")
                break
        for line in text.split("\n"):
            stripped = line.lstrip()
            # Numbered lists
            if re.match(r'^\d+\.\s', stripped):
                errors.append("Response must not contain numbered lists (1., 2., ...).")
                break
        # Code fences
        if "```" in text:
            errors.append("Response must not contain code fences (```).")
        # Tables
        if re.search(r'\|\s*[-:]+\s*\|', text):
            errors.append("Response must not contain markdown tables.")
        # Double-asterisk bold (but allow single-asterisk emphasis)
        if "**" in text:
            errors.append("Response must not contain double-asterisk bold (**word**).")
        if "__" in text:
            errors.append("Response must not contain double-underscore bold (__word__).")

        # ---- Soil moisture semantic grounding ----
        if context.soil_moisture_summary and context.availability.soil_moisture_available:
            sm_errors = self._validate_soil_moisture_semantics(text, context.soil_moisture_summary)
            errors.extend(sm_errors)

        return errors

    def _validate_soil_moisture_semantics(
        self,
        text: str,
        sm,
    ) -> List[str]:
        """
        Detect when the model incorrectly presents forecast_min or forecast_max as temporal
        start/end values in the trend description.

        Valid:
          "from 58 to 82"    (start → end)
          "ranging from 58 to 88"   (range description — OK)
          "minimum of 58 and maximum of 88"  (extremes — OK)
          "reaching a maximum of 88"  (peak — OK)

        Invalid:
          "increased from 58 to 88"  (temporal — max used as end)
          "from 58 to 88" in temporal language context
        """
        errors = []
        start = sm.start_percentile
        end = sm.end_percentile
        mn = sm.forecast_min_percentile
        mx = sm.forecast_max_percentile

        # Only flag if min/max are actually different from start/end
        # (no false positives when they coincide)
        if abs(mx - end) < 0.5 and abs(mn - start) < 0.5:
            return []  # min==start and max==end — no ambiguity possible

        # Temporal trigger words that indicate a start→end change claim
        temporal_triggers = [
            r'increas(?:es?|ing|ed)\s+from',
            r'ros(?:e|es|ing)\s+from',
            r'decreas(?:es?|ing|ed)\s+from',
            r'declin(?:es?|ing|ed)\s+from',
            r'fell?\s+from',
            r'drop(?:s|ped|ping)?\s+from',
            r'chang(?:es?|ing|ed)\s+from',
            r'(?:goes?|went|going)\s+from',
            r'trend(?:s|ing|ed)?\s+from',
            r'mov(?:es?|ing|ed)\s+from',
        ]

        # Normalise text for number matching — strip asterisks, strip "percentile"
        normalised = text.lower()
        normalised = re.sub(r'\*([0-9.]+)\*', r'\1', normalised)      # *58* → 58
        normalised = re.sub(r'\bpercentile\b', '', normalised)          # remove the word
        normalised = re.sub(r'\s+', ' ', normalised)

        def fmt_variants(val: float) -> List[str]:
            """All expected text forms for a percentile value."""
            rounded = round(val, 1)
            integer = int(val) if val == int(val) else None
            forms = [f"{rounded}", f"{rounded:.0f}"]
            if integer is not None:
                forms.append(str(integer))
            return list(set(forms))

        mn_variants = fmt_variants(mn)
        mx_variants = fmt_variants(mx)
        start_variants = fmt_variants(start)
        end_variants = fmt_variants(end)

        def number_follows(text_segment: str, variants: List[str]) -> Optional[str]:
            """Check if any variant appears shortly after a position in text."""
            for v in variants:
                pattern = rf'\b{re.escape(v)}\b'
                if re.search(pattern, text_segment):
                    return v
            return None

        for trigger_pattern in temporal_triggers:
            for m in re.finditer(trigger_pattern, normalised):
                after = normalised[m.start(): m.start() + 80]

                # Find what number appears as start in this phrase
                found_start = number_follows(after, mn_variants + start_variants)
                if not found_start:
                    continue

                # Find what number appears as end after the "to"
                to_match = re.search(r'\bto\b', after)
                if not to_match:
                    continue
                after_to = after[to_match.end():]
                found_end = number_follows(after_to, mx_variants + end_variants)
                if not found_end:
                    continue

                # Check if this is a min→max claim (bad) vs start→end claim (good)
                start_is_min = found_start in mn_variants and found_start not in end_variants
                end_is_max = found_end in mx_variants and found_end not in end_variants

                if start_is_min and end_is_max:
                    errors.append(
                        f"Soil moisture trend incorrectly describes forecast minimum ({mn:.1f}) "
                        f"to maximum ({mx:.1f}) as a temporal start→end change. "
                        f"Use start percentile ({start:.1f}) → end percentile ({end:.1f}) instead."
                    )
                    return errors  # one error is enough

        return errors

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_date(date_str: str) -> str:
        """Convert any date string (YYYY-MM-DD or ISO datetime) to DD/MM/YYYY."""
        if not date_str:
            return ""
        # Normalise: take first 10 chars to handle ISO like 2026-07-07T00:00:00.000Z
        clean = date_str[:10]
        try:
            dt = datetime.strptime(clean, "%Y-%m-%d")
            return dt.strftime("%d/%m/%Y")
        except Exception:
            return clean

    def _get_mock_advisory(self, context: AdvisoryContext) -> AdvisoryResponse:
        """
        Development/testing fallback ONLY.
        Never called in production (production path raises AdvisoryGenerationError instead).
        Contains NO agricultural recommendations.
        """
        if not getattr(settings, "ENABLE_MOCK_ADVISORY", False):
            raise AdvisoryGenerationError(
                "Mock advisory is disabled in production. "
                "Enable ENABLE_MOCK_ADVISORY=true only for local development."
            )

        loc = context.location
        crop = context.crop_context.crop_name
        if context.forecast_summary:
            start = self._fmt_date(context.forecast_summary.forecast_start_date)
            end = self._fmt_date(context.forecast_summary.forecast_end_date)
            rain = context.forecast_summary.total_rainfall_mm
            tmin = context.forecast_summary.minimum_temperature_c
            tmax = context.forecast_summary.maximum_temperature_c
        else:
            from datetime import timedelta
            today = datetime.now()
            start = today.strftime("%d/%m/%Y")
            end = (today + timedelta(days=9)).strftime("%d/%m/%Y")
            rain, tmin, tmax = 25.0, 23.0, 36.0

        p1 = (
            f"From {start} to {end}, the region of {loc.name}, {loc.district}, {loc.state} "
            f"is expected to experience a cumulative rainfall of *{rain:.0f} mm*, "
            f"with temperature ranging from *{tmin:.0f} °C* to *{tmax:.0f} °C*. "
            f"These conditions are broadly characteristic of the current agricultural season "
            f"and will have a moderate impact on {crop} crops in the field."
        )
        # Paragraph 2 contains no agricultural recommendations — only outlook
        p2 = (
            f"No ICAR knowledge was retrieved for {crop} in this configuration. "
            f"Farmers are advised to consult their local Krishi Vigyan Kendra (KVK) "
            f"for crop-specific guidance applicable to current conditions. "
            f"Overall, the agricultural outlook for this period is Cautionary."
        )
        return AdvisoryResponse(advisory_summary=f"{p1}\n\n{p2}")

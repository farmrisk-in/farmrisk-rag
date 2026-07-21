"""
AdvisoryEngine — generates the canonical English AI Advisory Summary.

Input:  AdvisoryContext (compact deterministic facts) + ICAR RAG chunks
Output: AdvisoryResponse (plain-text, formatted per rules)

Rules:
- Gemini receives structured text facts, not raw JSON arrays.
- All arithmetic is already done in ContextBuilder.
- Output is validated; retries preserve the full original base prompt.
- _get_mock_advisory() is available in dev mode only.
"""

import re
from datetime import datetime, timedelta
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
# SOIL MOISTURE CATEGORY & TRAJECTORY HELPERS
# (ClimateAdapt / project percentile bands)
# ---------------------------------------------------------------------------

def soil_word(p: Optional[float]) -> str:
    """Map a soil-moisture percentile to its category word."""
    if p is None:
        return ""
    if p > 98:
        return "Exceptional Wet"
    if p > 95:
        return "Extreme Wet"
    if p > 90:
        return "Severe Wet"
    if p > 80:
        return "Moderate Wet"
    if p > 70:
        return "Abnormally Wet"
    if p > 30:
        return "Normal"
    if p > 20:
        return "Abnormally Dry"
    if p > 10:
        return "Moderate Dry"
    if p > 5:
        return "Extreme Dry"
    if p > 2:
        return "Severe Dry"
    return "Exceptional Dry"


def _band_index(p: float) -> int:
    """Ordinal band index (0 = Exceptional Dry ... 10 = Exceptional Wet). Higher = wetter."""
    order = [
        "Exceptional Dry", "Severe Dry", "Extreme Dry", "Moderate Dry", "Abnormally Dry",
        "Normal",
        "Abnormally Wet", "Moderate Wet", "Severe Wet", "Extreme Wet", "Exceptional Wet",
    ]
    return order.index(soil_word(p))


def soil_trajectory_keypoints(
    series: Optional[List[float]],
    start_date: str = "",
    peak_rain_date: str = "",
) -> Dict[str, Any]:
    """Walk the DAILY soil-moisture percentile series and extract band-crossing
    key points (start, interior turning points that cross a category band, end).

    Returns:
      {
        'available': bool,
        'points': [ {'word': <category>, 'when': <relative phrase>, 'kind': 'start'|'trough'|'peak'|'end'} ],
        'direction': 'rising'|'falling'|'steady'|'mixed',
        'hint': <one-line phrase the model can build the sentence from>,
      }
    Numbers are NEVER included in any output field — only category words + timing.
    """
    if not series or len(series) < 2:
        return {"available": False, "points": [], "direction": "steady", "hint": ""}

    def _rel(day_idx: int) -> str:
        if peak_rain_date and start_date:
            try:
                _pk = datetime.strptime(peak_rain_date[:10], "%Y-%m-%d")
                _st = datetime.strptime(start_date[:10], "%Y-%m-%d")
                if (_st + timedelta(days=day_idx)) == _pk:
                    return "around the wettest day"
            except Exception:
                pass
        if start_date:
            try:
                _st = datetime.strptime(start_date[:10], "%Y-%m-%d")
                d = _st + timedelta(days=day_idx)
                dfmt = d.strftime("%d/%m/%Y")
                if day_idx <= 0:
                    return "at the start"
                if day_idx == 1:
                    return f"by {dfmt} (tomorrow)"
                return f"around {dfmt} (after about {day_idx} days)"
            except Exception:
                pass
        if day_idx <= 0:
            return "at the start"
        return f"after about {day_idx} days"

    n = len(series)
    bands = [_band_index(v) for v in series]

    points = [{"word": soil_word(series[0]), "when": _rel(0), "kind": "start", "_idx": 0, "_band": bands[0]}]

    i = 1
    while i < n - 1:
        prev, cur, nxt = series[i - 1], series[i], series[i + 1]
        is_trough = cur <= prev and cur <= nxt
        is_peak = cur >= prev and cur >= nxt
        if is_trough or is_peak:
            last_band = points[-1]["_band"]
            if bands[i] != last_band:
                kind = "trough" if is_trough else "peak"
                points.append({
                    "word": soil_word(cur),
                    "when": _rel(i),
                    "kind": kind,
                    "_idx": i,
                    "_band": bands[i]
                })
        i += 1

    end_word = soil_word(series[-1])
    if points[-1]["_band"] != bands[-1] or points[-1]["kind"] == "start":
        points.append({
            "word": end_word,
            "when": _rel(n - 1),
            "kind": "end",
            "_idx": n - 1,
            "_band": bands[-1]
        })
    else:
        points[-1]["kind"] = "end"
        points[-1]["when"] = _rel(n - 1)

    if bands[-1] > bands[0]:
        direction = "rising"
    elif bands[-1] < bands[0]:
        direction = "falling"
    else:
        direction = "steady"

    if any(pt["kind"] in ("trough", "peak") for pt in points):
        direction = "mixed"

    if len(points) == 1 or all(p["word"] == points[0]["word"] for p in points):
        hint = f"soil moisture stays {points[0]['word']} throughout"
    else:
        seq = " -> ".join(f"{p['word']} ({p['when']})" for p in points)
        hint = f"soil moisture trajectory: {seq}"

    clean = [{"word": p["word"], "when": p["when"], "kind": p["kind"]} for p in points]
    return {"available": True, "points": clean, "direction": direction, "hint": hint}


# ---------------------------------------------------------------------------
# OUTLOOK DECISION ENGINE
# Rule-based agricultural outlook, grounded in current IMD operational thresholds.
# ---------------------------------------------------------------------------

def decide_outlook(
    max_daily_rain: Optional[float],
    max_temp: Optional[float],
    wind_gusts: Optional[float],
    soil_moisture_available: bool,
    soil_percentile: Optional[float],
    station_type: str = "plains",
) -> Dict[str, Any]:
    """Return {'outlook': <Favorable|Cautionary|Unfavorable>, 'reasons': [...]}.

    Deterministic and auditable: each fired rule is recorded in 'reasons'.
    """
    st = (station_type or "plains").strip().lower()
    heat_threshold = {"plains": 40.0, "coastal": 37.0, "hilly": 30.0}.get(st, 40.0)

    sm = soil_percentile if soil_moisture_available else None

    unfavorable_reasons: List[str] = []
    cautionary_reasons: List[str] = []

    # ---- Rainfall (peak DAILY value, per IMD 24-hr bands) ----
    if max_daily_rain is not None:
        if max_daily_rain >= 115.6:
            unfavorable_reasons.append(
                f"peak daily rain {max_daily_rain:.1f} mm is Very Heavy+ (IMD >=115.6 mm) — flooding/waterlogging risk"
            )
        elif max_daily_rain >= 64.5:
            if sm is not None and sm > 70:
                unfavorable_reasons.append(
                    f"Heavy daily rain {max_daily_rain:.1f} mm (IMD 64.5-115.5) on already-wet soil ({soil_word(sm)})"
                )
            else:
                cautionary_reasons.append(
                    f"Heavy daily rain {max_daily_rain:.1f} mm (IMD 64.5-115.5 mm)"
                )
        elif max_daily_rain >= 15.6:
            if wind_gusts is not None and wind_gusts >= 40:
                cautionary_reasons.append(
                    f"Moderate rain {max_daily_rain:.1f} mm with gusty winds — spray drift/minor lodging risk"
                )

    # ---- Heat (IMD absolute thresholds) ----
    if max_temp is not None:
        if max_temp >= 45.0:
            unfavorable_reasons.append(
                f"max temperature {max_temp:.1f} C reaches IMD severe-heat level (>=45 C)"
            )
        elif max_temp >= heat_threshold:
            cautionary_reasons.append(
                f"max temperature {max_temp:.1f} C at/above IMD {st} heatwave threshold (>={heat_threshold:.0f} C)"
            )

    # ---- Wind gusts (IMD gust bands) ----
    if wind_gusts is not None:
        if wind_gusts >= 50:
            unfavorable_reasons.append(
                f"wind gusts {wind_gusts:.0f} km/h (>=50) — lodging and spray loss risk"
            )
        elif wind_gusts >= 40:
            cautionary_reasons.append(
                f"wind gusts {wind_gusts:.0f} km/h (40-50 band) — postpone spraying"
            )

    # ---- Soil moisture (percentile categories) ----
    if sm is not None:
        if sm > 90 or sm <= 5:
            unfavorable_reasons.append(f"soil moisture is {soil_word(sm)}")
        elif sm > 70 or sm <= 30:
            cautionary_reasons.append(f"soil moisture is {soil_word(sm)}")

    if unfavorable_reasons:
        return {"outlook": "Unfavorable", "reasons": unfavorable_reasons + cautionary_reasons}
    if cautionary_reasons:
        return {"outlook": "Cautionary", "reasons": cautionary_reasons}
    return {"outlook": "Favorable", "reasons": ["no IMD warning-level rain, heat, wind, or soil stress"]}


# ---------------------------------------------------------------------------
# Engine Class
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
        Generate plain-text advisory summary.

        If rag_chunks is empty, generation continues with a weather-only prompt
        (no ICAR recommendations invented).

        Raises:
            AdvisoryGenerationError: if all LLM providers fail on every attempt.
        """
        if not rag_chunks:
            logger.warning(
                "No ICAR RAG chunks retrieved. Generating weather-only advisory "
                "without crop-specific ICAR recommendations."
            )

        base_prompt = self._build_prompt(context, rag_chunks)
        prompt = base_prompt

        primary = get_primary_provider()
        fallback = get_fallback_provider()

        for attempt in range(1, 4):
            raw = None
            try:
                raw = await primary.generate_text(prompt=prompt, temperature=settings.TEMPERATURE)
            except Exception as e:
                logger.warning(f"Primary provider attempt {attempt} failed: {e}")

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

        # Extract Forecast Details
        forecast_days = 0
        total_rainfall = 0.0
        rainy_days = 0
        max_daily_rain = 0.0
        rainfall_pattern = "Dry"
        min_temp = 0.0
        max_temp = 0.0
        avg_min_temp = 0.0
        avg_max_temp = 0.0
        start_fmt = ""
        end_fmt = ""

        if ctx.forecast_summary:
            fs = ctx.forecast_summary
            forecast_days = fs.forecast_days
            total_rainfall = fs.total_rainfall_mm
            rainy_days = fs.rainy_days
            max_daily_rain = fs.maximum_daily_rainfall_mm
            rainfall_pattern = fs.rainfall_pattern
            min_temp = fs.minimum_temperature_c
            max_temp = fs.maximum_temperature_c
            avg_min_temp = fs.average_min_temperature_c
            avg_max_temp = fs.average_max_temperature_c
            start_fmt = self._fmt_date(fs.forecast_start_date)
            end_fmt = self._fmt_date(fs.forecast_end_date)
            forecast_section = (
                f"CORRECTED FORECAST SUMMARY (Primary — Bias-Corrected Model)\n"
                f"Forecast period  : {start_fmt} to {end_fmt} ({forecast_days} days)\n"
                f"Total rainfall   : {total_rainfall:.1f} mm\n"
                f"Rainy days       : {rainy_days} of {forecast_days}\n"
                f"Max daily rain   : {max_daily_rain:.1f} mm\n"
                f"Rainfall pattern : {rainfall_pattern}\n"
                f"Temp range       : {min_temp:.1f} °C to {max_temp:.1f} °C\n"
                f"Avg min/max temp : {avg_min_temp:.1f} °C / {avg_max_temp:.1f} °C"
            )
        else:
            wa = ctx.weather_api_summary
            start_fmt = self._fmt_date(wa.api_start_date) if wa.api_start_date else ""
            end_fmt = self._fmt_date(wa.api_end_date) if wa.api_end_date else ""
            max_daily_rain = wa.api_total_rainfall_mm
            max_temp = wa.api_max_temp_c
            forecast_section = "CORRECTED FORECAST SUMMARY: Not available."

        # Weather API secondary
        wa = ctx.weather_api_summary
        api_start_fmt = self._fmt_date(wa.api_start_date) if wa.api_start_date else ""
        api_end_fmt = self._fmt_date(wa.api_end_date) if wa.api_end_date else ""
        api_section = (
            f"WEATHER API DAILY ({api_start_fmt} to {api_end_fmt})\n"
            f"Total rainfall: {wa.api_total_rainfall_mm:.1f} mm | "
            f"Temp range: {wa.api_min_temp_c:.1f} °C – {wa.api_max_temp_c:.1f} °C"
        )

        # Rule-based outlook (IMD-grounded, deterministic)
        soil_pct_for_outlook = ctx.soil_moisture_summary.forecast_max_percentile if (ctx.soil_moisture_summary and av.soil_moisture_available) else None
        outlook_result = decide_outlook(
            max_daily_rain=max_daily_rain if ctx.forecast_summary else wa.api_total_rainfall_mm,
            max_temp=max_temp if ctx.forecast_summary else wa.api_max_temp_c,
            wind_gusts=cw.wind_gusts_kmh,
            soil_moisture_available=av.soil_moisture_available,
            soil_percentile=soil_pct_for_outlook,
            station_type="plains",
        )
        decided_outlook = outlook_result["outlook"]
        outlook_reasons = "; ".join(outlook_result["reasons"])
        outlook_sentence = f"Overall, the agricultural outlook for this period is *{decided_outlook}*."

        # Soil moisture trajectory
        soil_traj_hint = ""
        if ctx.soil_moisture_summary and av.soil_moisture_available:
            sm = ctx.soil_moisture_summary
            end_word = soil_word(sm.end_percentile)
            soil_traj_hint = f"soil moisture trajectory: {soil_word(sm.start_percentile)} (at the start) -> {end_word} (end)"

        if ctx.soil_moisture_summary and av.soil_moisture_available:
            sm = ctx.soil_moisture_summary
            _traj_line = f"Soil-moisture trajectory (category words): {soil_traj_hint}\n" if soil_traj_hint else f"Soil-moisture condition (category word): {soil_word(sm.end_percentile)}\n"
            sm_section = (
                f"SOIL MOISTURE SUMMARY (category words only — raw percentile values are deliberately omitted)\n"
                f"{_traj_line}"
                f"\n"
                f"IMPORTANT SOIL MOISTURE RULES:\n"
                f"- Describe soil moisture ONLY with the category words provided above "
                f"(Exceptional/Severe/Moderate/Abnormally Wet, Normal, Abnormally/Moderate/Extreme/Severe/Exceptional Dry).\n"
                f"- NEVER print a percentile number, a w_frac value, or the word 'percentile'.\n"
                f"- If the trajectory shows a dip or recovery, describe it with its timing and tie it to the weather; otherwise describe the steady condition.\n"
                f"- Add the farming implication (wet -> waterlogging/drainage; dry -> conserve water/irrigate)."
            )
        else:
            sm_section = "SOIL MOISTURE: No data available. Do NOT mention soil moisture at all in the advisory."

        # Source availability
        avail_section = (
            f"SOURCE AVAILABILITY\n"
            f"Corrected forecast : {'Yes' if av.corrected_forecast_available else 'No'}\n"
            f"Soil moisture      : {'Yes' if av.soil_moisture_available else 'No'}\n"
            f"Crop calendar      : {'Yes' if av.calendar_available else 'No'}"
        )

        # RAG Chunks
        rag_lines = []
        if rag_chunks:
            for i, chunk in enumerate(rag_chunks, 1):
                rag_lines.append(
                    f"[Chunk {i}]\n"
                    f"Source    : {chunk.get('source', 'ICAR')}\n"
                    f"Page      : {chunk.get('page', 'N/A')}\n"
                    f"Crop      : {chunk.get('crop', 'N/A')}\n"
                    f"Season    : {chunk.get('season', 'N/A')}\n"
                    f"Similarity: {round(chunk.get('score') or 0.0, 3)}\n"
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

        # Crop instructions
        is_general = crop.crop_id.lower().strip() == "general"
        if is_general:
            p_count_instruction = "Generate exactly ONE paragraph of plain text."
            word_count_instruction = "Total word count MUST be between 70 and 100 words."
            structure_instruction = (
                "AUDIENCE: An ordinary reader in a village — could be a farmer, a family member, or a "
                "field worker. Write the way a helpful local agriculture extension officer (KVK) would explain "
                "things — simple, direct, and focused on what to DO. Do NOT write like a weather report.\n"
                "\n"
                "The single paragraph MUST:\n"
                f"- Begin exactly with: \"Over the next {forecast_days} days, \"\n"
                "- Name the location (village, district, state) plainly, as a place — do NOT address a specific "
                "group such as \"farmers in\" the village. Speak to the reader directly (\"you\") or impersonally.\n"
                "- Use relative timing that anyone understands (e.g. \"after about 4 days\", "
                f"\"around {end_fmt}\") instead of listing full calendar dates for events.\n"
                "\n"
                "USE THE INPUT VALUES, BUT PAIR EACH WITH AN ACTION:\n"
                "- Rainfall: state the total rainfall over the whole period as a plain sum "
                f"(*{total_rainfall:.1f} mm* over *{rainy_days}* rainy days). Then describe the WETTEST SINGLE DAY "
                "as an upcoming EVENT, not a statistic, and you MUST say WHICH DAY it is. "
                "If no date for the wettest day is available, say \"on the wettest day\" and do NOT invent a date. "
                "Do NOT write it as \"your peak rainfall will be X\".\n"
                "- Pair the rain with the matching action (drainage for heavy days, water conservation for light rain).\n"
                "- Mention the temperature/heat and gusty-wind situation plainly, and if winds are strong warn that "
                "spraying will drift and be wasted — so postpone spray/fertilizer.\n"
                "- Give 1-2 concrete field actions tied to WHEN to act or WHAT sign to watch.\n"
                "\n"
                "DAILY RAINFALL CATEGORY (this table applies to a SINGLE DAY's rainfall ONLY — the peak daily value. "
                "NEVER apply it to the multi-day total):\n"
                "  0.1 - 15.5 mm    -> Light Rainfall\n"
                "  15.6 - 64.4 mm   -> Moderate Rainfall\n"
                "  64.5 - 115.5 mm  -> Heavy Rainfall\n"
                "  115.6 - 204.4 mm -> Very Heavy Rainfall\n"
                "  more than 204.4 mm -> Extremely Heavy Rainfall\n"
                "\n"
                "SOIL MOISTURE — describe how it changes over the period, in PLAIN WORDS ONLY "
                "(category words, never numbers):\n"
                + (
                    f"  The soil-moisture trajectory has already been worked out for you: {soil_traj_hint}. "
                    "Turn this into ONE natural sentence — mention the direction and, if the trajectory shows a "
                    "dip or recovery (a trough or peak), say when it happens and tie it to the weather "
                    "(e.g. drying out first, then recovering after the rain).\n"
                    if soil_traj_hint else
                    "  If no soil-moisture trajectory is provided, skip soil moisture silently.\n"
                )
                + "- Add the farming implication (wet -> watch for waterlogging, ensure drainage; "
                "dry -> conserve water, plan irrigation).\n"
                "- NEVER print any soil moisture number, the word 'percentile', or 'w_frac' — category words only.\n"
                "\n"
                "- Keep sentences short and plain. Avoid filler and jargon.\n"
                f"- The outlook for this period has ALREADY been decided as: {decided_outlook} "
                f"(reason: {outlook_reasons}). You MUST end the paragraph with EXACTLY this sentence, "
                f"word for word:\n  \"{outlook_sentence}\"\n"
                "  Do NOT choose a different outlook word; use the decided one above."
            )
        else:
            p_count_instruction = "Generate exactly THREE paragraphs of plain text, separated by a blank line."
            word_count_instruction = "Total word count MUST be between 90 and 140 words."
            structure_instruction = (
                "REGISTER: Write as an expert agrometeorologist — professional, precise, and detailed. "
                "But address the reader neutrally: do NOT use \"farmers in\" or address any specific group; "
                "speak impersonally or to \"you\". Use relative timing anyone understands (e.g. \"after about "
                f"4 days\", \"around {end_fmt}\") rather than listing full calendar dates for events.\n"
                "\n"
                f"PARAGRAPH 1 — Weather & Climate Analysis (45–65 words):\n"
                f"- MUST begin exactly with: \"Over the next {forecast_days} days, \"\n"
                "- Name the location (village, district, state) as a place, not as \"farmers in\".\n"
                "- State total rainfall as a plain sum (*{total_rainfall:.1f} mm* over *{rainy_days}* rainy days).\n"
                "- Describe the WETTEST SINGLE DAY as an event and you MUST say WHICH DAY it is. "
                "If no date for the wettest day is available, say \"on the wettest day\" and do NOT invent one.\n"
                "- Cover temperature range and heat-stress implications, and current observed conditions "
                "(humidity, wind speed, gusts, cloud cover) in plain professional terms.\n"
                + (
                    f"- Soil moisture: the trajectory has already been worked out: {soil_traj_hint}. "
                    "Describe how it changes over the period in CATEGORY WORDS only (never a number or word 'percentile'); "
                    "if it dips or recovers, note when and tie it to the weather.\n"
                    if soil_traj_hint else
                    "- Soil moisture: skip silently (no data).\n"
                )
                + "- Write in flowing sentences, not a list. No recommendations in Paragraph 1.\n"
                "\n"
                "DAILY RAINFALL CATEGORY (this table applies to a SINGLE DAY's rainfall ONLY — the peak daily value. "
                "NEVER apply it to the multi-day total):\n"
                "  0.1 - 15.5 mm    -> Light Rainfall\n"
                "  15.6 - 64.4 mm   -> Moderate Rainfall\n"
                "  64.5 - 115.5 mm  -> Heavy Rainfall\n"
                "  115.6 - 204.4 mm -> Very Heavy Rainfall\n"
                "  more than 204.4 mm -> Extremely Heavy Rainfall\n"
                "\n"
                f"PARAGRAPH 2 — Crop-Specific Advisory & Field Management (25–40 words):\n"
                f"- {rag_instruction}\n"
                "- Give the most relevant actions for the current stage: irrigation, fertilizer (type/rate/timing), "
                "pest and disease watch, drainage — tied to WHEN to act or WHAT sign to watch.\n"
                f"- Reference the specific crop and its {crop.crop_stage} stage, and the key weather risk (excess rain, "
                "heat, wind) and the response. Be concrete, not vague. Do NOT make claims about data sources.\n"
                "\n"
                f"PARAGRAPH 3 — Outlook & Risk Summary (15–25 words):\n"
                "- Summarise the overall risk level and the single biggest weather risk to watch this period.\n"
                "- Advise on monitoring frequency or the key action to take.\n"
                f"- The outlook has ALREADY been decided as: {decided_outlook} (reason: {outlook_reasons}). "
                f"Your risk summary MUST be consistent with this, and you MUST end with EXACTLY this sentence, word for word:\n  \"{outlook_sentence}\"\n"
                "  Do NOT choose a different outlook word; use the decided one above."
            )

        prompt = f"""You are an expert agrometeorologist at FarmRisk. Generate a professional {forecast_days}-Day Crop-Specific Advisory Summary.

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

=== {avail_section} ===

=== RETRIEVED ICAR KNOWLEDGE ===
{rag_section}

=== OUTPUT INSTRUCTIONS ===
{p_count_instruction}
{word_count_instruction} Be concise, focused, and precise — ensure total word count is strictly within the allowed range.
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

{structure_instruction}

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

        error_list = "\n".join(f"  - {e}" for e in errors)
        length_warning = ""
        if any("Too long" in e for e in errors):
            length_warning = "CRITICAL LENGTH WARNING: Your previous response exceeded the max word count. Trim fluff, reduce wordiness, and ensure the entire text is strictly under 130 words!\n\n"

        return (
            f"=== ORIGINAL FACTUAL CONTEXT AND RETRIEVED ICAR KNOWLEDGE ===\n"
            f"{base_prompt}\n\n"
            f"=== PREVIOUS INVALID OUTPUT (DO NOT COPY) ===\n"
            f"{previous}\n\n"
            f"=== VALIDATION ERRORS THAT MUST BE CORRECTED ===\n"
            f"{error_list}\n\n"
            f"=== CORRECTION INSTRUCTIONS ===\n"
            f"{length_warning}"
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

        is_general = context.crop_context.crop_id.lower().strip() == "general"
        expected_paragraphs = 1 if is_general else 3

        # ---- Paragraph count check ----
        if len(paragraphs) != expected_paragraphs:
            errors.append(
                f"Expected exactly {expected_paragraphs} paragraph{'s' if expected_paragraphs > 1 else ''} "
                f"separated by a blank line, got {len(paragraphs)}."
            )

        # ---- Word count check ----
        min_words = 70 if is_general else 90
        max_words = 100 if is_general else 140
        word_count = len(text.split())
        if word_count < min_words:
            errors.append(f"Too short: {word_count} words (min {min_words}).")
        if word_count > max_words:
            errors.append(f"Too long: {word_count} words (max {max_words}).")

        # ---- Paragraph 1 date/period prefix check ----
        forecast_days = context.forecast_summary.forecast_days if context.forecast_summary else 0
        if paragraphs:
            start_dd = None
            end_dd = None
            if context.forecast_summary:
                start_dd = self._fmt_date(context.forecast_summary.forecast_start_date)
                end_dd = self._fmt_date(context.forecast_summary.forecast_end_date)
            elif context.weather_api_summary.api_start_date:
                start_dd = self._fmt_date(context.weather_api_summary.api_start_date)
                end_dd = self._fmt_date(context.weather_api_summary.api_end_date)

            expected_prefix = f"Over the next {forecast_days} days," if forecast_days > 0 else (f"From {start_dd}" if start_dd else None)
            alt_prefix = f"From {start_dd}" if start_dd else None

            p1_start = paragraphs[0]
            prefix_ok = False
            if expected_prefix and p1_start.startswith(expected_prefix):
                prefix_ok = True
            elif alt_prefix and p1_start.startswith(alt_prefix):
                prefix_ok = True

            if not prefix_ok and expected_prefix:
                errors.append(f"Paragraph 1 must start with '{expected_prefix}'.")

        # ---- Expected outlook determination & validation ----
        max_daily_rain = context.forecast_summary.maximum_daily_rainfall_mm if context.forecast_summary else context.weather_api_summary.api_total_rainfall_mm
        max_temp = context.forecast_summary.maximum_temperature_c if context.forecast_summary else context.weather_api_summary.api_max_temp_c
        soil_pct = context.soil_moisture_summary.forecast_max_percentile if (context.soil_moisture_summary and context.availability.soil_moisture_available) else None
        
        outlook_info = decide_outlook(
            max_daily_rain=max_daily_rain,
            max_temp=max_temp,
            wind_gusts=context.current_weather.wind_gusts_kmh,
            soil_moisture_available=context.availability.soil_moisture_available,
            soil_percentile=soil_pct,
            station_type="plains",
        )
        expected_outlook = outlook_info["outlook"]

        valid_endings = [
            "Overall, the agricultural outlook for this period is Favorable.",
            "Overall, the agricultural outlook for this period is *Favorable*.",
            "Overall, the agricultural outlook for this period is Cautionary.",
            "Overall, the agricultural outlook for this period is *Cautionary*.",
            "Overall, the agricultural outlook for this period is Unfavorable.",
            "Overall, the agricultural outlook for this period is *Unfavorable*.",
        ]
        if paragraphs:
            last_p = paragraphs[-1]
            if not any(last_p.endswith(e) for e in valid_endings):
                errors.append("Last paragraph must end with exactly one of the valid outlook sentences.")
            else:
                required_endings = [
                    f"Overall, the agricultural outlook for this period is {expected_outlook}.",
                    f"Overall, the agricultural outlook for this period is *{expected_outlook}*.",
                ]
                if not any(last_p.endswith(e) for e in required_endings):
                    errors.append(
                        f"Outlook mismatch: model must end with the rule-decided outlook "
                        f"('{expected_outlook}' or '*{expected_outlook}*'), but it used a different one."
                    )

        # ---- Paragraph 1 recommendation check (standard crops only) ----
        if paragraphs and not is_general:
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

        # ---- Structural markdown check ----
        for line in text.split("\n"):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                errors.append("Response must not contain markdown headings (lines starting with #).")
                break
        for line in text.split("\n"):
            stripped = line.lstrip()
            if re.match(r'^[-+•]\s', stripped):
                errors.append("Response must not contain markdown bullet lists (-, +, •).")
                break
        for line in text.split("\n"):
            stripped = line.lstrip()
            if re.match(r'^\d+\.\s', stripped):
                errors.append("Response must not contain numbered lists (1., 2., ...).")
                break
        if "```" in text:
            errors.append("Response must not contain code fences (```).")
        if re.search(r'\|\s*[-:]+\s*\|', text):
            errors.append("Response must not contain markdown tables.")
        if "**" in text:
            errors.append("Response must not contain double-asterisk bold (**word**).")
        if "__" in text:
            errors.append("Response must not contain double-underscore bold (__word__).")

        # ---- Soil Moisture Wording Guard ----
        if re.search(r'\bpercentile\b', text, re.IGNORECASE):
            errors.append("Response must not contain the word 'percentile' — describe soil moisture with a category word only.")

        # ---- Soil moisture semantic grounding check ----
        if context.soil_moisture_summary and context.availability.soil_moisture_available:
            sm_errors = self._validate_soil_moisture_semantics(text, context.soil_moisture_summary)
            errors.extend(sm_errors)

        return errors

    def _validate_soil_moisture_semantics(
        self,
        text: str,
        sm: Any,
    ) -> List[str]:
        errors = []
        start = sm.start_percentile
        end = sm.end_percentile
        mn = sm.forecast_min_percentile
        mx = sm.forecast_max_percentile

        if abs(mx - end) < 0.5 and abs(mn - start) < 0.5:
            return []

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

        normalised = text.lower()
        normalised = re.sub(r'\*([0-9.]+)\*', r'\1', normalised)
        normalised = re.sub(r'\bpercentile\b', '', normalised)
        normalised = re.sub(r'\s+', ' ', normalised)

        def fmt_variants(val: float) -> List[str]:
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
            for v in variants:
                pattern = rf'\b{re.escape(v)}\b'
                if re.search(pattern, text_segment):
                    return v
            return None

        for trigger_pattern in temporal_triggers:
            for m in re.finditer(trigger_pattern, normalised):
                after = normalised[m.start(): m.start() + 80]
                found_start = number_follows(after, mn_variants + start_variants)
                if not found_start:
                    continue

                to_match = re.search(r'\bto\b', after)
                if not to_match:
                    continue
                after_to = after[to_match.end():]
                found_end = number_follows(after_to, mx_variants + end_variants)
                if not found_end:
                    continue

                start_is_min = found_start in mn_variants and found_start not in end_variants
                end_is_max = found_end in mx_variants and found_end not in end_variants

                if start_is_min and end_is_max:
                    errors.append(
                        f"Soil moisture trend incorrectly describes forecast minimum ({mn:.1f}) "
                        f"to maximum ({mx:.1f}) as a temporal start→end change. "
                        f"Use start percentile ({start:.1f}) → end percentile ({end:.1f}) instead."
                    )
                    return errors

        return errors

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_date(date_str: str) -> str:
        """Convert any date string (YYYY-MM-DD or ISO datetime) to DD/MM/YYYY."""
        if not date_str:
            return ""
        clean = date_str[:10]
        try:
            dt = datetime.strptime(clean, "%Y-%m-%d")
            return dt.strftime("%d/%m/%Y")
        except Exception:
            return clean

    def _get_mock_advisory(self, context: AdvisoryContext) -> AdvisoryResponse:
        """
        Development/testing fallback ONLY.
        """
        if not getattr(settings, "ENABLE_MOCK_ADVISORY", False):
            raise AdvisoryGenerationError(
                "Mock advisory is disabled in production. "
                "Enable ENABLE_MOCK_ADVISORY=true only for local development."
            )

        loc = context.location
        crop = context.crop_context.crop_name
        days = context.forecast_summary.forecast_days if context.forecast_summary else 10
        if context.forecast_summary:
            start = self._fmt_date(context.forecast_summary.forecast_start_date)
            end = self._fmt_date(context.forecast_summary.forecast_end_date)
            rain = context.forecast_summary.total_rainfall_mm
            tmin = context.forecast_summary.minimum_temperature_c
            tmax = context.forecast_summary.maximum_temperature_c
        else:
            today = datetime.now()
            start = today.strftime("%d/%m/%Y")
            end = (today + timedelta(days=9)).strftime("%d/%m/%Y")
            rain, tmin, tmax = 25.0, 23.0, 36.0

        p1 = (
            f"Over the next {days} days, the region of {loc.name}, {loc.district}, {loc.state} "
            f"is expected to experience a cumulative rainfall of *{rain:.0f} mm*, "
            f"with temperature ranging from *{tmin:.0f} °C* to *{tmax:.0f} °C*. "
            f"These conditions are broadly characteristic of the current agricultural season "
            f"and will have a moderate impact on {crop} crops in the field."
        )
        p2 = (
            f"Consult local agricultural extensions for crop care. Maintain optimal soil conditions, "
            f"monitor crop health regularly, and ensure adequate field drainage."
        )
        p3 = (
            f"Farmers should monitor local weather closely. "
            f"Overall, the agricultural outlook for this period is Cautionary."
        )
        return AdvisoryResponse(advisory_summary=f"{p1}\n\n{p2}\n\n{p3}")

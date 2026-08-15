"""
pest_disease_card.py
====================================================================
Builds the crop-specific "PEST & DISEASE" overview card for the Farm
Risk page.

Production port of `pest_disease_card_urmin.py` (standalone script).
The card-building logic — deterministic score, risk band, RAG-grounded
prompt, strict-JSON LLM output and anti-hallucination validation — is
kept identical; only the LLM plumbing is replaced with the existing
app.llm.providers (Gemini primary / Groq fallback).

Design:
  1. The NUMBER is decided deterministically (score_pest) -> risk band.
  2. The LLM only *names* pests and *phrases* actions, grounded ONLY in
     the retrieved RAG chunks. It may not invent pest names or chemicals.
  3. Output is strict JSON so the frontend can render the card directly.
  4. `crop_id == "general"` produces the crop-agnostic fallback card
     (existing general-advisory behaviour preserved).
"""

from typing import Optional, List, Dict, Any, Callable, Tuple
import json
import re

from app.core.logging import logger
from app.llm.providers import get_primary_provider, get_fallback_provider


# ---------------------------------------------------------------------------
# 0. Deterministic pest index
# ---------------------------------------------------------------------------
def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Linear interpolation of x from [x0,x1] onto [y0,y1], clamped to the ends."""
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    t = max(0.0, min(1.0, t))
    return y0 + t * (y1 - y0)


def _clamp100(v: float) -> float:
    return max(0.0, min(100.0, v))


def score_pest(
    avg_max_temp: Optional[float],
    humidity: Optional[float],
    soil_percentile: Optional[float],
    rainy_days: Optional[int],
    total_rainfall: Optional[float] = None,
) -> Tuple[float, str, List[str]]:
    """Pest / disease pressure (heuristic index).
    Blend of temperature suitability, humidity, and standing wetness — the
    conditions that favour fungal disease and sucking pests.
    """
    reasons: List[str] = []
    contributions: List[Tuple[float, str]] = []
    total = 0.0
    components = 0
    if avg_max_temp is not None:
        if 25 <= avg_max_temp <= 35:
            t_score = _lerp(abs(avg_max_temp - 30), 0, 5, 100, 60)
        elif avg_max_temp < 25:
            t_score = _lerp(avg_max_temp, 15, 25, 20, 60)
        else:
            t_score = _lerp(avg_max_temp, 35, 45, 60, 20)
        total += t_score
        components += 1
        reasons.append("warm temperatures favour pest activity")
        contributions.append((t_score, f"warm temperatures ({avg_max_temp:.1f} C) in the pest-active band"))
    if humidity is not None:
        h_score = _lerp(humidity, 50, 95, 10, 100)
        total += h_score
        components += 1
        if humidity >= 70:
            reasons.append(f"high humidity {humidity:.0f}% favours disease")
        contributions.append((h_score, f"humidity of {humidity:.0f}% favouring fungal disease"))
    wet_score = 0.0
    wet_label = None
    if soil_percentile is not None:
        s = _lerp(soil_percentile, 40, 95, 20, 90)
        if s > wet_score:
            wet_score, wet_label = s, "wet soil supporting pest/disease build-up"
    if rainy_days is not None:
        s = _lerp(rainy_days, 2, 8, 20, 90)
        if s > wet_score:
            wet_score, wet_label = s, f"{rainy_days} rainy days keeping foliage wet"
    if wet_label is not None:
        total += wet_score
        components += 1
        reasons.append("prolonged wetness supports pest/disease build-up")
        contributions.append((wet_score, wet_label))
    if components == 0:
        return 0.0, "insufficient data for pest index", ["insufficient data for pest index"]
    score = _clamp100(total / components)
    major = max(contributions, key=lambda c: c[0])[1]
    return score, major, reasons


# ---------------------------------------------------------------------------
# 1. Deterministic band from the numeric pest index
# ---------------------------------------------------------------------------
def pest_band(score: float) -> str:
    """Map the 0-100 pest index onto a display band."""
    if score >= 66:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# 2. Build a RAG-grounded prompt that returns STRICT JSON
# ---------------------------------------------------------------------------
def build_pest_prompt(
    band: str,
    driver: str,
    reasons: List[str],
    crop_name: str,
    crop_stage: str,
    avg_max_temp: Optional[float],
    humidity: Optional[float],
    rainy_days: Optional[int],
    total_rainfall: Optional[float],
    rag_chunks: List[Dict[str, Any]],
    season: str = "",
    is_general: bool = False,
) -> str:
    # In the general case we never name a specific pest/disease/chemical,
    # even if some crop chunks were retrieved — mirror the main prompt builder.
    if is_general:
        rag_chunks = []

    # ---- RAG context (the ONLY source allowed to name pests/chemicals) ----
    rag_lines = []
    if rag_chunks:
        for i, c in enumerate(rag_chunks, 1):
            rag_lines.append(
                f"[Chunk {i}] source={c.get('source', 'N/A')} "
                f"page={c.get('page', 'N/A')} crop={c.get('crop', 'N/A')} "
                f"season={c.get('season', 'N/A')} sim={round(c.get('score', 0.0), 3)}\n"
                f"{c.get('content', '').strip()}"
            )
        rag_block = "\n\n".join(rag_lines)
        naming_rule = (
            "Name pests/diseases and any control measures ONLY if they appear "
            "in the RETRIEVED KNOWLEDGE below. If the knowledge names specific "
            "pests (e.g. sucking pests, bollworm, aphids, jassids, whitefly) or "
            "diseases, use those names. Do NOT invent any pest, disease, or "
            "chemical/pesticide name that is not in the retrieved text."
        )
    else:
        if is_general:
            rag_block = "[General advisory — no single crop targeted]"
            naming_rule = (
                "This is a GENERAL advisory not tied to any one crop. Do NOT name "
                "any specific pest, disease, crop, or pesticide. Refer ONLY to the "
                "broad disease/pest category implied by the weather (e.g. \"fungal "
                "disease and sucking pests\" when humid and wet). Actions must be "
                "generic scouting / cultural measures (inspect undersides, improve "
                "airflow, remove weeds, improve drainage) — never a chemical name."
            )
        else:
            rag_block = "[No crop-specific knowledge retrieved]"
            naming_rule = (
                "No crop-specific knowledge was retrieved. Do NOT name any specific "
                "pest, disease, or pesticide. Refer only to the GENERAL disease/pest "
                "category implied by the weather (e.g. \"fungal disease and sucking "
                "pests\" when humid and wet). Actions must be generic scouting / "
                "cultural measures (inspect undersides, improve airflow, remove weeds, "
                "improve drainage) — never a chemical name."
            )

    weather_facts = []
    if humidity is not None:
        weather_facts.append(f"humidity {humidity:.0f}%")
    if avg_max_temp is not None:
        weather_facts.append(f"avg max temp {avg_max_temp:.1f} C")
    if rainy_days is not None:
        weather_facts.append(f"{rainy_days} rainy days")
    if total_rainfall is not None:
        weather_facts.append(f"{total_rainfall:.1f} mm total rain")
    weather_line = ", ".join(weather_facts) if weather_facts else "limited weather data"

    reason_line = "; ".join(reasons) if reasons else driver

    # In a crop-specific card, the summary must name the crop and be grounded
    # in the crop calendar (season + growth stage); in the general case it must
    # stay crop-agnostic.
    if is_general:
        summary_crop_hint = " Do NOT name any crop, season, or growth stage."
    else:
        _cal_bits = [f"the crop '{crop_name}'"]
        if season:
            _cal_bits.append(f"the {season} season")
        _stg = (crop_stage or "").strip()
        if _stg:
            _cal_bits.append(_stg if _stg.lower().endswith("stage") else f"its {_stg} stage")
        _cal_phrase = ", and ".join([", ".join(_cal_bits[:-1]), _cal_bits[-1]]) if len(_cal_bits) > 1 else _cal_bits[0]
        summary_crop_hint = (
            f" You MUST name {crop_name} in the summary and tie the pest/disease risk "
            f"to the crop calendar — reference {_cal_phrase}."
        )

    prompt = f"""You are an agronomy assistant generating a PEST & DISEASE risk card.

The risk level has ALREADY been decided deterministically. Do NOT change it.
  Risk band       : {band}
  Main driver      : {driver}
  Contributing     : {reason_line}

{"CROP CONTEXT (crop calendar)" if not is_general else "CONTEXT"}
{(f"  Crop   : {crop_name}" + chr(10) + (f"  Season : {season}" + chr(10) if season else "") + f"  Stage  : {crop_stage}") if not is_general else "  General advisory — not tied to a specific crop or stage."}

WEATHER CONDITIONS (for phrasing the "why" only)
  {weather_line}

RETRIEVED KNOWLEDGE (the ONLY allowed source for pest/disease/chemical names)
{rag_block}

NAMING RULE
{naming_rule}

TASK
Return a JSON object (and NOTHING else — no markdown, no code fences) with EXACTLY this schema:
{{
  "risk": "{band}",
  "summary": "<two to three sentences (approximately 30-50 words) on WHY risk is elevated: tie humidity/warmth/wetness to the pest or disease type, and explain what this means for the crop right now. Do not repeat the risk band word.{summary_crop_hint}>",
  "potential": ["<named pest or disease 1>", "<named pest or disease 2>"],
  "actions": [
    {{"title": "<2-4 word action, imperative>", "detail": "<a SINGLE short sentence of 12-16 words (never more than 18): what to do and, when useful, WHY it helps — tie the benefit to the current weather/soil conditions. Keep it concise and specific, name the relevant pest/disease/crop-stage only when it matters>", "cites": [<chunk numbers that support this action, e.g. 1>]}},
    {{"title": "<2-4 word action>", "detail": "<a SINGLE short sentence of 12-16 words (never more than 18)>", "cites": [<chunk numbers>]}}
  ],
  "cites": [<chunk numbers that back the named pests/diseases in "potential">]
}}

RULES
- "risk" MUST be exactly "{band}".
- "potential": 1-3 items. Use ONLY names supported by the retrieved knowledge; if none, use the general category (e.g. "Fungal disease", "Sucking pests").
- "actions": 2-3 items; return 3 whenever the retrieved knowledge supports at least 3 distinct, grounded recommendations, otherwise return only the 2-3 you can genuinely support. Each is a concrete scouting or cultural/management step. Prefer non-chemical actions (scout undersides, improve airflow, remove weeds, ensure drainage). Only mention a chemical/pesticide if it is explicitly in the retrieved knowledge.
- Each action "detail" must explain WHY the step helps (its benefit under the current weather/soil conditions), not just what to do. Do not invent pesticides, chemicals, doses, diseases, pests, or treatments.
- WORD CAP: every action "detail" MUST be a SINGLE short sentence of 12-16 words and MUST NOT exceed 18 words. One full sentence only — no periods mid-detail, no second sentence. The "title" does not count toward this limit.
- CITATIONS: every "cites" value is a list of the [Chunk N] numbers shown above whose text supports that item. Cite ONLY chunks that actually contain the pest/action. If an item is a generic weather-driven category or a general cultural step not taken from any chunk, use an empty list [].
- Do NOT invent pest names, disease names, pesticide names, rates, or chunk numbers.
- Output raw JSON only.
"""
    return prompt


# ---------------------------------------------------------------------------
# 3. Parse + validate the model's JSON (defensive)
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> Dict[str, Any]:
    """Pull the first JSON object out of the model text, tolerating fences."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    # Grab from first { to last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _chunk_source(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """A compact, display-ready source descriptor for one RAG chunk."""
    return {
        "source": chunk.get("source", "N/A"),
        "page": chunk.get("page", "N/A"),
        "crop": chunk.get("crop", "N/A"),
        "season": chunk.get("season", "N/A"),
        "score": round(float(chunk.get("score", 0.0) or 0.0), 3),
    }


def _resolve_cites(raw_cites: Any, rag_chunks: List[Dict[str, Any]]) -> List[int]:
    """Turn a model 'cites' value into a list of valid 1-based chunk indices.

    Only keeps numbers that actually point at a retrieved chunk — fabricated
    or out-of-range citations are dropped.
    """
    if raw_cites is None:
        return []
    if isinstance(raw_cites, (int, str)):
        raw_cites = [raw_cites]
    valid = []
    for c in raw_cites:
        try:
            n = int(c)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= len(rag_chunks) and n not in valid:
            valid.append(n)
    return valid


def _cap_detail_words(detail: str, max_words: int = 18) -> str:
    """Enforce a word cap on an action detail while keeping the text natural.

    Never hard-truncates mid-word or mid-sentence. If the detail is over the
    cap it is cut at the last clause/sentence boundary ('.', ',', ';', '-', '—')
    that stays within the limit; otherwise the first `max_words` whole words
    are kept.
    """
    detail = (detail or "").strip()
    if not detail:
        return detail
    words = detail.split()
    if len(words) <= max_words:
        return detail
    boundaries = {".", ",", ";", "-", "\u2014", ":", "!", "?"}
    cut = -1
    for i, w in enumerate(words):
        if i + 1 >= max_words:
            break
        if w and w[-1] in boundaries:
            cut = i
    if cut >= 0:
        return " ".join(words[: cut + 1]).rstrip(" \u2014,;")
    return " ".join(words[:max_words])


def _validate_card(
    data: Dict[str, Any],
    band: str,
    rag_chunks: List[Dict[str, Any]],
    is_general: bool = False,
) -> Tuple[Dict[str, Any], List[str]]:
    """Coerce into the card shape and flag violations. Never raises."""
    errors: List[str] = []
    # A general advisory grounds nothing in crop RAG.
    if is_general:
        rag_chunks = []

    # Enforce the deterministic band regardless of what the model said.
    if str(data.get("risk", "")).upper() != band:
        errors.append(f"model risk '{data.get('risk')}' overridden to '{band}'")
    data["risk"] = band

    summary = str(data.get("summary", "")).strip()
    if not summary:
        errors.append("empty summary")
        summary = "Weather conditions favour pest and disease pressure this period."
    data["summary"] = summary

    potential = data.get("potential", [])
    if isinstance(potential, str):
        potential = [potential]
    potential = [str(p).strip() for p in potential if str(p).strip()][:3]
    if not potential:
        potential = ["Fungal disease", "Sucking pests"]
        errors.append("no pests named; fell back to general category")

    # Anti-hallucination: when no RAG, block anything that looks like a
    # specific chemical/product (very light heuristic).
    if not rag_chunks:
        chem_like = re.compile(
            r"(cid|thoate|conazole|mycin|prid|azole|mectin|dazim|thoxam|pronil|cozeb|strobin|aniliprole)\b",
            re.I,
        )
        for p in potential:
            if chem_like.search(p):
                errors.append(f"blocked chemical-like name without RAG: {p}")
        potential = [p for p in potential if not chem_like.search(p)] or [
            "Fungal disease",
            "Sucking pests",
        ]

    # General advisory: only allow broad category phrases; drop anything that
    # names a specific pest/disease/crop the model may have slipped in.
    if is_general:
        allowed = {
            "fungal disease", "fungal diseases", "sucking pests", "sucking pest",
            "pest pressure", "pests", "disease", "diseases",
            "pest and disease pressure", "bacterial disease", "root rot",
            "waterlogging stress",
        }
        kept = [p for p in potential if p.strip().lower() in allowed]
        dropped = [p for p in potential if p.strip().lower() not in allowed]
        if dropped:
            errors.append(f"general case: dropped specific names {dropped}")
        potential = kept or ["Fungal disease", "Sucking pests"]

    data["potential"] = potential

    # ---- Citations for the named pests/diseases (top-level "cites") ----
    # Empty for the general/no-RAG case since nothing is chunk-grounded.
    potential_cite_ids = _resolve_cites(data.get("cites"), rag_chunks)

    actions = data.get("actions", [])
    fixed_actions = []
    for a in actions[:3]:
        if isinstance(a, dict):
            title = str(a.get("title", "")).strip()
            detail = str(a.get("detail", "")).strip()
            cite_ids = _resolve_cites(a.get("cites"), rag_chunks)
        else:
            title, detail, cite_ids = str(a).strip(), "", []
        if title:
            fixed_actions.append({
                "title": title,
                "detail": _cap_detail_words(detail),
                "sources": [_chunk_source(rag_chunks[i - 1]) for i in cite_ids],
            })
    if len(fixed_actions) < 2:
        # sensible generic fallbacks matching the mock-up (no sources — generic)
        defaults = [
            {"title": "Scout daily", "detail": "Inspect leaf undersides and stem bases for early signs of feeding damage or pest activity.", "sources": []},
            {"title": "Improve airflow", "detail": "Remove weeds between rows to open the canopy, lower humidity, and reduce conditions that favour disease.", "sources": []},
        ]
        for d in defaults:
            if len(fixed_actions) >= 2:
                break
            fixed_actions.append(d)
        errors.append("fewer than 2 actions; padded with defaults")
    data["actions"] = fixed_actions[:3]

    # ---- Provenance ----
    # Sources backing the named pests/diseases.
    data["potential_sources"] = [_chunk_source(rag_chunks[i - 1]) for i in potential_cite_ids]

    # Deduped union of every chunk cited anywhere on the card, in chunk order.
    used_ids = set(potential_cite_ids)
    for a in fixed_actions:
        for s in a.get("sources", []):
            for idx, ch in enumerate(rag_chunks, 1):
                if _chunk_source(ch) == s:
                    used_ids.add(idx)
    data["sources"] = [_chunk_source(rag_chunks[i - 1]) for i in sorted(used_ids)]

    return data, errors


# ---------------------------------------------------------------------------
# 4. LLM adapter — reuses the existing primary/fallback providers
# ---------------------------------------------------------------------------
async def call_llm_text(prompt: str, _api_key: str = "") -> str:
    """Call the configured LLM provider (primary, then fallback) for raw text.

    Returns the raw model text. The card builder owns parsing/validation and
    degrades to the deterministic fallback if both providers fail.
    """
    primary = get_primary_provider()
    fallback = get_fallback_provider()
    try:
        return await primary.generate_text(prompt=prompt, temperature=0.2)
    except Exception as e:
        logger.warning(f"Primary provider failed for pest card: {e}")
        if fallback:
            try:
                return await fallback.generate_text(prompt=prompt, temperature=0.2)
            except Exception as fe:
                logger.error(f"Fallback provider failed for pest card: {fe}")
        raise


# ---------------------------------------------------------------------------
# 5. Orchestrator — the one function you call
# ---------------------------------------------------------------------------
async def build_pest_disease_card(
    avg_max_temp: Optional[float],
    humidity: Optional[float],
    soil_percentile: Optional[float],
    rainy_days: Optional[int],
    total_rainfall: Optional[float],
    crop_name: str,
    crop_stage: str,
    rag_chunks: List[Dict[str, Any]],
    llm_call: Callable,          # async (prompt, api_key) -> raw text
    api_key: str,
    crop_id: str = "",           # "general" -> generic, crop-agnostic card
    season: str = "",            # crop calendar: Kharif / Rabi / Zaid
) -> Dict[str, Any]:
    """Return a fully-formed pest/disease card dict, ready to serialise."""
    is_general = crop_id.lower().strip() == "general"

    # (1) deterministic score + band
    score, driver, reasons = score_pest(
        avg_max_temp=avg_max_temp,
        humidity=humidity,
        soil_percentile=soil_percentile,
        rainy_days=rainy_days,
        total_rainfall=total_rainfall,
    )
    band = pest_band(score)

    # (2) build grounded prompt
    prompt = build_pest_prompt(
        band=band,
        driver=driver,
        reasons=reasons,
        crop_name=crop_name,
        crop_stage=crop_stage,
        avg_max_temp=avg_max_temp,
        humidity=humidity,
        rainy_days=rainy_days,
        total_rainfall=total_rainfall,
        rag_chunks=rag_chunks or [],
        season=season,
        is_general=is_general,
    )

    # (3) call LLM + parse, with a deterministic fallback card on failure
    try:
        raw = await llm_call(prompt, api_key)
        data = _extract_json(raw)
        data, warnings = _validate_card(data, band, rag_chunks or [], is_general)
    except Exception as e:  # never let the card crash the pipeline
        logger.warning(f"Pest card LLM/parse failed: {e}; using deterministic fallback")
        warnings = [f"llm/parse failed: {e}; using deterministic fallback"]
        data = {
            "risk": band,
            "summary": f"{driver.capitalize()}.",
            "potential": ["Fungal disease", "Sucking pests"],
            "actions": [
                {"title": "Scout daily", "detail": "Inspect leaf undersides and stem bases for early signs of feeding damage or pest activity.", "sources": []},
                {"title": "Improve airflow", "detail": "Remove weeds between rows to open the canopy, lower humidity, and reduce conditions that favour disease.", "sources": []},
            ],
            "potential_sources": [],
            "sources": [],
        }

    # (4) attach the machine-readable score for the frontend / auditing
    data["score"] = round(score, 1)
    data["driver"] = driver
    data["_warnings"] = warnings
    return data

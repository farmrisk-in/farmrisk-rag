"""
weather_todos.py
====================================================================
Weather-driven "What To Do Today" actions (rain / temperature / wind).

Production port of `todo_card.py` (standalone script at the repo root).
The decision logic is kept VERBATIM — thresholds, IMD band cut-offs,
severity ranking and the LLM phrasing rules are untouched. Only the LLM
plumbing is replaced with the existing `app.llm.providers` (Gemini
primary / Groq fallback), mirroring `pest_disease_card.py`.

Design contract (auditable):
  * The code decides WHAT the action is and its severity + timing.
    The LLM only rephrases each action into a farmer-facing one-liner.
  * `build_todos` is deterministic and sorts most-severe first (stable:
    rain > temp > wind on ties).
  * `phrase_todos` rephrases every action in ONE call; on any failure it
    falls back to the deterministic strings (`_fallback_phrasing`).
"""

from typing import Any, Dict, List, Optional
import json
import re

from app.core.logging import logger
from app.llm.providers import get_primary_provider, get_fallback_provider


# ==============================================================================
# THRESHOLDS (identical to todo_card.py — never change these)
# Grounded in IMD operational warning bands (per-day / 24-hr):
#   Rain (mm/day): Light 2.5-15.5, Moderate 15.6-64.4, Heavy 64.5-115.5,
#                  Very Heavy 115.6-204.4, Extremely Heavy >204.4
#   Heat (plains): heatwave >=40 C, severe heatwave >=45 C (coastal 37, hilly 30)
#   Cold        : cold-wave watch when tmin drops well below normal (default <=8 C)
#   Wind (gust) : IMD bands 30-40, 40-50 kmph; >=50 damaging
# ==============================================================================
STATION_TYPE = "plains"  # plains / coastal / hilly

# --- Rain thresholds (mm) ---
RAIN_24H_HEAVY = 64.5          # heavy rain in next 24h -> act now
RAIN_24H_MODERATE = 15.6       # moderate rain in next 24h
RAIN_5D_VERYHEAVY = 115.6      # very heavy over window -> drainage
RAIN_5D_HEAVY = 64.5           # heavy over window -> drainage prep

# --- Rain SPELL thresholds (cm/hour — IMD "Category of Rain Spell") ---
# Light 1, Moderate 1-2, Intense 2-3, Very Intense 3-5,
# Extremely Intense 5-10, Cloud Burst >10 cm/hour.
# Spell drives the card only when it reaches Moderate+ (>=1 cm/hr = 10 mm/hr);
# below that we fall back to the daily IMD band.
SPELL_MIN_MMH = 10.0           # 1 cm/hr — Moderate spell floor to override daily

# --- Temperature thresholds (C) ---
HEAT_THRESHOLD = {"plains": 40.0, "coastal": 37.0, "hilly": 30.0}
HEAT_SEVERE = 45.0
COLD_TMIN = 8.0                # tmin at/below this -> cold-stress action

# --- Wind thresholds (km/h gusts) ---
WIND_DAMAGING = 50.0           # lodging / structural risk
WIND_SPRAY = 40.0             # drift -> postpone spraying

# Severity ranks: higher = more urgent (used for sorting)
SEV = {"unfavorable": 3, "cautionary": 2, "favorable": 1}


# ==============================================================================
# IMD 24-HOUR RAINFALL CLASSIFICATION (rainfall ending 0830 IST)
#   Very Light 0-2.4, Light 2.5-15.5, Moderate 15.6-64.4, Heavy 64.5-115.5,
#   Very Heavy 115.6-204.4, Extremely Heavy >=204.5 mm
# ==============================================================================
def rain_word(mm: Optional[float]) -> str:
    """Map a 24-hour rainfall total (mm) to its IMD category word."""
    if mm is None:            return ""
    if mm >= 204.5:  return "Extremely Heavy Rain"
    if mm >= 115.6:  return "Very Heavy Rain"
    if mm >= 64.5:   return "Heavy Rain"
    if mm >= 15.6:   return "Moderate Rain"
    if mm >= 2.5:    return "Light Rain"
    if mm > 0:       return "Very Light Rain"
    return "No Rain"


# ==============================================================================
# IMD RAIN SPELL CLASSIFICATION (peak hourly intensity)
#   Light 1, Moderate 1-2, Intense 2-3, Very Intense 3-5,
#   Extremely Intense 5-10, Cloud Burst >10 cm/hour.
# Input here is mm/hour (1 cm/hr = 10 mm/hr).
# ==============================================================================
def spell_word(mmph: Optional[float]) -> str:
    """Map a peak hourly rain intensity (mm/hour) to its IMD spell category."""
    if mmph is None:          return ""
    cmph = mmph / 10.0
    if cmph > 10:   return "Cloud Burst"
    if cmph >= 5:   return "Extremely Intense spell"
    if cmph >= 3:   return "Very Intense spell"
    if cmph >= 2:   return "Intense spell"
    if cmph >= 1:   return "Moderate spell"
    if cmph > 0:    return "Light spell"
    return ""


# Severity attached to each spell band (drives whether a spell escalates the card).
_SPELL_SEV = {
    "Light spell": "cautionary",
    "Moderate spell": "cautionary",
    "Intense spell": "unfavorable",
    "Very Intense spell": "unfavorable",
    "Extremely Intense spell": "unfavorable",
    "Cloud Burst": "unfavorable",
}


def _peak_hour(series: Optional[List[float]]):
    """Return (peak_value, hour_index) for an hourly series, or (None, None)."""
    if not series:
        return None, None
    peak_val = None
    peak_idx = None
    for i, v in enumerate(series):
        if v is None:
            continue
        if peak_val is None or v > peak_val:
            peak_val = v
            peak_idx = i
    return peak_val, peak_idx


def _hour_phrase(hour_idx: Optional[int]) -> str:
    """Relative timing phrase for an hour offset from now (0 = current hour)."""
    if hour_idx is None:
        return ""
    if hour_idx <= 0:            return "within the hour"
    if hour_idx == 1:           return "in about 1 hour"
    if hour_idx < 24:           return f"in about {hour_idx} hours"
    day = hour_idx // 24
    if day == 1:                return "tomorrow"
    return f"in about {day} days"


# ==============================================================================
# DETERMINISTIC DECISION ENGINE
# Code decides WHAT the action is and its severity + timing.
# Each candidate: {key, severity, title, why, timing, _sev}
# ==============================================================================
def _heat_threshold() -> float:
    return HEAT_THRESHOLD.get((STATION_TYPE or "plains").strip().lower(), 40.0)


def decide_rain_action(
    rain_hourly_24h: Optional[List[float]] = None,
    rain_24h: Optional[float] = None,
    rain_5d: Optional[float] = None,
    peak_rain_timing: str = "",
) -> Optional[Dict[str, Any]]:
    """One rain action, spell-first then daily-band fallback.

    Precedence:
      1. Peak HOURLY intensity from the 24h forecast -> IMD rain spell.
         If the spell reaches Moderate+ (>=1 cm/hr), it drives the card.
      2. Otherwise fall back to the daily IMD band: 24h total first,
         then 5-day accumulation.
    """
    # ---- 1. Hourly spell (24h) ----
    peak_mmph, peak_hr = _peak_hour(rain_hourly_24h)
    if peak_mmph is not None and peak_mmph >= SPELL_MIN_MMH:
        word = spell_word(peak_mmph)
        sev = _SPELL_SEV.get(word, "cautionary")
        when = _hour_phrase(peak_hr)
        when_clause = f" {when}" if when else ""
        cmph = peak_mmph / 10.0
        return {
            "key": "rain",
            "severity": sev,
            "title": "Clear field drainage",
            "why": (
                f"{word} ({cmph:.1f} cm/hr) expected{when_clause} — "
                f"sudden runoff and waterlogging risk"
            ),
            "timing": "Now" if (peak_hr is not None and peak_hr < 6) else "Today",
        }

    # ---- 2. Daily IMD band fallback ----
    r24 = rain_24h if rain_24h is not None else 0.0
    r5 = rain_5d if rain_5d is not None else 0.0

    if r24 >= RAIN_24H_HEAVY:
        return {
            "key": "rain",
            "severity": "unfavorable",
            "title": "Clear field drainage",
            "why": f"{rain_word(r24)} ({r24:.0f} mm) expected in next 24 hours — waterlogging risk",
            "timing": "Now",
        }

    if r5 >= RAIN_5D_VERYHEAVY:
        return {
            "key": "rain",
            "severity": "unfavorable",
            "title": "Clear field drainage",
            "why": f"very heavy rain ({r5:.0f} mm over 5 days) building{(' ' + peak_rain_timing) if peak_rain_timing else ''}",
            "timing": "Before rain",
        }

    if r5 >= RAIN_5D_HEAVY:
        return {
            "key": "rain",
            "severity": "cautionary",
            "title": "Clear field drainage",
            "why": f"heavy rain ({r5:.0f} mm over 5 days) ahead{(' ' + peak_rain_timing) if peak_rain_timing else ''}",
            "timing": "Before rain",
        }

    if r24 >= RAIN_24H_MODERATE:
        return {
            "key": "rain",
            "severity": "cautionary",
            "title": "Hold irrigation",
            "why": f"{rain_word(r24)} ({r24:.0f} mm) expected in next 24 hours",
            "timing": "Today",
        }

    return None


def decide_temp_action(
    tmax_24h: Optional[float],
    tmin_24h: Optional[float],
    tmax_5d: Optional[float],
    tmin_5d: Optional[float],
) -> Optional[Dict[str, Any]]:
    """One temperature action. Look at the 24h extremes FIRST; only if nothing
    fires there, fall back to the 5-day daily extremes."""
    ht = _heat_threshold()

    def _eval(tmax, tmin, near_term):
        if tmax is not None and tmax >= HEAT_SEVERE:
            return {
                "key": "temp",
                "severity": "unfavorable",
                "title": "Protect crop from heat",
                "why": f"severe heat ({tmax:.0f} C) expected — irrigate early morning, avoid midday work",
                "timing": "Today" if near_term else "This week",
            }
        if tmax is not None and tmax >= ht:
            return {
                "key": "temp",
                "severity": "cautionary",
                "title": "Guard against heat stress",
                "why": f"high temperatures ({tmax:.0f} C) {'today' if near_term else 'this week'} — water in cooler hours",
                "timing": "Today" if near_term else "This week",
            }
        if tmin is not None and tmin <= COLD_TMIN:
            return {
                "key": "temp",
                "severity": "cautionary",
                "title": "Protect against cold",
                "why": f"low temperatures ({tmin:.0f} C) {'tonight' if near_term else 'this week'} — cold-stress risk",
                "timing": "Tonight" if near_term else "This week",
            }
        return None

    # 24h first
    action = _eval(tmax_24h, tmin_24h, near_term=True)
    if action is not None:
        return action

    # fall back to 5-day daily extremes
    return _eval(tmax_5d, tmin_5d, near_term=False)


def decide_wind_action(
    gust_hourly_24h: Optional[List[float]] = None,
    gust_hourly_5d: Optional[List[float]] = None,
) -> Optional[Dict[str, Any]]:
    """One wind action from hourly gusts.

    Look at the peak gust in the next 24h FIRST (state when it hits). If nothing
    significant in 24h, fall back to the 5-day hourly peak.
    """
    def _eval(series, window_24h):
        peak, hr = _peak_hour(series)
        if peak is None:
            return None

        if window_24h:
            when = _hour_phrase(hr)
            when_clause = f" {when}" if when else ""
        else:
            # 5-day window: express the peak day rather than the hour
            day = (hr // 24) if hr is not None else None
            if day and day >= 1:
                when_clause = f" in about {day} day{'s' if day != 1 else ''}"
            else:
                when_clause = " later this week"

        if peak >= WIND_DAMAGING:
            return {
                "key": "wind",
                "severity": "unfavorable",
                "title": "Skip pesticide spraying",
                "why": f"damaging gusts ({peak:.0f} km/h) expected{when_clause} — drift and lodging risk, resume when winds ease",
                "timing": "All day",
            }
        if peak >= WIND_SPRAY:
            return {
                "key": "wind",
                "severity": "cautionary",
                "title": "Skip pesticide spraying",
                "why": f"gusty winds ({peak:.0f} km/h) expected{when_clause} — spray drift risk, resume when winds ease",
                "timing": "All day",
            }
        return None

    # 24h first
    action = _eval(gust_hourly_24h, window_24h=True)
    if action is not None:
        return action

    # fall back to 5-day hourly peak
    return _eval(gust_hourly_5d, window_24h=False)


def build_todos(
    rain_hourly_24h: Optional[List[float]] = None,
    rain_24h: Optional[float] = None,
    rain_5d: Optional[float] = None,
    peak_rain_timing: str = "",
    tmax_24h: Optional[float] = None,
    tmin_24h: Optional[float] = None,
    tmax_5d: Optional[float] = None,
    tmin_5d: Optional[float] = None,
    gust_hourly_24h: Optional[List[float]] = None,
    gust_hourly_5d: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Return the fired weather actions, sorted most-severe first.

    Deterministic and auditable. Each item carries its own severity + timing.
    'title' and 'why' here are plain deterministic strings; the LLM step
    (phrase_todos) later rewrites them into short farmer-facing lines.
    """
    candidates = [
        decide_rain_action(rain_hourly_24h, rain_24h, rain_5d, peak_rain_timing),
        decide_temp_action(tmax_24h, tmin_24h, tmax_5d, tmin_5d),
        decide_wind_action(gust_hourly_24h, gust_hourly_5d),
    ]
    todos = [c for c in candidates if c is not None]

    # Sort by severity (desc). Stable sort keeps rain > temp > wind on ties,
    # matching the order candidates were built.
    todos.sort(key=lambda c: SEV.get(c["severity"], 0), reverse=True)

    for t in todos:
        t["_sev"] = SEV.get(t["severity"], 0)
    return todos


# ==============================================================================
# LLM PHRASING STEP (phrase only — never decides or scores)
# One combined call: rewrites every action into {title, hint} farmer one-liners.
# Falls back to deterministic strings if no key / call fails.
# ==============================================================================
def build_phrasing_prompt(todos: List[Dict[str, Any]], language: str = "English") -> str:
    lines = []
    for i, t in enumerate(todos):
        lines.append(
            f"[{i}] action_key={t['key']} | severity={t['severity']} | "
            f"draft_title=\"{t['title']}\" | reason=\"{t['why']}\""
        )
    actions_block = "\n".join(lines)

    return (
        f"You are helping write a farmer's 'What to do today' card for an Indian "
        f"agro-advisory app. Rewrite each action below into a short, plain, "
        f"actionable to-do in {language}.\n\n"
        f"ACTIONS:\n{actions_block}\n\n"
        f"RULES:\n"
        f"- For each action, output a 'title' (2-4 words, imperative, e.g. "
        f"'Clear field drainage', 'Skip pesticide spraying') and a 'hint' "
        f"(one sentence, max ~12 words, plain farmer language).\n"
        f"- The ACTION itself is stated plainly. Only forecast-dependent parts "
        f"(rain, heat, wind expected) may be hedged with 'expected'/'may'.\n"
        f"- Keep the meaning of the reason; do not invent new numbers, dates, "
        f"crops, or thresholds. Do not add any number that is not in the reason.\n"
        f"- No markdown, no emoji, no percentile talk.\n"
        f"- Return ONLY a JSON array, one object per action IN THE SAME ORDER, "
        f"each: {{\"index\": <int>, \"title\": <str>, \"hint\": <str>}}. "
        f"No preamble, no code fences."
    )


def _fallback_phrasing(todos: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out = []
    for t in todos:
        why = t["why"]
        hint = why[0].upper() + why[1:] + ("." if not why.endswith(".") else "")
        out.append({"title": t["title"], "hint": hint})
    return out


def _parse_llm_json(text: str, n: int) -> Optional[List[Dict[str, str]]]:
    if not text:
        return None
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?|```$", "", clean, flags=re.MULTILINE).strip()
    try:
        data = json.loads(clean)
    except Exception:
        m = re.search(r"\[.*\]", clean, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except Exception:
            return None
    if not isinstance(data, list) or len(data) != n:
        return None

    result: List[Optional[Dict[str, str]]] = [None] * n
    for obj in data:
        if not isinstance(obj, dict):
            return None
        idx = obj.get("index")
        title = obj.get("title")
        hint = obj.get("hint")
        if not isinstance(idx, int) or not (0 <= idx < n):
            return None
        if not isinstance(title, str) or not isinstance(hint, str):
            return None
        result[idx] = {"title": title.strip(), "hint": hint.strip()}
    if any(r is None for r in result):
        return None
    return result  # type: ignore


async def phrase_todos(
    todos: List[Dict[str, Any]],
    language: str = "English",
) -> List[Dict[str, Any]]:
    """Attach farmer-facing 'title' and 'hint' to each to-do. LLM phrases;
    deterministic fallback used when the providers fail / return bad output."""
    if not todos:
        return []

    phrased = None
    prompt = build_phrasing_prompt(todos, language=language)
    primary = get_primary_provider()
    fallback = get_fallback_provider()
    raw = None
    try:
        raw = await primary.generate_text(prompt=prompt, temperature=0.2)
    except Exception as e:
        logger.warning(f"Primary provider failed for weather to-dos phrasing: {e}")
        if fallback:
            try:
                raw = await fallback.generate_text(prompt=prompt, temperature=0.2)
            except Exception as fe:
                logger.error(f"Fallback provider failed for weather to-dos phrasing: {fe}")

    if raw:
        try:
            phrased = _parse_llm_json(raw, len(todos))
        except Exception:
            phrased = None

    if phrased is None:
        phrased = _fallback_phrasing(todos)

    result = []
    for t, p in zip(todos, phrased):
        result.append({
            "key": t["key"],
            "severity": t["severity"],
            "timing": t["timing"],
            "title": p["title"],
            "hint": p["hint"],
        })
    return result
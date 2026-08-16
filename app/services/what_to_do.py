"""
what_to_do.py
====================================================================
Aggregation layer for the "What To Do Today" dashboard card.

This is NOT a new recommendation engine. It is a deterministic, auditable
display layer on top of the EXISTING recommendation systems:

  * Pest & Disease  : crop-specific card (RAG-grounded, deterministic band)
                      -> best (highest-band, highest-priority) action.
  * Irrigation       : deterministic soil-moisture decision + insight.
  * Weather          : rain / temperature / wind to-dos (port of todo_card.py)
                      -> used only as a FALLBACK when a slot is empty.

Selection rules (deterministic, no LLM involved in the choice):
  1. At most TWO recommendations are returned.
  2. One slot is the best Pest & Disease action (highest severity band).
  3. One slot is the best Irrigation / Soil Moisture recommendation.
  4. When either slot is empty, it is filled with the best remaining
     non-duplicate recommendation (weather actions), else dropped.
  5. Same recommendation from multiple sources is never duplicated.
"""

from typing import Any, Dict, List, Optional, Sequence

from app.core.logging import logger
from app.services.weather_todos import build_todos, _fallback_phrasing


# Categories (stable keys the frontend uses for icons/labels)
PEST_CATEGORY = "pest"
IRRIGATION_CATEGORY = "irrigation"
WEATHER_CATEGORY = "weather"

# Normalised severity ranks (higher = more urgent). Each source's OWN severity
# vocabulary is preserved in the item; the rank only drives ordering.
PEST_SEV = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
IRR_SEV = {"now": 3, "schedule": 2, "drainage": 2, "none": 1}
WEATHER_SEV = {"unfavorable": 3, "cautionary": 2, "favorable": 1}

# Short deterministic title per irrigation action (display only).
_IRR_TITLE = {
    "now": "Irrigation recommended immediately",
    "schedule": "Irrigation recommended",
    "drainage": "Focus on drainage",
    "none": "Irrigation not needed",
}


# ==============================================================================
# WEATHER INPUT EXTRACTION (only what the payload actually carries)
# ==============================================================================
def _peak_rain_timing(forecast_days: Sequence[Any]) -> str:
    """Deterministic 'on DD Mon' phrase for the wettest forecast day, or ''."""
    if not forecast_days:
        return ""
    best = max(forecast_days, key=lambda d: getattr(d, "pcp_corrected", 0.0) or 0.0)
    if (getattr(best, "pcp_corrected", 0.0) or 0.0) < 2.5:
        return ""
    try:
        from datetime import datetime
        dt = datetime.strptime(str(best.date)[:10], "%Y-%m-%d")
        return f"on {dt.strftime('%d %b')}"
    except Exception:
        return ""


def _weather_inputs_from_request(request: Any) -> Dict[str, Any]:
    """Derive the weather to-do inputs from the frontend payload.

    Bias-corrected forecast (preferred) is used for 24h / 5-day values; the
    weather API daily series is the fallback. Hourly rain amount and hourly
    gusts are NOT carried by the request schema, so they are passed as None —
    the engine then uses only the daily bands it can genuinely support and
    never fires an action it has no data for.
    """
    daily = request.weatherData.daily

    forecast_days: List[Any] = []
    if request.forecastData and request.forecastData.forecast and request.forecastData.forecast.forecast:
        forecast_days = sorted(
            request.forecastData.forecast.forecast, key=lambda d: d.date
        )[:5]

    rain_24h = None
    if forecast_days:
        rain_24h = forecast_days[0].pcp_corrected
    elif daily.precipitation_sum:
        rain_24h = daily.precipitation_sum[0]

    rain_5d = None
    if forecast_days:
        rain_5d = sum(d.pcp_corrected for d in forecast_days)
    elif daily.precipitation_sum:
        rain_5d = sum(daily.precipitation_sum[:5])

    tmax_24h = daily.temperature_2m_max[0] if daily.temperature_2m_max else None
    tmin_24h = daily.temperature_2m_min[0] if daily.temperature_2m_min else None

    if forecast_days:
        tmax_5d = max(d.tmax_corrected for d in forecast_days)
        tmin_5d = min(d.tmin_corrected for d in forecast_days)
    else:
        tmax_5d = max(daily.temperature_2m_max[:5]) if daily.temperature_2m_max else None
        tmin_5d = min(daily.temperature_2m_min[:5]) if daily.temperature_2m_min else None

    return {
        "rain_hourly_24h": None,   # not available in the request schema
        "rain_24h": rain_24h,
        "rain_5d": rain_5d,
        "peak_rain_timing": _peak_rain_timing(forecast_days),
        "tmax_24h": tmax_24h,
        "tmin_24h": tmin_24h,
        "tmax_5d": tmax_5d,
        "tmin_5d": tmin_5d,
        "gust_hourly_24h": None,   # not available in the request schema
        "gust_hourly_5d": None,    # not available in the request schema
    }


def build_weather_todos_from_request(request: Any) -> List[Dict[str, Any]]:
    """Run the existing weather to-do engine on the request payload.

    Returns farmer-facing items ({key, severity, timing, title, hint}) sorted
    most-severe first. Phrasing uses the engine's DETERMINISTIC fallback so the
    card adds no extra LLM call; thresholds/severity are unchanged.
    """
    inputs = _weather_inputs_from_request(request)
    todos = build_todos(**inputs)
    if not todos:
        return []
    phrased = _fallback_phrasing(todos)
    out = []
    for t, p in zip(todos, phrased):
        out.append({
            "key": t["key"],
            "severity": t["severity"],
            "timing": t["timing"],
            "title": p["title"],
            "hint": p["hint"],
        })
    return out


# ==============================================================================
# ITEM BUILDERS (each source -> candidate items with a normalised rank)
# ==============================================================================
def _pest_items(card: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not card:
        return []
    band = str(card.get("risk", "")).upper()
    rank = PEST_SEV.get(band, 1)
    items = []
    for i, a in enumerate((card.get("actions") or [])[:3]):
        if not isinstance(a, dict):
            continue
        title = str(a.get("title", "")).strip()
        detail = str(a.get("detail", "")).strip()
        if not title:
            continue
        try:
            priority = max(1, int(a.get("priority", i + 1)))
        except (TypeError, ValueError):
            priority = i + 1
        items.append({
            "category": PEST_CATEGORY,
            "severity": band,
            "_rank": rank,
            "_priority": priority,
            "title": title,
            "hint": detail,
            "sources": a.get("sources", []) or [],
            "crop_name": card.get("crop_name"),
            "is_general": card.get("is_general", False),
            "_seq": len(items),
        })
    return items


def _split_irrigation_insight(insight: str, action: str) -> "tuple[str, str]":
    """Split the existing insight into {title, hint} at its stable template
    separator ('; ' for now/drainage/none, ', ' for schedule). Defensive: on
    any unexpected shape the whole line becomes the hint."""
    if not insight:
        return _IRR_TITLE.get(action, "Irrigation recommendation"), ""
    for sep in ("; ", ", "):
        if sep in insight:
            head, _, tail = insight.partition(sep)
            return head.strip(), tail.strip()
    return _IRR_TITLE.get(action, "Irrigation recommendation"), insight.strip()


def _irrigation_items(irr_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not irr_result:
        return []
    decision = irr_result.get("decision") or {}
    if not decision.get("available"):
        return []
    action = decision.get("action", "none")
    rank = IRR_SEV.get(action, 1)
    insight = irr_result.get("insight") or ""
    title, hint = _split_irrigation_insight(insight, action)
    return [{
        "category": IRRIGATION_CATEGORY,
        "severity": action,
        "_rank": rank,
        "title": title,
        "hint": hint,
        "sources": [],
        "crop_name": None,
        "is_general": None,
        "_seq": 0,
    }]


def _weather_items(todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for t in todos[:3]:
        sev = str(t.get("severity", "favorable"))
        items.append({
            "category": WEATHER_CATEGORY,
            "severity": sev,
            "_rank": WEATHER_SEV.get(sev, 1),
            "title": str(t.get("title", "")).strip(),
            "hint": str(t.get("hint", "")).strip(),
            "sources": [],
            "crop_name": None,
            "is_general": None,
            "_seq": len(items),
        })
    return items


# ==============================================================================
# SELECTION (deterministic — no LLM involved in the choice)
# ==============================================================================
def _is_dup(item: Dict[str, Any], chosen: List[Dict[str, Any]]) -> bool:
    for c in chosen:
        if c.get("category") != item.get("category"):
            continue
        a = str(c.get("title", "")).strip().lower()
        b = str(item.get("title", "")).strip().lower()
        if a and b and a == b:
            return True
        ah = str(c.get("hint", "")).strip().lower()
        bh = str(item.get("hint", "")).strip().lower()
        if ah and bh and ah == bh:
            return True
    return False


def _public(item: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(item)
    out.pop("_rank", None)
    out.pop("_priority", None)
    out.pop("_seq", None)
    return out


def select_what_to_do(
    pest_card: Optional[Dict[str, Any]],
    irr_result: Optional[Dict[str, Any]],
    weather_todos: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Deterministically select at most TWO recommendations.

    Priority:
      1. best Pest & Disease action (highest band, then per-action priority)
      2. best Irrigation / Soil Moisture recommendation
      3. missing slots are filled with the best remaining non-duplicate
         recommendation (weather to-dos), otherwise dropped.
    """
    pest_items = sorted(
        _pest_items(pest_card),
        key=lambda it: (-it["_rank"], it["_priority"], it["_seq"]),
    )
    irr_items = sorted(
        _irrigation_items(irr_result),
        key=lambda it: (-it["_rank"], it["_seq"]),
    )
    weather_items = sorted(
        _weather_items(weather_todos),
        key=lambda it: (-it["_rank"], it["_seq"]),
    )

    chosen: List[Dict[str, Any]] = []

    def add(item: Optional[Dict[str, Any]]) -> None:
        if item is None or len(chosen) >= 2:
            return
        if _is_dup(item, chosen):
            return
        chosen.append(item)

    add(pest_items[0] if pest_items else None)
    add(irr_items[0] if irr_items else None)

    # Fill any still-empty slot with the best remaining (weather) item.
    for it in weather_items:
        if len(chosen) >= 2:
            break
        add(it)

    result = [_public(item) for item in chosen]
    logger.info(
        "What-to-do selection | pest=%d irr=%d weather=%d -> %d item(s): %s",
        len(pest_items), len(irr_items), len(weather_items), len(result),
        [f"{r.get('category')}:{r.get('title')}" for r in result],
    )
    return result
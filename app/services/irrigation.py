"""
Irrigation insight engine — deterministic, farmer-facing one-liner for the
"SOIL MOISTURE" card.

Logic is copied VERBATIM from the project's reference engine
(`Irrigation_comment.py` at the repo root): the irrigation decision
(`decide_irrigation`) walks the model's OWN daily soil-moisture percentile
series; rainfall is already baked into that series by the hydrological model,
so rain NEVER changes the verdict — it only annotates the "why".

Design contract (auditable):
  * The decision is deterministic. Thresholds and helpers here are identical
    to `Irrigation_comment.py`:
      - IRR_TRIGGERS  (sensitive / normal / tolerant plan+urgent thresholds)
      - IRR_WET_PCT   (>= 80.0 -> wet soil, drainage instead of irrigation)
      - IRR_FALLBACK_SENSITIVITY  ('sensitive' — safe direction on a drying field)
      - soil_word()          percentile -> category band
      - _irr_date_phrase()   day offset -> DD/MM/YYYY + relative phrase
      - _irr_peak_rain_phrase()  wettest day >= 2.5 mm inside a window
      - decide_irrigation()  the full decision walk
  * Only the crop-stage -> sensitivity LABEL is resolved deterministically here
    (same 3 buckets the reference LLM would pick from). No date or verdict is
    invented; the LLM never calculates anything for this field.
  * Numbers (percentiles / w_frac) are NEVER emitted — category words only.
  * If the required data is missing the builder returns "" (callers map to null)
    so the frontend simply hides the message.
"""

from typing import Any, Dict, List, Optional, Sequence

from app.core.logging import logger


# ----------------------------- TUNABLES (mirror Irrigation_comment.py) --------
# Dryness triggers as SOIL-MOISTURE PERCENTILES.
#   PLAN   : soil approaching stress -> schedule irrigation on/after this day.
#   URGENT : soil actively stressed  -> irrigate now (already overdue).
# One pair per crop-stage water-sensitivity class. Higher = act earlier (wetter).
IRR_TRIGGERS = {
    "sensitive": {"plan": 40.0, "urgent": 30.0},  # flowering, grain/boll fill, tuber bulking
    "normal":    {"plan": 30.0, "urgent": 20.0},  # vegetative / established
    "tolerant":  {"plan": 20.0, "urgent": 10.0},  # maturity / ripening / dry-down
}

# Soil-moisture percentile at/above which the field counts as "already wet":
#   -> no irrigation, drainage focus (the wet card in the mockup).
IRR_WET_PCT = 80.0

# Fallback sensitivity when the stage label is missing / unparseable.
# 'sensitive' = act early = safe direction on a drying field.
IRR_FALLBACK_SENSITIVITY = "sensitive"
# ------------------------------------------------------------------------------


# ==============================================================================
# SOIL MOISTURE CATEGORY (ClimateAdapt / project percentile bands)
# ==============================================================================
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


# ==============================================================================
# IRRIGATION DECISION ENGINE  (for the "SOIL MOISTURE" card one-liner)
# Identical to Irrigation_comment.py — reuse, never fork.
# ==============================================================================

def _irr_date_phrase(day_idx: int, start_date: str) -> Dict[str, str]:
    """Return {'date': 'DD/MM/YYYY'|'', 'rel': relative phrase} for a day offset."""
    from datetime import datetime, timedelta
    out = {"date": "", "rel": ""}
    if start_date:
        try:
            _st = datetime.strptime(start_date[:10], "%Y-%m-%d")
            d = _st + timedelta(days=day_idx)
            out["date"] = d.strftime("%d/%m/%Y")
        except Exception:
            pass
    if day_idx <= 0:
        out["rel"] = "today"
    elif day_idx == 1:
        out["rel"] = "tomorrow"
    else:
        out["rel"] = f"in about {day_idx} days"
    return out


def _irr_date_rel(date_ddmmyy: str, reference_date: str) -> str:
    """Relative position of a DD/MM/YYYY date vs the reference 'today' date:
        diff 0 -> 'today', diff 1 -> 'tomorrow', otherwise -> 'later'.
    The reference is the current observation date (weatherData.current.time),
    falling back to the series start date. Never hardcoded."""
    if not date_ddmmyy:
        return "later"
    try:
        from datetime import datetime
        d = datetime.strptime(date_ddmmyy, "%d/%m/%Y")
        ref = datetime.strptime((reference_date or "")[:10], "%Y-%m-%d")
        diff = (d - ref).days
    except Exception:
        return "later"
    if diff == 0:
        return "today"
    if diff == 1:
        return "tomorrow"
    return "later"


def _irr_when_phrase(date_ddmmyy: str, reference_date: str, later_prefix: str = "") -> str:
    """Dynamic date wording (phrasing ONLY — never affects the decision):
        today    -> 'today (DD/MM/YYYY)'
        tomorrow -> 'tomorrow (DD/MM/YYYY)'
        later    -> '{later_prefix} DD/MM/YYYY'  (later_prefix is '' for the
                    irrigation date, 'on' for the expected-rain date)
    'today' is the reference date (weather current observation date). Returns
    '' when no date is available."""
    if not date_ddmmyy:
        return ""
    rel = _irr_date_rel(date_ddmmyy, reference_date)
    if rel == "today":
        return f"today ({date_ddmmyy})"
    if rel == "tomorrow":
        return f"tomorrow ({date_ddmmyy})"
    return f"{later_prefix} {date_ddmmyy}".strip()


def _irr_peak_rain_phrase(rain_series, start_date, lo, hi):
    """Find the wettest day within [lo, hi] (inclusive) of the rain series and
    return a short 'around DD/MM/YYYY' phrase, or '' if no meaningful rain.
    Used ONLY to explain WHY soil recovers/stays wet — never to decide."""
    if not rain_series:
        return ""
    lo = max(0, lo)
    hi = min(len(rain_series) - 1, hi)
    if hi < lo:
        return ""
    best_i, best_v = lo, rain_series[lo]
    for i in range(lo, hi + 1):
        v = rain_series[i]
        if v is not None and (best_v is None or v > best_v):
            best_i, best_v = i, v
    if best_v is None or best_v < 2.5:   # below IMD "light rain" floor -> not worth citing
        return ""
    ph = _irr_date_phrase(best_i, start_date)
    return f"around {ph['date']}" if ph["date"] else f"{ph['rel']}"


def decide_irrigation(
    sm_series: Optional[List[float]],
    rain_series: Optional[List[float]] = None,
    sensitivity: str = "normal",
    start_date: str = "",
    horizon_days: Optional[int] = None,
) -> Dict[str, Any]:
    """Deterministic irrigation verdict from the DAILY soil-moisture percentile
    series (rain already reflected in sm_series by the hydrological model).

    Returns a structured, number-free decision:
      {
        'available': bool,
        'sensitivity': 'sensitive'|'normal'|'tolerant',
        'action': 'drainage'|'now'|'schedule'|'none',
        'next_irrigation_date': 'DD/MM/YYYY' | '',   # only a REAL trigger crossing
        'days_until': int | None,
        'horizon_days': int,
        'soil_word_now': <category>,
        'soil_word_at_trigger': <category> | '',
        'reason_code': 'already_saturated'|'already_dry'|'drying_no_recovery'
                       |'recovers_before_stress'|'steady_comfortable',
        'why': <short cause phrase, may cite a rain date>,
        'hint': <one compact line the LLM rephrases from>,
      }
    """
    if not sm_series or len(sm_series) < 1:
        return {"available": False, "action": "none", "hint": ""}

    sens = (sensitivity or "normal").strip().lower()
    if sens not in IRR_TRIGGERS:
        sens = IRR_FALLBACK_SENSITIVITY
    plan = IRR_TRIGGERS[sens]["plan"]
    urgent = IRR_TRIGGERS[sens]["urgent"]

    n = len(sm_series)
    N = horizon_days if horizon_days else n
    now = sm_series[0]
    now_word = soil_word(now)

    def _pack(action, reason_code, why, trigger_idx=None, trigger_word=""):
        date_ph = _irr_date_phrase(trigger_idx, start_date) if trigger_idx is not None else {"date": "", "rel": ""}
        days_until = trigger_idx if trigger_idx is not None else None
        # Build the compact hint the LLM will rephrase (category words + timing only).
        # The ACTION is stated plainly; only the forecast-dependent clauses
        # (soil trajectory, rain, recharge) are hedged (expected / may / likely).
        if action == "drainage":
            hint = (f"irrigation not needed for the next {N} days; soil looks {now_word} "
                    f"(appears saturated) — focus on drainage")
        elif action == "now":
            hint = (f"irrigation needed soon; soil looks {now_word} and little recovery "
                    f"is expected within the next {N} days")
        elif action == "schedule":
            when = date_ph["date"] or date_ph["rel"]
            tail = f" ({why})" if why else ""
            hint = (f"irrigation needed around {when}, when soil moisture is expected to "
                    f"approach {trigger_word}{tail}")
        else:  # none
            tail = f" ({why})" if why else ""
            hint = (f"irrigation not needed for the next {N} days; soil is expected to "
                    f"stay adequate (currently {now_word}){tail}")
        return {
            "available": True,
            "sensitivity": sens,
            "action": action,
            "next_irrigation_date": date_ph["date"],
            "days_until": days_until,
            "horizon_days": N,
            "soil_word_now": now_word,
            "soil_word_at_trigger": trigger_word,
            "reason_code": reason_code,
            "why": why,
            "hint": hint,
        }

    # ---- Branch 1: already wet -> drainage, no irrigation ---------------------
    if now >= IRR_WET_PCT:
        why = _irr_peak_rain_phrase(rain_series, start_date, 0, min(4, n - 1))
        why = f"expected rain {why}".strip() if why else ""
        return _pack("drainage", "already_saturated", why)

    # ---- Branch 2: already at/below the urgent trigger -> irrigate now --------
    if now <= urgent:
        return _pack("now", "already_dry", "")

    # ---- Walk the series for the FIRST day soil reaches the PLAN trigger ------
    first_cross = None
    for i in range(1, n):
        if sm_series[i] <= plan:
            first_cross = i
            break

    if first_cross is not None:
        # Does it dip to PLAN but stay ABOVE urgent, then recover? Still schedule
        # at the crossing day (the moment stress begins). This is a real date.
        trig_word = soil_word(sm_series[first_cross])
        # Explain WHY: is there rain shortly AFTER that lifts it back?
        why = _irr_peak_rain_phrase(rain_series, start_date, first_cross, min(first_cross + 4, n - 1))
        why = f"expected rain {why} may recharge it" if why else ""
        return _pack("schedule", "drying_no_recovery" if not why else "recovers_before_stress",
                     why, trigger_idx=first_cross, trigger_word=trig_word)

    # ---- Never reaches PLAN over the horizon -> comfortable, no irrigation ----
    # Was there a trough (dip that recovered) worth explaining? cite rain if so.
    trough_i = min(range(n), key=lambda k: sm_series[k])
    why = ""
    if trough_i > 0 and sm_series[trough_i] < now:
        rain_ph = _irr_peak_rain_phrase(rain_series, start_date, trough_i, min(trough_i + 4, n - 1))
        if rain_ph:
            why = f"a brief dip may recover if expected rain {rain_ph} arrives"
    return _pack("none", "steady_comfortable", why)


# ==============================================================================
# IRRIGATION INSIGHT BUILDER (farmer-facing one-liner)
# ==============================================================================

def classify_water_sensitivity(crop_stage: Optional[str]) -> str:
    """Map the deterministic crop-stage label to one of the 3 engine buckets.

    Same buckets the reference engine's LLM step would select from
    (sensitive / normal / tolerant). 'unknown' falls back to the safe direction.
    """
    s = (crop_stage or "").strip().lower()
    if not s or s in ("unknown", "none"):
        return IRR_FALLBACK_SENSITIVITY
    sensitive_keys = ("flower", "reproduct", "boll", "grain fill", "pegging", "tuber", "bulk")
    tolerant_keys = ("matur", "harvest", "ripen", "dry", "post-harvest", "post harvest")
    if any(k in s for k in sensitive_keys):
        return "sensitive"
    if any(k in s for k in tolerant_keys):
        return "tolerant"
    return "normal"


def extract_irrigation_series(
    sm_records: Sequence[Any],
    forecast_rows: Sequence[Any],
) -> Optional[Dict[str, Any]]:
    """Build the two aligned daily series consumed by decide_irrigation.

    sm_records   : records with .date / .sm_percentile / .is_forecast
    forecast_rows: records with .date / .pcp_corrected

    The soil-moisture series starts at the latest OBSERVED condition ("now")
    and continues through the forecast days (cold-start: forecast-only).
    Rain is aligned to the same calendar so every day index matches.

    Returns {
      'sm_series': [...], 'rain_series': [...], 'start_date': 'YYYY-MM-DD'
    } or None when the data cannot support a decision.
    """
    records = [r for r in sm_records if getattr(r, "sm_percentile", None) is not None]
    if not records:
        return None
    records = sorted(records, key=lambda r: r.date)
    obs = [r for r in records if getattr(r, "is_forecast", 1) == 0]
    fc = [r for r in records if getattr(r, "is_forecast", 1) == 1]
    if not fc:
        return None

    head = [obs[-1]] if obs else fc[:1]
    series_records = head + fc

    pcp_map = {f.date: getattr(f, "pcp_corrected", 0.0) for f in forecast_rows}

    return {
        "sm_series": [r.sm_percentile for r in series_records],
        "rain_series": [pcp_map.get(r.date) for r in series_records],
        "start_date": series_records[0].date,
    }


def _irr_duration_phrase(days: int) -> str:
    """Wording-only rule for the no-irrigation branches.

    When the deterministic decision says irrigation is not needed for MORE
    than 7 days, render the duration as "at least a week" instead of exposing
    the raw forecast-day count. For <= 7 days keep the precise "the next N days".

    This is PHRASING ONLY — the decision engine's horizon_days is unchanged.
    """
    return "at least a week" if (days or 0) > 7 else f"the next {days} days"


def build_irrigation_insight(
    sm_series: Optional[List[float]],
    rain_series: Optional[List[float]] = None,
    sensitivity: str = "normal",
    start_date: str = "",
    horizon_days: Optional[int] = None,
    reference_date: Optional[str] = None,
) -> str:
    """Render the deterministic farmer-facing irrigation insight.

    Reuses ONLY the outputs of decide_irrigation (the existing engine). The
    phrasing follows the agreed templates:
      1. schedule + meaningful future rain ->
         "Irrigation recommended <when>, as expected rain <rain-when> may later
          recharge <category> soil moisture levels."
      2. schedule, no meaningful future rain -> "Irrigation recommended <when>."
      3. now  -> irrigation recommended immediately (existing 'now' action).
      4. drainage -> focus on drainage (existing saturated branch).
      5. none -> irrigation not needed (existing comfortable branch).
    <when> / <rain-when> are dynamic date phrases from _irr_when_phrase:
      today -> "today (DD/MM/YYYY)", tomorrow -> "tomorrow (DD/MM/YYYY)",
      later -> "DD/MM/YYYY" (irrigation) / "on DD/MM/YYYY" (rain).
    'today' is the reference_date (the current observation date); when not
    supplied it falls back to the series start date.
    For the no-irrigation branches (4 & 5) the duration is phrased by
    _irr_duration_phrase: > 7 days -> "at least a week", <= 7 -> "the next N days".

    Returns "" when the decision is unavailable (callers map to null).
    Numbers / percentiles / w_frac are never emitted.
    """
    if not sm_series or len(sm_series) < 1:
        return ""

    ref = reference_date or start_date

    decision = decide_irrigation(
        sm_series,
        rain_series,
        sensitivity=sensitivity,
        start_date=start_date,
        horizon_days=horizon_days,
    )
    if not decision.get("available"):
        return ""

    action = decision["action"]
    now_word = decision["soil_word_now"]
    N = decision["horizon_days"]

    if action == "drainage":
        return (
            f"Irrigation not needed for {_irr_duration_phrase(N)}; soil looks {now_word} "
            f"and appears saturated — focus on drainage."
        )

    if action == "now":
        return (
            f"Irrigation recommended immediately; soil looks {now_word} and little "
            f"recovery is expected soon."
        )

    if action == "schedule":
        date = decision["next_irrigation_date"]
        idx = decision["days_until"]
        rain_date = ""
        if idx is not None and rain_series:
            lo = max(0, idx)
            hi = min(len(rain_series) - 1, idx + 4)
            rain_ph = _irr_peak_rain_phrase(rain_series, start_date, lo, hi)
            if rain_ph.startswith("around "):
                rain_date = rain_ph[len("around "):].strip()

        if date:
            when = _irr_when_phrase(date, ref)
            if rain_date:
                rain_when = _irr_when_phrase(rain_date, ref, later_prefix="on")
                return (
                    f"Irrigation recommended {when}, as expected rain {rain_when} "
                    f"may later recharge {now_word} soil moisture levels."
                )
            return f"Irrigation recommended {when}."

        rel = _irr_date_phrase(idx, start_date)["rel"] if idx is not None else "today"
        if rain_date:
            rain_when = _irr_when_phrase(rain_date, ref, later_prefix="on")
            return (
                f"Irrigation recommended {rel}, as expected rain {rain_when} "
                f"may later recharge {now_word} soil moisture levels."
            )
        return f"Irrigation recommended {rel}."

    # action == "none"
    return (
        f"Irrigation not needed for {_irr_duration_phrase(N)}; soil is expected to stay "
        f"adequate (currently {now_word})."
    )


def compute_irrigation_insight_for_request(request: Any, crop_stage: str = "") -> Optional[str]:
    """Top-level helper used by the advisory endpoint.

    Returns the ready-to-send string, or None when the required data (soil
    moisture / rainfall series) is unavailable so the frontend hides the message.
    """
    from app.models.schemas import AIAdvisoryRequest  # local import to avoid cycles

    fd = request.forecastData
    if (
        fd is None
        or fd.soil_moisture is None
        or fd.forecast is None
        or not fd.soil_moisture.soil_moisture
        or not fd.forecast.forecast
    ):
        return None

    series = extract_irrigation_series(
        sm_records=fd.soil_moisture.soil_moisture,
        forecast_rows=fd.forecast.forecast,
    )
    if not series:
        return None

    sensitivity = classify_water_sensitivity(crop_stage)

    # Reference 'today' = the current weather observation date; falls back to
    # the series start date when the observation is unavailable.
    reference_date = ""
    try:
        _cur = getattr(request, "weatherData", None)
        if _cur is not None and getattr(_cur, "current", None) is not None:
            reference_date = str(getattr(_cur.current, "time", "") or "")[:10]
    except Exception:
        reference_date = ""
    if not reference_date:
        reference_date = series["start_date"]

    # TEMPORARY DEV LOGGING — remove after verification.
    # Recomputes the decision (O(n), deterministic) so we can log the audit
    # fields the user asked for. Never changes the verdict.
    decision = decide_irrigation(
        series["sm_series"],
        series["rain_series"],
        sensitivity=sensitivity,
        start_date=series["start_date"],
    )
    logger.info(
        "Irrigation insight | decision=%s reason=%s next_date=%s days_until=%s "
        "soil_now=%s rain_why=%r sensitivity=%s ref_today=%s",
        decision.get("action"),
        decision.get("reason_code"),
        decision.get("next_irrigation_date"),
        decision.get("days_until"),
        decision.get("soil_word_now"),
        decision.get("why"),
        sensitivity,
        reference_date,
    )

    insight = build_irrigation_insight(
        sm_series=series["sm_series"],
        rain_series=series["rain_series"],
        sensitivity=sensitivity,
        start_date=series["start_date"],
        reference_date=reference_date,
    )
    logger.info("Generated irrigation insight: %r", insight)
    return insight or None
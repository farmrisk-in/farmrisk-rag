"""
AdvisoryContextBuilder — deterministic aggregation of the full AIAdvisoryRequest
into a compact AdvisoryContext.

RULES:
- No I/O, no LLM calls, no network requests.
- All arithmetic (totals, averages, trends) happens here.
- Gemini receives only the compact AdvisoryContext — never raw arrays.
"""

import hashlib
from datetime import datetime
from typing import List, Optional

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import (
    AIAdvisoryRequest,
    AdvisoryContext,
    AvailabilityFlags,
    CropContext,
    CurrentWeatherSummary,
    ForecastSummary,
    HourlyWeatherSummary,
    LightningSummary,
    LocationSummary,
    SoilMoistureSummary,
    WeatherApiSummary,
)


class AdvisoryContextBuilder:
    """Convert a raw AIAdvisoryRequest into a compact, validated AdvisoryContext."""

    def build(self, request: AIAdvisoryRequest) -> AdvisoryContext:
        t0 = datetime.now()

        availability = self._build_availability(request)
        location = self._build_location(request)
        crop_ctx = self._build_crop_context(request)
        current_wx = self._build_current_weather(request)
        forecast_summary = self._build_forecast_summary(request) if availability.corrected_forecast_available else None
        weather_api = self._build_weather_api_summary(request)
        hourly = self._build_hourly_summary(request)
        soil = self._build_soil_moisture_summary(request) if availability.soil_moisture_available else None
        lightning = self._build_lightning_summary(request) if availability.lightning_available else None

        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(
            f"ContextBuilder completed in {elapsed:.3f}s | "
            f"crop={crop_ctx.crop_name} season={crop_ctx.season} stage={crop_ctx.crop_stage} | "
            f"forecast={availability.corrected_forecast_available} "
            f"soil={availability.soil_moisture_available} "
            f"lightning={availability.lightning_available}"
        )

        return AdvisoryContext(
            location=location,
            crop_context=crop_ctx,
            current_weather=current_wx,
            forecast_summary=forecast_summary,
            weather_api_summary=weather_api,
            hourly_summary=hourly,
            soil_moisture_summary=soil,
            lightning_summary=lightning,
            availability=availability,
        )

    # ------------------------------------------------------------------
    # AVAILABILITY FLAGS
    # ------------------------------------------------------------------

    def _build_availability(self, req: AIAdvisoryRequest) -> AvailabilityFlags:
        fd = req.forecastData
        return AvailabilityFlags(
            weather_available=True,  # weather is always present if we reach here
            corrected_forecast_available=(
                fd is not None
                and fd.forecast.success
                and len(fd.forecast.forecast) > 0
            ),
            soil_moisture_available=(
                fd is not None
                and fd.soil_moisture.success
                and len(fd.soil_moisture.soil_moisture) > 0
            ),
            lightning_available=req.weatherData.lightning is not None,
            calendar_available=(
                req.calendarData.success
                and len(req.calendarData.calendar) > 0
            ),
        )

    # ------------------------------------------------------------------
    # LOCATION
    # ------------------------------------------------------------------

    def _build_location(self, req: AIAdvisoryRequest) -> LocationSummary:
        return LocationSummary(
            lat=req.location.lat,
            lng=req.location.lng,
            name=req.location.name,
            display_name=req.location.displayName,
            state=req.calendarData.state,
            district=req.calendarData.district,
        )

    # ------------------------------------------------------------------
    # CROP CONTEXT: name, season, stage
    # ------------------------------------------------------------------

    def _build_crop_context(self, req: AIAdvisoryRequest) -> CropContext:
        crop_id = req.cropId.lower().strip()
        calendar = req.calendarData.calendar

        # Find the matching calendar entry
        matched_entry = None
        for entry in calendar:
            if entry.crop.lower().replace(" ", "").replace("-", "") == crop_id.replace(" ", "").replace("-", ""):
                matched_entry = entry
                break
            # Also try simple contains
            if crop_id in entry.crop.lower() or entry.crop.lower() in crop_id:
                matched_entry = entry
                break

        # Resolve display name
        if crop_id == "general":
            crop_name = "General (All Crops)"
        elif matched_entry:
            crop_name = matched_entry.crop
        else:
            # Fall back to title-casing the cropId
            crop_name = crop_id.title()

        # Resolve season
        if matched_entry:
            season = matched_entry.season
        else:
            season = self._month_based_season()

        # Estimate crop stage
        if matched_entry:
            crop_stage = self._estimate_crop_stage(
                matched_entry.sowFromMon,
                matched_entry.sowToMon,
                matched_entry.harvFromMon,
                matched_entry.harvToMon,
            )
        else:
            crop_stage = "unknown"

        return CropContext(
            crop_id=req.cropId,
            crop_name=crop_name,
            season=season,
            crop_stage=crop_stage,
        )

    def _month_based_season(self) -> str:
        month = datetime.now().month
        return "Kharif" if 6 <= month <= 10 else "Rabi"

    def _estimate_crop_stage(
        self,
        sow_from: Optional[int],
        sow_to: Optional[int],
        harv_from: Optional[int],
        harv_to: Optional[int],
    ) -> str:
        if sow_from is None:
            return "unknown"

        M = datetime.now().month

        def months_in_range(start: int, end: int) -> List[int]:
            if start is None or end is None:
                return []
            months = []
            m = start
            for _ in range(13):
                months.append(m)
                if m == end:
                    break
                m = m % 12 + 1
            return months

        sow_months = months_in_range(sow_from, sow_to or sow_from)
        harv_months = months_in_range(harv_from, harv_to or harv_from) if harv_from else []

        if M in sow_months:
            return "sowing"

        if harv_months and M in harv_months:
            return "maturity/harvesting"

        # Growing season between sow end and harvest start
        if sow_to and harv_from:
            next_after_sow = sow_to % 12 + 1
            prev_before_harv = (harv_from - 2) % 12 + 1
            growing = months_in_range(next_after_sow, prev_before_harv)
            if M in growing and growing:
                idx = growing.index(M)
                total = len(growing)
                if total == 0:
                    return "vegetative"
                third = max(total // 3, 1)
                if idx < third:
                    return "early establishment"
                elif idx < 2 * third:
                    return "vegetative"
                else:
                    return "flowering/reproductive"

        # Pre-sowing: within 3 months before sow_from
        pre_sow = []
        m = (sow_from - 2) % 12 + 1
        for _ in range(3):
            pre_sow.append(m)
            m = (m - 2) % 12 + 1
        if M in pre_sow:
            return "pre-sowing"

        # Post-harvest: within 2 months after harv_to
        if harv_to:
            post = []
            m = harv_to % 12 + 1
            for _ in range(2):
                post.append(m)
                m = m % 12 + 1
            if M in post:
                return "post-harvest"

        return "unknown"

    # ------------------------------------------------------------------
    # CURRENT WEATHER
    # ------------------------------------------------------------------

    def _build_current_weather(self, req: AIAdvisoryRequest) -> CurrentWeatherSummary:
        c = req.weatherData.current
        condition_en = c.condition.get("en", "") if isinstance(c.condition, dict) else ""
        return CurrentWeatherSummary(
            observation_time=c.time,
            temperature_c=round(c.temperature_2m, 1),
            apparent_temperature_c=round(c.apparent_temperature, 1),
            relative_humidity_percent=round(c.relative_humidity_2m, 1),
            precipitation_mm=round(c.precipitation, 2),
            wind_speed_kmh=round(c.wind_speed_10m, 1),
            wind_gusts_kmh=round(c.wind_gusts_10m, 1),
            cloud_cover_percent=round(c.cloud_cover, 1),
            weather_condition=condition_en,
        )

    # ------------------------------------------------------------------
    # CORRECTED FORECAST SUMMARY
    # ------------------------------------------------------------------

    def _build_forecast_summary(self, req: AIAdvisoryRequest) -> Optional[ForecastSummary]:
        if req.forecastData is None:
            return None
        raw_days = req.forecastData.forecast.forecast
        if not raw_days:
            return None

        # Sort chronologically — defensive against out-of-order API responses
        days = sorted(raw_days, key=lambda row: row.date)

        pcp = [d.pcp_corrected for d in days]
        tmax = [d.tmax_corrected for d in days]
        tmin = [d.tmin_corrected for d in days]

        total_rain = round(sum(pcp), 2)
        rainy = sum(1 for p in pcp if p >= settings.RAINY_DAY_THRESHOLD_MM)
        max_daily = round(max(pcp), 2)
        avg_daily = round(total_rain / len(pcp), 2)

        # Rainfall pattern classification
        if total_rain < 5:
            pattern = "Dry"
        elif total_rain < 20:
            pattern = "Mostly dry with trace rainfall"
        elif total_rain < 50:
            pattern = "Intermittent light rainfall"
        elif total_rain < 100:
            pattern = "Moderate rainfall"
        elif total_rain < 200:
            pattern = "Widespread rainfall"
        else:
            pattern = "Heavy/extreme rainfall event"

        # Heavy rain days add detail
        heavy_days = sum(1 for p in pcp if p >= settings.HEAVY_RAIN_THRESHOLD_MM)
        if heavy_days > 0:
            pattern += f" (with {heavy_days} heavy rain day{'s' if heavy_days > 1 else ''})"

        # Deterministic fingerprint for caching (rounded to 1 dp)
        fingerprint_str = "|".join(f"{round(p, 1)}" for p in pcp)
        fingerprint = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:12]

        return ForecastSummary(
            forecast_start_date=days[0].date,
            forecast_end_date=days[-1].date,
            forecast_days=len(days),
            total_rainfall_mm=total_rain,
            average_daily_rainfall_mm=avg_daily,
            maximum_daily_rainfall_mm=max_daily,
            rainy_days=rainy,
            dry_days=len(days) - rainy,
            minimum_temperature_c=round(min(tmin), 1),
            maximum_temperature_c=round(max(tmax), 1),
            average_min_temperature_c=round(sum(tmin) / len(tmin), 1),
            average_max_temperature_c=round(sum(tmax) / len(tmax), 1),
            rainfall_pattern=pattern,
            forecast_fingerprint=fingerprint,
        )

    # ------------------------------------------------------------------
    # WEATHER API DAILY (secondary context)
    # ------------------------------------------------------------------

    def _build_weather_api_summary(self, req: AIAdvisoryRequest) -> WeatherApiSummary:
        d = req.weatherData.daily
        total = round(sum(d.precipitation_sum), 2) if d.precipitation_sum else 0.0
        min_t = round(min(d.temperature_2m_min), 1) if d.temperature_2m_min else 0.0
        max_t = round(max(d.temperature_2m_max), 1) if d.temperature_2m_max else 0.0
        start_date = d.time[0] if d.time else ""
        end_date = d.time[-1] if d.time else ""
        return WeatherApiSummary(
            api_start_date=start_date,
            api_end_date=end_date,
            api_total_rainfall_mm=total,
            api_min_temp_c=min_t,
            api_max_temp_c=max_t,
        )

    # ------------------------------------------------------------------
    # HOURLY (derived only, first 24 hours)
    # ------------------------------------------------------------------

    def _build_hourly_summary(self, req: AIAdvisoryRequest) -> HourlyWeatherSummary:
        h = req.weatherData.hourly
        precip_probs = h.precipitation_probability[:24] if h.precipitation_probability else []
        wind_speeds = h.wind_speed_10m[:24] if h.wind_speed_10m else []
        return HourlyWeatherSummary(
            max_precip_probability_next24h=round(max(precip_probs), 1) if precip_probs else 0.0,
            max_wind_speed_next24h=round(max(wind_speeds), 1) if wind_speeds else 0.0,
        )

    # ------------------------------------------------------------------
    # SOIL MOISTURE SUMMARY
    # ------------------------------------------------------------------

    def _build_soil_moisture_summary(self, req: AIAdvisoryRequest) -> Optional[SoilMoistureSummary]:
        if req.forecastData is None:
            return None
        records = req.forecastData.soil_moisture.soil_moisture
        if not records:
            return None

        # Sort all records chronologically — defensive against out-of-order API responses
        sorted_records = sorted(records, key=lambda r: r.date)

        historical = [r for r in sorted_records if r.is_forecast == 0]
        forecast_rows = [r for r in sorted_records if r.is_forecast == 1]

        if not forecast_rows:
            return None

        # latest_row = the most recent OBSERVED condition just before the forecast period.
        # If no historical rows exist (cold start), fall back to the first forecast row
        # as a proxy, but this is explicitly a degraded reading.
        latest_row = historical[-1] if historical else forecast_rows[0]

        pcts = [r.sm_percentile for r in forecast_rows]
        fracs = [r.w_frac for r in forecast_rows]

        # Trend is calculated from chronological start → end of the forecast period
        start_pct = pcts[0]
        end_pct = pcts[-1]
        delta = end_pct - start_pct

        if delta > settings.SOIL_MOISTURE_TREND_TOLERANCE:
            trend = "increasing"
        elif delta < -settings.SOIL_MOISTURE_TREND_TOLERANCE:
            trend = "decreasing"
        else:
            trend = "stable"

        sm_block = req.forecastData.soil_moisture  # type: ignore[union-attr]
        return SoilMoistureSummary(
            latest_w_frac=round(latest_row.w_frac, 4),
            latest_sm_percentile=round(latest_row.sm_percentile, 2),
            forecast_start_date=forecast_rows[0].date,
            forecast_end_date=forecast_rows[-1].date,
            forecast_min_w_frac=round(min(fracs), 4),
            forecast_max_w_frac=round(max(fracs), 4),
            forecast_average_w_frac=round(sum(fracs) / len(fracs), 4),
            forecast_min_percentile=round(min(pcts), 2),
            forecast_max_percentile=round(max(pcts), 2),
            forecast_average_percentile=round(sum(pcts) / len(pcts), 2),
            start_percentile=round(start_pct, 2),
            end_percentile=round(end_pct, 2),
            soil_moisture_trend=trend,
            cold_start=sm_block.cold_start,
            days_computed=sm_block.days_computed,
        )

    # ------------------------------------------------------------------
    # LIGHTNING SUMMARY
    # ------------------------------------------------------------------

    def _build_lightning_summary(self, req: AIAdvisoryRequest) -> Optional[LightningSummary]:
        lg = req.weatherData.lightning
        if lg is None:
            return None
        return LightningSummary(score=lg.score, category=lg.category)

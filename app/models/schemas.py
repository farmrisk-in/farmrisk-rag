"""
Pydantic schemas for the FarmRisk AI Advisory API.

Hierarchy:
  AIAdvisoryRequest  — Full frontend payload received at POST /api/advisory
  AdvisoryContext    — Compact deterministic context produced by ContextBuilder
  AdvisoryResponse   — Returned to the frontend
  TranslationResult  — Internal helper for translation service
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# INPUT: Location
# ---------------------------------------------------------------------------

class LocationInput(BaseModel):
    lat: float
    lng: float
    name: str
    displayName: str


# ---------------------------------------------------------------------------
# INPUT: Crop Calendar
# ---------------------------------------------------------------------------

class CalendarEventInput(BaseModel):
    crop: str
    season: str
    sowingPeriod: str
    harvestingPeriod: str
    sowFromMon: Optional[int] = None
    sowToMon: Optional[int] = None
    harvFromMon: Optional[int] = None
    harvToMon: Optional[int] = None


class CalendarDataInput(BaseModel):
    success: bool
    state: str
    district: str
    districtCode: Optional[str] = None
    calendar: List[CalendarEventInput] = []


# ---------------------------------------------------------------------------
# INPUT: Weather (Open-Meteo via Next.js /api/weather)
# ---------------------------------------------------------------------------

class CurrentWeatherInput(BaseModel):
    time: str                           # ISO string from TS Date
    temperature_2m: float
    relative_humidity_2m: float
    apparent_temperature: float
    weather_code: int
    pressure_msl: float
    surface_pressure: float
    wind_speed_10m: float
    wind_direction_10m: float
    wind_gusts_10m: float
    precipitation: float
    cloud_cover: float
    icon: str
    condition: Dict[str, str]           # {en, hi, mr, ta, gu}


class HourlyWeatherInput(BaseModel):
    time: List[str]
    temperature_2m: List[float]
    precipitation_probability: List[float]
    wind_speed_10m: List[float]
    weather_code: List[int]
    icon: List[str]


class DailyWeatherInput(BaseModel):
    time: List[str]
    temperature_2m_max: List[float]
    temperature_2m_min: List[float]
    precipitation_sum: List[float]


class LightningInput(BaseModel):
    score: float
    category: str


class WeatherDataInput(BaseModel):
    latitude: float
    longitude: float
    elevation: float
    timezone: Optional[str] = None
    timezoneAbbreviation: Optional[str] = None
    utcOffsetSeconds: int
    current: CurrentWeatherInput
    hourly: HourlyWeatherInput
    daily: DailyWeatherInput
    lightning: Optional[LightningInput] = None


# ---------------------------------------------------------------------------
# INPUT: Corrected Forecast + Soil Moisture (soilModel backend)
# ---------------------------------------------------------------------------

class GridUsedInput(BaseModel):
    lat: float
    lon: float
    distance_deg: float
    cached: bool


class DailyForecastCorrectionInput(BaseModel):
    date: str
    tmax_raw: float
    tmax_corrected: float
    tmin_raw: float
    tmin_corrected: float
    pcp_raw: float
    pcp_corrected: float


class ForecastLocationInput(BaseModel):
    lat: float
    lon: float
    elevation_m: float


class ForecastBlockInput(BaseModel):
    success: bool
    location: ForecastLocationInput
    grids_used: List[GridUsedInput] = []
    forecast_source: str = ""
    forecast: List[DailyForecastCorrectionInput] = []
    runtime_seconds: float = 0.0


class SoilMoistureRecordInput(BaseModel):
    date: str
    P_obs: float
    Tmean: float
    PE: float
    P_eff: float
    snowpack: float
    w: float
    E: float
    R: float
    G: float
    w_frac: float
    sm_percentile: float
    is_forecast: int                    # 0 = historical, 1 = forecast


class SoilMoistureLocationInput(BaseModel):
    lat: float
    lon: float


class SoilMoistureBlockInput(BaseModel):
    success: bool
    location: SoilMoistureLocationInput
    cold_start: bool = False
    days_computed: int = 0
    checkpoint_last_date: str = ""
    soil_moisture: List[SoilMoistureRecordInput] = []
    runtime_seconds: float = 0.0


class ForecastDataInput(BaseModel):
    requested_lat: float
    requested_lon: float
    village_id: int
    forecast: ForecastBlockInput
    soil_moisture: SoilMoistureBlockInput
    cache_hit: bool = False
    total_runtime_seconds: float = 0.0
    cache_key: Optional[str] = None


# ---------------------------------------------------------------------------
# TOP-LEVEL INPUT: Full frontend payload
# ---------------------------------------------------------------------------

class AIAdvisoryRequest(BaseModel):
    location: LocationInput
    cropId: str = Field(..., min_length=1)
    calendarData: CalendarDataInput
    weatherData: WeatherDataInput
    forecastData: Optional[ForecastDataInput] = None
    language: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# ADVISORY CONTEXT: Compact deterministic output of ContextBuilder
# ---------------------------------------------------------------------------

class LocationSummary(BaseModel):
    lat: float
    lng: float
    name: str
    display_name: str
    state: str
    district: str


class CropContext(BaseModel):
    crop_id: str
    crop_name: str
    season: str
    crop_stage: str


class CurrentWeatherSummary(BaseModel):
    observation_time: str
    temperature_c: float
    apparent_temperature_c: float
    relative_humidity_percent: float
    precipitation_mm: float
    wind_speed_kmh: float
    wind_gusts_kmh: float
    cloud_cover_percent: float
    weather_condition: str              # English only


class ForecastSummary(BaseModel):
    source: str = "corrected_model"
    forecast_start_date: str            # YYYY-MM-DD
    forecast_end_date: str
    forecast_days: int
    total_rainfall_mm: float
    average_daily_rainfall_mm: float
    maximum_daily_rainfall_mm: float
    rainy_days: int
    dry_days: int
    minimum_temperature_c: float
    maximum_temperature_c: float
    average_min_temperature_c: float
    average_max_temperature_c: float
    rainfall_pattern: str
    forecast_fingerprint: str           # SHA256 of rounded pcp_corrected, for cache key


class WeatherApiSummary(BaseModel):
    source: str = "weather_api_daily"
    api_start_date: str = ""        # YYYY-MM-DD, first date in daily.time
    api_end_date: str = ""          # YYYY-MM-DD, last date in daily.time
    api_total_rainfall_mm: float
    api_min_temp_c: float
    api_max_temp_c: float


class HourlyWeatherSummary(BaseModel):
    max_precip_probability_next24h: float
    max_wind_speed_next24h: float


class SoilMoistureSummary(BaseModel):
    latest_w_frac: float
    latest_sm_percentile: float
    forecast_start_date: str
    forecast_end_date: str
    forecast_min_w_frac: float
    forecast_max_w_frac: float
    forecast_average_w_frac: float
    forecast_min_percentile: float
    forecast_max_percentile: float
    forecast_average_percentile: float
    start_percentile: float
    end_percentile: float
    soil_moisture_trend: str            # increasing / decreasing / stable / unknown
    cold_start: bool
    days_computed: int


class LightningSummary(BaseModel):
    score: float
    category: str


class AvailabilityFlags(BaseModel):
    weather_available: bool
    corrected_forecast_available: bool
    soil_moisture_available: bool
    lightning_available: bool
    calendar_available: bool


class AdvisoryContext(BaseModel):
    location: LocationSummary
    crop_context: CropContext
    current_weather: CurrentWeatherSummary
    forecast_summary: Optional[ForecastSummary] = None
    weather_api_summary: WeatherApiSummary
    hourly_summary: HourlyWeatherSummary
    soil_moisture_summary: Optional[SoilMoistureSummary] = None
    lightning_summary: Optional[LightningSummary] = None
    availability: AvailabilityFlags


# ---------------------------------------------------------------------------
# OUTPUT: Advisory response returned to Next.js
# ---------------------------------------------------------------------------

class AdvisoryResponse(BaseModel):
    advisory_summary: str = Field(
        ..., description="Two-paragraph plain-text agrometeorological advisory."
    )


# ---------------------------------------------------------------------------
# INTERNAL: Translation service helper
# ---------------------------------------------------------------------------

class TranslationResult(BaseModel):
    data: Dict[str, Any]
    translated: bool
    provider: Optional[str] = None


# ---------------------------------------------------------------------------
# LEGACY: Kept for any internal tooling that still uses the old flat request
# ---------------------------------------------------------------------------

class LocationDetail(BaseModel):
    village: Optional[str] = Field(None)
    district: Optional[str] = Field(None)
    state: str
    latitude: float
    longitude: float

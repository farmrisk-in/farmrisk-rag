---
title: FarmRisk
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# FarmRisk AI Backend

FarmRisk is a production-grade agro-meteorological advisory and decision-support backend built with **FastAPI**. It combines bias-corrected meteorological forecasts, hydrological soil-moisture modeling, crop calendars, and official **ICAR (Indian Council of Agricultural Research)** agronomic wisdom using a **3-Stage Retrieval-Augmented Generation (RAG)** architecture and deterministic agricultural decision rules.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Deterministic Rule Engines](#deterministic-rule-engines)
- [3-Stage Weather-Aware RAG Engine](#3-stage-weather-aware-rag-engine)
- [Caching & Request Deduplication](#caching--request-deduplication)
- [Multilingual Translation Engine](#multilingual-translation-engine)
- [API Endpoints Reference](#api-endpoints-reference)
  - [1. Root & Health Check](#1-root--health-check)
  - [2. Location Search](#2-location-search)
  - [3. Full AI Agrometeorological Advisory](#3-full-ai-agrometeorological-advisory)
  - [4. 24-Hour Weather Summary](#4-24-hour-weather-summary)
  - [5. Pest & Disease Card](#5-pest--disease-card)
  - [6. "What To Do Today" Recommendations](#6-what-to-do-today-recommendations)
- [Input Schemas & Parameters](#input-schemas--parameters)
- [Output Schemas](#output-schemas)
- [Offline Knowledge Ingestion Pipeline](#offline-knowledge-ingestion-pipeline)
- [Environment Variables & Configuration](#environment-variables--configuration)
- [Local Setup & Development](#local-setup--development)
- [Docker & Hugging Face Deployment](#docker--hugging-face-deployment)

---

## System Architecture

```
                                    +-----------------------------------------+
                                    |       Next.js / Frontend Client         |
                                    +-----------------------------------------+
                                                         |
                                                         | JSON Payloads
                                                         v
                                    +-----------------------------------------+
                                    |         FastAPI Gateway (Uvicorn)       |
                                    +-----------------------------------------+
                                        |                   |              |
                +-----------------------+                   |              +------------------------+
                |                                           |                                       |
                v                                           v                                       v
    +-----------------------+                   +------------------------+              +-----------------------+
    | Location Resolver     |                   | Deterministic Engines  |              | Advisory Context      |
    | (Nominatim + OSM +    |                   | - Irrigation Decision  |              | Builder               |
    | Canonical State Norm) |                   | - IMD Weather To-Dos   |              | - Compact Aggregation |
    +-----------------------+                   | - Pest/Disease Index   |              | - Trend Computations  |
                                                | - Ag Outlook Evaluator |              +-----------------------+
                                                +------------------------+                          |
                                                            |                                       v
                                                            |                          +------------------------+
                                                            |                          | Caching & Deduplication|
                                                            |                          | - Redis / In-Memory    |
                                                            |                          | - Async Keyed Locks    |
                                                            |                          +------------------------+
                                                            |                                       | (Cache Miss)
                                                            v                                       v
                                                +------------------------+              +------------------------+
                                                | Multilingual Engine    |              | 3-Stage RAG Retriever  |
                                                | - Regional Languages   |<-------------| - BGE-Small-EN-v1.5    |
                                                | - @@N@@ Term Locking   |              | - Supabase pgvector    |
                                                +------------------------+              +------------------------+
                                                            ^                                       |
                                                            |                                       v
                                                +------------------------+              +------------------------+
                                                | Primary: Google Gemini |<-------------| Knowledge Grounding    |
                                                | Fallback: Groq (Llama) |              | (ICAR Bulletins)       |
                                                +------------------------+              +------------------------+
```

---

## 📂 Modular Project Structure

The API is architected so that **every feature and endpoint lives in its own dedicated file**. Adding or removing a feature requires modifying only its dedicated file and registering or unregistering it in the router index:

```
app/
├── api/
│   ├── __init__.py                # Master API Gateway (mounts all feature routers)
│   ├── health.py                  # Root ('/') & Health ('/health') endpoints
│   ├── location.py                # Location search ('/api/location/search')
│   └── advisory/
│       ├── __init__.py            # Advisory Router (aggregates advisory sub-features)
│       ├── dependencies.py        # Shared singletons, retriever & pest card service
│       ├── overview.py            # POST /api/advisory (canonical crop advisory)
│       ├── weather_summary.py     # POST /api/advisory/weather-summary (24h summary)
│       ├── pest_card.py           # POST /api/advisory/pest-card (pest & disease card)
│       └── what_to_do.py          # POST /api/advisory/what-to-do (daily actions)
├── core/
│   ├── caching.py                 # Redis / In-Memory cache & Async LockManager
│   ├── config.py                  # Pydantic Settings & Environment Variables
│   └── logging.py                 # Structlog & formatted logging
├── database/
│   ├── client.py                  # Supabase client singleton
│   └── pgvector.py                # Supabase RPC pgvector similarity search
├── llm/
│   ├── advisory_engine.py         # Canonical English advisory prompt & validation
│   └── providers.py               # Gemini (Primary) & Groq (Fallback) providers
├── models/
│   └── schemas.py                 # Pydantic request/response schemas
├── rag/
│   └── retriever.py               # 3-Stage weather-aware vector retriever
└── services/
    ├── context_builder.py         # Deterministic aggregation of frontend payload
    ├── irrigation.py              # Hydrological soil-moisture irrigation engine
    ├── location.py                # Nominatim OSM geocoding & state normalization
    ├── pest_disease_card.py       # Heuristic pest risk index & card builder
    ├── weather_todos.py           # IMD operational weather warning actions
    └── what_to_do.py              # Deterministic multi-source action aggregator
```

### Adding a New Feature
1. Create a new endpoint file (e.g. `app/api/advisory/soil_profile.py` or `app/api/alerts.py`).
2. Instantiate `router = APIRouter()`.
3. In `app/api/advisory/__init__.py` (or `app/api/__init__.py`), add `router.include_router(new_router)`.

### Removing a Feature
Simply remove or comment out `router.include_router(...)` in the corresponding `__init__.py`.

---
## Key Features

### 1. Full Agrometeorological Advisory Generation (`/api/advisory`)

- Generates a canonical 2-3 paragraph plain-text advisory summary for farmers.
- Seamlessly fuses observed weather, 5-10 day bias-corrected meteorological forecasts, hydrological soil-moisture percentile trajectories, crop phenological stage, and ICAR advisory bulletins.
- Strictly adheres to agronomic communication guidelines: outputs plain language, avoids exposing technical jargon or raw model percentiles to farmers, and references explicit dates and relative timing (e.g., "tomorrow", "after about 3 days").
- Emits an auditable, deterministic agricultural outlook (_Normal_, _Challenging_, _Hazardous_, _Critical_) based on IMD (India Meteorological Department) operational risk standards.

### 2. 24-Hour Actionable Weather Summary (`/api/advisory/weather-summary`)

- Produces a punchy, 1-2 sentence operational briefing highlighting the single dominant weather hazard in the next 24 hours.
- Samples 5 representative hourly intervals (current hour, +6h, +12h, +18h, +23h) to detect sharp transitions (e.g., clear skies transitioning into evening thunderstorms).
- Intelligently de-emphasizes heat when rain suppresses peak temperatures, highlights peak wind gust windows, and flags heat stress/muggy discomfort when temperature and humidity peak together.

### 3. Crop-Specific Pest & Disease Card (`/api/advisory/pest-card`)

- Generates a dedicated Pest & Disease risk assessment card grounded strictly in retrieved ICAR documents.
- **Deterministic Risk Banding**: Computes a heuristic index (0–100) and risk tier (`LOW`, `MEDIUM`, `HIGH`) from temperature suitability, relative humidity, soil saturation, and rainy days. The LLM is **never** permitted to override this band.
- **Anti-Hallucination Guardrails**: The LLM can only name specific pests, diseases, or cultural interventions that explicitly appear in the retrieved ICAR knowledge chunks. If no crop-specific chunk is retrieved, it falls back to safe cultural measures (e.g., weeding, scouting undersides, clearing drainage channels) without inventing chemical brand names or dosages.
- Action items are strictly constrained to 12–16 words per item and returned with document citations.

### 4. "What To Do Today" Daily Recommendations (`/api/advisory/what-to-do`)

- Serves an aggregated, high-priority dashboard widget returning at most **two** urgent action items for the farmer.
- **Deterministic Selection**:
  - **Slot 1**: Highest-severity Pest & Disease action item.
  - **Slot 2**: Best Irrigation / Soil Moisture recommendation (`now`, `schedule`, `drainage`, or `none`).
  - **Fallback**: Fills any empty slot with high-priority IMD weather to-dos (e.g., postponement of pesticide spraying due to wind gusts > 40 km/h, drainage preparation before heavy rainfall).
  - Eliminates cross-category duplicates automatically.

### 5. Indian Village & Town Geocoding (`/api/location/search`)

- Fast location resolution tailored for India via OpenStreetMap Nominatim.
- Auto-extracts village, town, suburb, district, and state from address details.
- Normalizes resolved states to canonical Indian states and union territories.

### 6. Dual LLM Provider Pipeline with Seamless Fallback

- **Primary Provider**: Google Gemini (`gemini-3.1-flash-lite` via `google-genai` SDK).
- **Fallback Provider**: Groq Cloud (`llama-3.3-70b-versatile` via `groq` SDK).
- Automatic retries with exponential backoff (2s, 4s, 8s) for transient HTTP errors (429, 500, 502, 503, 504) and connection timeouts.

---

## Deterministic Rule Engines

To eliminate hallucinations in mission-critical farming operations, core calculations are performed by deterministic Python engines rather than generative AI:

### 1. Irrigation Decision Engine (`app/services/irrigation.py`)

- Evaluates the hydrological soil-moisture percentile trajectory against crop-specific water sensitivity thresholds:
  - **Sensitive crops/stages** (flowering, grain fill, tuber bulking): Plan at 40%, Urgent at 30%.
  - **Normal crops/stages** (vegetative, established): Plan at 30%, Urgent at 20%.
  - **Tolerant crops/stages** (maturity, ripening, dry-down): Plan at 20%, Urgent at 10%.
  - **Saturated soil** (percentile $\ge 80\%$): Action is `drainage` (irrigation suppressed).
- Determines four concrete actions: `drainage`, `now`, `schedule`, `none`.
- Identifies the exact crossing date and relative timing ("today", "tomorrow", "around DD/MM/YYYY").
- Explains _why_ soil recovers or dries by detecting peak rain events ($\ge 2.5\text{ mm}$) within the forecast window.
- Communicates using qualitative soil moisture categories (_Exceptional Wet_, _Severe Wet_, _Normal_, _Abnormally Dry_, _Extreme Dry_) and never reveals raw percentile figures.

### 2. Weather To-Do Engine (`app/services/weather_todos.py`)

- Based directly on IMD operational warning categories:
  - **24-Hour & 5-Day Rainfall**: Heavy ($\ge 64.5\text{ mm}$), Very Heavy ($\ge 115.6\text{ mm}$), Moderate ($\ge 15.6\text{ mm}$).
  - **Rain Spells**: Moderate ($\ge 10\text{ mm/hr}$), Intense ($\ge 20\text{ mm/hr}$), Very Intense ($\ge 30\text{ mm/hr}$), Cloudburst ($> 100\text{ mm/hr}$).
  - **Extreme Heat**: Plains $\ge 40^\circ\text{C}$, Severe $\ge 45^\circ\text{C}$; Coastal $\ge 37^\circ\text{C}$; Hilly $\ge 30^\circ\text{C}$.
  - **Cold Stress**: Minimum temperature $\le 8^\circ\text{C}$.
  - **Wind & Gusts**: $\ge 40\text{ km/h}$ (spray drift warning), $\ge 50\text{ km/h}$ (lodging & physical damage risk).
- Sorts recommendations by severity: `unfavorable` (rank 3) > `cautionary` (rank 2) > `favorable` (rank 1).

### 3. Pest & Disease Pressure Index (`app/services/pest_disease_card.py`)

- Blends temperature suitability ($25\text{--}35^\circ\text{C}$), high relative humidity ($\ge 70\%$), soil saturation, and consecutive rainy days into a composite score (0–100).
- Categorizes risk into `HIGH` ($\ge 66$), `MEDIUM` ($40\text{--}65$), and `LOW` ($< 40$).

### 4. Agricultural Outlook Evaluator (`app/llm/advisory_engine.py`)

- Assesses weather risks to assign an overall outlook:
  - **Critical**: Extremely heavy rain ($\ge 204.5\text{ mm}$), extreme heat ($\ge 45^\circ\text{C}$), or damaging winds ($\ge 60\text{ km/h}$).
  - **Hazardous**: Very heavy rain ($\ge 115.6\text{ mm}$), heatwave ($\ge 42^\circ\text{C}$), high winds ($\ge 45\text{ km/h}$), or exceptional soil saturation.
  - **Challenging**: Heavy rain ($\ge 64.5\text{ mm}$), moderate heat ($\ge 38^\circ\text{C}$), gusty winds ($\ge 30\text{ km/h}$), or dry soil.
  - **Normal**: Favorable agricultural conditions without extreme meteorological hazards.

---

## 3-Stage Weather-Aware RAG Engine

The retriever (`app/rag/retriever.py`) uses a progressive 3-stage fallback strategy querying Supabase **pgvector** using **`BAAI/bge-small-en-v1.5`** embeddings:

1. **Stage 1 (Strict Local Match)**: Filters by `crop + state + season`.
2. **Stage 2 (Regional/National Match)**: If fewer than `RAG_TOP_K` (default 5) chunks are found, widens search to `crop + season` (ignoring state).
3. **Stage 3 (General Crop Guidance)**: If still under `top_k`, queries by `crop` alone.
4. **General Crop Bypass**: When `cropId == "general"`, RAG retrieval is bypassed completely to produce a safe, crop-agnostic general advisory without bias toward any single crop.

### Query Augmentation via `RetrievalContext`

The search vector is built not just from the crop name, but enriched with real-time agronomic facts:

- Crop growth stage (e.g., sowing, vegetative, flowering, maturity)
- Rainfall pattern & total rainfall (mm)
- Temperature range (minimum & maximum $^\circ\text{C}$)
- Soil moisture trend (increasing, decreasing, stable) and forecast average percentile
- Lightning threat category

---

## Caching & Request Deduplication

1. **Two-Tier Cache Manager (`app/core/caching.py`)**:
   - Supports **In-Memory** caching (default) and **Redis** (`CACHE_TYPE=redis`).
   - Automatically degrades to in-memory caching if Redis becomes unreachable.
   - Default advisory TTL: **12 hours** (43,200 seconds).
2. **Deterministic Cache Key Generation**:
   - Keyed on `crop`, `latitude`, `longitude` (or `village_id`), and a SHA-256 `weather_hash` combining the forecast rainfall fingerprint and soil moisture availability.
3. **Async Key-Level Deduplication Locks (`LockManager`)**:
   - Simultaneous requests for the exact same crop, village, and weather conditions lock on the specific cache key.
   - The first request executes generation; all concurrent requests wait asynchronously and resolve as immediate cache hits, preventing **cache stampedes (thundering herds)**.

---

## Multilingual Translation Engine

- **Target Languages**: English (`en`), Hindi (`hi`), Marathi (`mr`), Tamil (`ta`), Gujarati (`gu`), Telugu (`te`), Bengali (`bn`), and other Indian regional languages.
- **Entity Placeholder Locking (`@@N@@`)**:
  - In pest cards and advisories, critical agronomic entities (crop names, growth stages, seasons, and identified pest names) are replaced with temporary tokens (`@@0@@`, `@@1@@`) before translation.
  - The model translates the sentence structure and cultural advice without mutating or mis-inflecting official terms, and the exact localized entities are restored after translation.
- **Dedicated Translation Cache**: Translated responses are cached independently under `translation:<hash>:<lang>` with deduplication locks.

---

## API Endpoints Reference

| Method         | Endpoint                        | Description                                  | Request Body / Params             |
| :------------- | :------------------------------ | :------------------------------------------- | :-------------------------------- |
| `GET` / `HEAD` | `/`                             | Root health check for Hugging Face Spaces    | None                              |
| `GET` / `HEAD` | `/health`                       | API operational status & environment         | None                              |
| `GET`          | `/api/location/search`          | Search Indian villages, towns, and cities    | `q` (query string, min length: 2) |
| `POST`         | `/api/advisory`                 | Generate comprehensive AI agro-advisory      | `AIAdvisoryRequest` (JSON)        |
| `POST`         | `/api/advisory/weather-summary` | 24-hour concise 1–2 sentence summary         | `AIAdvisoryRequest` (JSON)        |
| `POST`         | `/api/advisory/pest-card`       | Crop-specific Pest & Disease overview card   | `AIAdvisoryRequest` (JSON)        |
| `POST`         | `/api/advisory/what-to-do`      | "What To Do Today" top-2 prioritized actions | `AIAdvisoryRequest` (JSON)        |

---

## Input Schemas & Parameters

All advisory endpoints accept the unified **`AIAdvisoryRequest`** payload:

```json
{
  "location": {
    "lat": 26.9124,
    "lng": 75.7873,
    "name": "Jaipur",
    "displayName": "Jaipur, Rajasthan, India"
  },
  "cropId": "wheat",
  "language": "en",
  "calendarData": {
    "success": true,
    "state": "Rajasthan",
    "district": "Jaipur",
    "districtCode": "RJ_JAI",
    "calendar": [
      {
        "crop": "Wheat",
        "season": "Rabi",
        "sowingPeriod": "Nov - Dec",
        "harvestingPeriod": "Mar - Apr",
        "sowFromMon": 11,
        "sowToMon": 12,
        "harvFromMon": 3,
        "harvToMon": 4
      }
    ]
  },
  "weatherData": {
    "latitude": 26.9124,
    "longitude": 75.7873,
    "elevation": 435.0,
    "timezone": "Asia/Kolkata",
    "timezoneAbbreviation": "IST",
    "utcOffsetSeconds": 19800,
    "current": {
      "time": "2026-08-30T06:00",
      "temperature_2m": 28.5,
      "relative_humidity_2m": 78.0,
      "apparent_temperature": 32.1,
      "weather_code": 3,
      "pressure_msl": 1008.2,
      "surface_pressure": 965.4,
      "wind_speed_10m": 14.5,
      "wind_direction_10m": 240.0,
      "wind_gusts_10m": 22.0,
      "precipitation": 0.0,
      "cloud_cover": 65.0,
      "icon": "cloudy",
      "condition": {
        "en": "Partly Cloudy",
        "hi": "आंशिक रूप से बादल छाए रहेंगे"
      }
    },
    "hourly": {
      "time": ["2026-08-30T06:00", "..."],
      "temperature_2m": [28.5, 29.2, 31.0],
      "precipitation_probability": [10.0, 20.0, 45.0],
      "wind_speed_10m": [14.5, 16.0, 18.2],
      "weather_code": [3, 3, 61],
      "icon": ["cloudy", "cloudy", "rain"]
    },
    "daily": {
      "time": ["2026-08-30", "..."],
      "temperature_2m_max": [34.5, 33.0],
      "temperature_2m_min": [24.0, 23.5],
      "precipitation_sum": [4.5, 12.0]
    },
    "lightning": {
      "score": 15.0,
      "category": "low"
    }
  },
  "forecastData": {
    "requested_lat": 26.9124,
    "requested_lon": 75.7873,
    "village_id": 104231,
    "forecast": {
      "success": true,
      "forecast_source": "bias_corrected_gfs",
      "forecast": [
        {
          "date": "2026-08-30",
          "tmax_raw": 35.0,
          "tmax_corrected": 34.2,
          "tmin_raw": 23.5,
          "tmin_corrected": 24.1,
          "pcp_raw": 5.2,
          "pcp_corrected": 4.8
        }
      ]
    },
    "soil_moisture": {
      "success": true,
      "cold_start": false,
      "days_computed": 365,
      "checkpoint_last_date": "2026-08-29",
      "soil_moisture": [
        {
          "date": "2026-08-29",
          "P_obs": 0.0,
          "Tmean": 29.1,
          "PE": 5.2,
          "P_eff": 0.0,
          "snowpack": 0.0,
          "w": 120.5,
          "E": 3.8,
          "R": 0.0,
          "G": 0.5,
          "w_frac": 0.45,
          "sm_percentile": 42.5,
          "is_forecast": 0
        },
        {
          "date": "2026-08-30",
          "P_obs": 4.8,
          "Tmean": 28.5,
          "PE": 4.5,
          "P_eff": 4.0,
          "snowpack": 0.0,
          "w": 122.0,
          "E": 3.2,
          "R": 0.0,
          "G": 0.4,
          "w_frac": 0.48,
          "sm_percentile": 48.0,
          "is_forecast": 1
        }
      ]
    }
  }
}
```

---

## Output Schemas

### 1. `POST /api/advisory` Response

```json
{
  "advisory_summary": "Over the next 7 days in Jaipur, Rajasthan, total rainfall is expected to be around 24.5 mm across 3 rainy days, with the wettest single day on 02/09/2026 bringing moderate rain of around 16.0 mm. Day temperatures will reach up to 34.5°C with moderate humidity. Soil moisture is currently Abnormally Dry but is expected to recover to Normal following the midweek rainfall.\n\nFor Wheat in its vegetative stage, ensure field drainage channels are clear before the expected rain on Wednesday to prevent temporary waterlogging. Postpone scheduled urea top-dressing and spray applications until after rainfall subsides.\n\nOverall, the agricultural outlook for this period is *Challenging*.",
  "irrigation_insight": "irrigation not needed for the next 7 days; soil is expected to stay adequate (currently Abnormally Dry) (expected rain around 02/09/2026 may recharge it)",
  "sources": [
    {
      "id": "rajasthan_wheat_rabi_1_3a8b9f",
      "crop": "Wheat",
      "state": "Rajasthan",
      "season": "Rabi",
      "source": "ICAR Rabi Agro-Advisory",
      "page": 42,
      "score": 0.842,
      "content": "Wheat: At crown root initiation and vegetative stage, irrigate only if soil moisture is deficient..."
    }
  ]
}
```

### 2. `POST /api/advisory/weather-summary` Response

```json
{
  "advisory_summary": "Moderate rainfall of around 16 mm is expected between 2 pm and 6 pm tomorrow, temporarily lowering peak daytime temperatures to 29°C."
}
```

### 3. `POST /api/advisory/pest-card` Response

```json
{
  "risk": "MEDIUM",
  "score": 54.2,
  "driver": "humidity of 78% favouring fungal disease",
  "summary": "Moderate risk for Wheat during its vegetative stage as relative humidity near 78% and intermittent rainfall create conditions favorable for foliar blights and sucking pests.",
  "potential": ["Foliar Blight", "Aphids"],
  "actions": [
    {
      "priority": 1,
      "title": "Inspect Lower Leaves",
      "detail": "Inspect lower leaves and crown root area daily for early oval brown blight lesions.",
      "cites": [1]
    },
    {
      "priority": 2,
      "title": "Clear Field Drains",
      "detail": "Ensure field drainage channels are clear to prevent water stagnation that accelerates fungal development.",
      "cites": [2]
    }
  ],
  "cites": [1, 2],
  "crop_id": "wheat",
  "crop_name": "Wheat",
  "crop_stage": "vegetative",
  "season": "Rabi",
  "is_general": false
}
```

### 4. `POST /api/advisory/what-to-do` Response

```json
{
  "success": true,
  "crop_id": "wheat",
  "crop_name": "Wheat",
  "is_general": false,
  "language": "en",
  "recommendations": [
    {
      "category": "pest",
      "severity": "MEDIUM",
      "title": "Inspect Lower Leaves",
      "hint": "Inspect lower leaves and crown root area daily for early oval brown blight lesions.",
      "sources": [],
      "crop_name": "Wheat",
      "is_general": false
    },
    {
      "category": "irrigation",
      "severity": "none",
      "title": "Irrigation not needed",
      "hint": "soil is expected to stay adequate (currently Abnormally Dry) (expected rain around 02/09/2026 may recharge it)",
      "sources": [],
      "crop_name": null,
      "is_general": null
    }
  ]
}
```

### 5. `GET /api/location/search?q=Jaipur` Response

```json
[
  {
    "village": "Jaipur",
    "district": "Jaipur",
    "state": "Rajasthan",
    "latitude": 26.9124,
    "longitude": 75.7873
  }
]
```

---

## Offline Knowledge Ingestion Pipeline

The repository includes a complete offline ETL pipeline (`pipeline/`) to process raw ICAR agro-advisory documents into vector embeddings:

```
PDF Advisories (data/pdfs/)
       │
       ▼  rag/extract.py (PyMuPDF)
Raw Page JSONs (data/extracted/)
       │
       ▼  pipeline/02_parse.py (State/Crop Detect, Ligature Fixes)
Parsed Advisories (data/parsed/advisories.json)
       │
       ▼  pipeline/03_validate.py (Canonical Check, Min Length, SHA256 Dedup)
Valid Advisories (data/parsed/valid_advisories.json)  [Quarantine: data/quarantine/]
       │
       ▼  pipeline/04_chunk.py (Tiktoken cl100k_base, 700 tokens, 100 overlap)
Advisory Chunks (data/parsed/chunks.json)
       │
       ▼  pipeline/05_embed_upload.py (SentenceTransformers bge-small-en-v1.5)
Supabase pgvector Database (Table: advisories)
```

To run the entire ingestion pipeline end-to-end:

```bash
python pipeline/run_all.py
```

---

## Environment Variables & Configuration

Configure these variables in `.env` or in your deployment environment:

| Variable                        | Type    | Default                    | Description                                           |
| :------------------------------ | :------ | :------------------------- | :---------------------------------------------------- |
| `HOST`                          | `str`   | `127.0.0.1`                | Host address for Uvicorn                              |
| `PORT`                          | `int`   | `8000`                     | Port for FastAPI (Docker uses `7860`)                 |
| `DEBUG`                         | `bool`  | `true`                     | Debug mode toggle                                     |
| `APP_ENV`                       | `str`   | `development`              | Application environment (`development`, `production`) |
| `SUPABASE_URL`                  | `str`   | `""`                       | Supabase project URL for pgvector                     |
| `SUPABASE_ANON_KEY`             | `str`   | `""`                       | Supabase anonymous API key                            |
| `SUPABASE_SERVICE_ROLE_KEY`     | `str`   | `""`                       | Supabase service role key (for pipeline upload)       |
| `LLM_PROVIDER`                  | `str`   | `gemini`                   | Primary LLM provider (`gemini`)                       |
| `ENABLE_GROQ_FALLBACK`          | `bool`  | `true`                     | Enable automated Groq fallback on Gemini failures     |
| `GOOGLE_API_KEY`                | `str`   | `""`                       | Google Gemini API Key                                 |
| `GEMINI_MODEL`                  | `str`   | `gemini-3.1-flash-lite`    | Gemini model identifier                               |
| `TEMPERATURE`                   | `float` | `0.2`                      | Generation sampling temperature                       |
| `GROQ_API_KEY`                  | `str`   | `""`                       | Groq API Key                                          |
| `GROQ_MODEL`                    | `str`   | `llama-3.3-70b-versatile`  | Groq model identifier                                 |
| `CACHE_TYPE`                    | `str`   | `in_memory`                | Cache backend: `in_memory` or `redis`                 |
| `REDIS_URL`                     | `str`   | `redis://localhost:6379/0` | Redis connection string (if `CACHE_TYPE=redis`)       |
| `ADVISORY_CACHE_TTL`            | `int`   | `43200`                    | Advisory cache TTL in seconds (12 hours)              |
| `LOG_FORMAT`                    | `str`   | `TEXT`                     | Log format: `TEXT` or `JSON`                          |
| `LOG_LEVEL`                     | `str`   | `INFO`                     | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)   |
| `RAG_TOP_K`                     | `int`   | `5`                        | Maximum number of RAG knowledge chunks retrieved      |
| `RAINY_DAY_THRESHOLD_MM`        | `float` | `2.5`                      | IMD floor for a rainy day (mm)                        |
| `HEAVY_RAIN_THRESHOLD_MM`       | `float` | `35.0`                     | Heavy rainfall threshold for context builder (mm)     |
| `SOIL_MOISTURE_TREND_TOLERANCE` | `float` | `5.0`                      | Percentile delta to classify trend direction          |
| `ENABLE_MOCK_ADVISORY`          | `bool`  | `false`                    | Development-only mock generator                       |

---

## Local Setup & Development

### 1. Prerequisites

- Python 3.12+
- (Optional) Redis server

### 2. Clone and Install Dependencies

```bash
git clone https://github.com/farmrisk-in/farmrisk-rag.git
cd farmrisk-rag

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Pre-download Embedding Model

```bash
python download_models.py
```

### 4. Configure Environment

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
```

### 5. Launch Development Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive Swagger API documentation will be available at `http://localhost:8000/docs`.

---

## Docker & Hugging Face Deployment

This project is pre-configured to run as a **Hugging Face Docker Space** listening on port `7860`.

### Build & Run via Docker

```bash
docker build -t farmrisk-backend .
docker run -p 7860:7860 --env-file .env farmrisk-backend
```

The `Dockerfile`:

1. Uses `python:3.12-slim` base image.
2. Installs system build dependencies (`build-essential`, `libpq-dev`).
3. Pre-downloads `BAAI/bge-small-en-v1.5` directly into the container image cache via `download_models.py` during build time for fast cold starts.
4. Grants full permissions to `/code` for Hugging Face's non-root execution user (UID 1000).
5. Serves Uvicorn on `0.0.0.0:7860`.

---

## 🚀 Automatic Sync to Hugging Face Space (CI/CD)

The repository includes a GitHub Actions workflow (`.github/workflows/sync_to_hf.yml`) that **automatically syncs every new commit on `main` to your Hugging Face Space**.

### Step 1: Create a Space on Hugging Face
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose **Docker** as the Space SDK (Blank template).
3. Name your space (e.g. `farmrisk-rag`).

### Step 2: Get a Hugging Face Access Token
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2. Create a new token with **Write** permission.
3. Copy the token.

### Step 3: Configure GitHub Secrets
In your GitHub repository (`farmrisk-in/farmrisk-rag`):
1. Navigate to **Settings** → **Secrets and variables** → **Actions**.
2. Click **New repository secret**:
   - **Name**: `HF_TOKEN`
   - **Secret**: *(Paste your Hugging Face Write Token)*
3. (Optional) Under **Variables** (or Secrets), add:
   - **Name**: `HF_SPACE`
   - **Value**: `<YOUR_HF_USERNAME>/<YOUR_SPACE_NAME>` (e.g. `farmrisk-in/farmrisk-rag`)

### Step 4: Add Environment Secrets in Hugging Face Space
In your Hugging Face Space:
1. Navigate to **Settings** → **Variables and secrets**.
2. Add your environment secrets:
   - `GOOGLE_API_KEY`
   - `GROQ_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`
   - `SUPABASE_SERVICE_ROLE_KEY`

### Step 5: Push and Auto-Deploy
Now, whenever you push any new commit to `main`:
```bash
git push origin main
```
GitHub Actions will automatically push to Hugging Face, trigger a Docker rebuild, and deploy your latest code.

*(Optional)* **Local Direct Dual-Push**: You can also push to both GitHub and Hugging Face simultaneously from your local terminal:
```bash
git remote set-url --add --push origin https://github.com/farmrisk-in/farmrisk-rag.git
git remote set-url --add --push origin https://huggingface.co/spaces/<YOUR_HF_USERNAME>/<YOUR_SPACE_NAME>
```
With this configured, a simple `git push` pushes to both remotes in one command.

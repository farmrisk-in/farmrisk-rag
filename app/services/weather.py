import hashlib
import httpx
from typing import Dict, Any, List

class WeatherService:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0)

    async def fetch_10day_forecast(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Fetch 10-day meteorological data from Open-Meteo."""
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_max",
            "timezone": "auto",
            "forecast_days": 10
        }
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            daily = data.get("daily", {})
            time_list = daily.get("time", [])
            temp_max = daily.get("temperature_2m_max", [])
            temp_min = daily.get("temperature_2m_min", [])
            precip = daily.get("precipitation_sum", [])
            wind = daily.get("wind_speed_10m_max", [])
            humidity = daily.get("relative_humidity_2m_max", [])
            
            forecast = []
            for idx, date_str in enumerate(time_list):
                forecast.append({
                    "day": idx + 1,
                    "date": date_str,
                    "temp_max": temp_max[idx] if idx < len(temp_max) else None,
                    "temp_min": temp_min[idx] if idx < len(temp_min) else None,
                    "precipitation_sum": precip[idx] if idx < len(precip) else 0.0,
                    "wind_speed_max": wind[idx] if idx < len(wind) else 0.0,
                    "humidity_max": humidity[idx] if idx < len(humidity) else 0.0
                })
                
            # Perform rule-based risk evaluation
            risks = self._assess_risks(forecast)
            
            # Generate a hash for spatial/temporal caching
            weather_hash = self._generate_weather_hash(forecast)
            
            return {
                "forecast": forecast,
                "risks": risks,
                "weather_hash": weather_hash
            }
            
        except Exception as e:
            print(f"Error fetching weather forecast: {e}")
            # Fallback mock/empty weather forecast so RAG doesn't crash the server
            fallback_forecast = []
            for d in range(1, 11):
                fallback_forecast.append({
                    "day": d,
                    "date": f"Day {d}",
                    "temp_max": 30.0,
                    "temp_min": 20.0,
                    "precipitation_sum": 0.0,
                    "wind_speed_max": 10.0,
                    "humidity_max": 60.0
                })
            return {
                "forecast": fallback_forecast,
                "risks": ["No current warnings. (Weather API Offline fallback)"],
                "weather_hash": "offline_fallback"
            }

    def _assess_risks(self, forecast: List[Dict[str, Any]]) -> List[str]:
        warnings = []
        has_heatwave = False
        has_heavy_rain = False
        has_gale = False
        has_frost = False
        
        for day in forecast:
            tmax = day.get("temp_max") or 0.0
            tmin = day.get("temp_min") or 0.0
            prec = day.get("precipitation_sum") or 0.0
            wind = day.get("wind_speed_max") or 0.0
            
            if tmax > 42.0 and not has_heatwave:
                warnings.append("Extreme Heat Alert: Maximum temperatures exceeding 42°C expected.")
                has_heatwave = True
            if tmin < 5.0 and not has_frost:
                warnings.append("Frost Warning: Night temperatures dropping below 5°C expected.")
                has_frost = True
            if prec > 40.0 and not has_heavy_rain:
                warnings.append(f"Heavy Rain Advisory: Daily precipitation exceeding 40mm predicted (Day {day['day']}).")
                has_heavy_rain = True
            if wind > 35.0 and not has_gale:
                warnings.append(f"High Wind Warning: Wind gust speeds exceeding 35 km/h expected (Day {day['day']}).")
                has_gale = True
                
        return warnings

    def _generate_weather_hash(self, forecast: List[Dict[str, Any]]) -> str:
        """Create a hash of the forecast parameters (rounded) to check cache stability."""
        hash_string = ""
        for day in forecast[:5]:  # Focus on next 5 days for stability
            tmax = round(day.get("temp_max") or 30.0)
            prec = round(day.get("precipitation_sum") or 0.0, 1)
            hash_string += f"{tmax}:{prec}|"
        return hashlib.sha256(hash_string.encode()).hexdigest()[:12]

    async def close(self):
        await self.client.aclose()

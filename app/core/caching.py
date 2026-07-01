import time
import hashlib
from typing import Optional, Dict, Any
from app.core.config import settings

class InMemoryCache:
    def __init__(self):
        # Format: key -> (value, expiry_time_seconds)
        self._cache: Dict[str, tuple] = {}

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        val, expiry = self._cache[key]
        if expiry and time.time() > expiry:
            del self._cache[key]
            return None
        return val

    def set(self, key: str, value: Any, ttl_seconds: int = 43200):
        expiry = time.time() + ttl_seconds if ttl_seconds else None
        self._cache[key] = (value, expiry)

    def clear(self):
        self._cache.clear()

class CacheManager:
    def __init__(self):
        self.provider = InMemoryCache()
        # In the future, if settings.CACHE_TYPE == "redis", we could instantiate a Redis provider.

    def get_advisory(self, crop: str, latitude: float, longitude: float, weather_hash: str) -> Optional[Dict[str, Any]]:
        # Round coordinates to 3 decimal places (approx. 110m grid accuracy)
        lat_grid = f"{latitude:.3f}"
        lon_grid = f"{longitude:.3f}"
        key = f"adv:{crop.lower().strip()}:{lat_grid}:{lon_grid}:{weather_hash}"
        return self.provider.get(key)

    def set_advisory(self, crop: str, latitude: float, longitude: float, weather_hash: str, advisory_data: Dict[str, Any], ttl: int = 43200):
        lat_grid = f"{latitude:.3f}"
        lon_grid = f"{longitude:.3f}"
        key = f"adv:{crop.lower().strip()}:{lat_grid}:{lon_grid}:{weather_hash}"
        self.provider.set(key, advisory_data, ttl)

    def get_translation(self, english_json: Dict[str, Any], language: str) -> Optional[Dict[str, Any]]:
        # Generate hash of English JSON structure
        serialized = json_stable_hash(english_json)
        key = f"trans:{serialized}:{language.lower().strip()}"
        return self.provider.get(key)

    def set_translation(self, english_json: Dict[str, Any], language: str, translated_data: Dict[str, Any], ttl: int = 43200):
        serialized = json_stable_hash(english_json)
        key = f"trans:{serialized}:{language.lower().strip()}"
        self.provider.set(key, translated_data, ttl)

def json_stable_hash(data: Dict[str, Any]) -> str:
    """Serialize dict in stable sorted way and SHA256 hash it."""
    import json
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

cache_manager = CacheManager()

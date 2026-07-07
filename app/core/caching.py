import time
import json
import hashlib
import asyncio
from typing import Optional, Dict, Any
from app.core.config import settings
from app.core.logging import logger

# 1. Abstract Cache Provider Interface
class CacheProvider:
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError()

    def set(self, key: str, value: Any, ttl_seconds: int = 43200):
        raise NotImplementedError()

    def ttl(self, key: str) -> Optional[int]:
        raise NotImplementedError()


# 2. InMemoryCache Provider
class InMemoryCache(CacheProvider):
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

    def ttl(self, key: str) -> Optional[int]:
        if key not in self._cache:
            return None
        val, expiry = self._cache[key]
        if not expiry:
            return None
        remaining = int(expiry - time.time())
        return remaining if remaining > 0 else 0


# 3. RedisCache Provider
class RedisCache(CacheProvider):
    def __init__(self, redis_url: str):
        import redis
        # Set short timeouts so fallback triggers quickly if Redis goes down
        self.client = redis.from_url(
            redis_url,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
            decode_responses=True
        )
        self.client.ping()

    def get(self, key: str) -> Optional[Any]:
        val = self.client.get(key)
        if val is None:
            return None
        return json.loads(val)

    def set(self, key: str, value: Any, ttl_seconds: int = 43200):
        serialized = json.dumps(value)
        if ttl_seconds:
            self.client.setex(key, ttl_seconds, serialized)
        else:
            self.client.set(key, serialized)

    def ttl(self, key: str) -> Optional[int]:
        ttl_val = self.client.ttl(key)
        if ttl_val < 0:
            return None
        return ttl_val


# Stable hashing helper for JSON/dictionaries
def json_stable_hash(data: Dict[str, Any]) -> str:
    """Serialize dict in stable sorted way and SHA256 hash it."""
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


# 4. CacheManager
class CacheManager:
    def __init__(self):
        self.in_memory = InMemoryCache()
        self.redis = None
        self.use_redis = settings.CACHE_TYPE.lower() == "redis"

        if self.use_redis:
            try:
                logger.info(f"Initializing RedisCache at {settings.REDIS_URL}...")
                self.redis = RedisCache(settings.REDIS_URL)
                logger.info("RedisCache connected successfully.")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}. Falling back to InMemoryCache.")
                self.redis = None
                self.use_redis = False

    @property
    def provider(self) -> CacheProvider:
        if self.use_redis and self.redis:
            return self.redis
        return self.in_memory

    def get(self, key: str) -> Optional[Any]:
        try:
            return self.provider.get(key)
        except Exception as e:
            logger.error(f"Cache provider error on get for key '{key}': {e}. Falling back to InMemoryCache.")
            return self.in_memory.get(key)

    def set(self, key: str, value: Any, ttl_seconds: int = 43200):
        try:
            self.provider.set(key, value, ttl_seconds)
        except Exception as e:
            logger.error(f"Cache provider error on set for key '{key}': {e}. Falling back to InMemoryCache.")
            self.in_memory.set(key, value, ttl_seconds)

    def ttl(self, key: str) -> Optional[int]:
        try:
            return self.provider.ttl(key)
        except Exception as e:
            return self.in_memory.ttl(key)

    # ----------------------------------------------------
    # Backward Compatibility Helpers
    # ----------------------------------------------------
    def get_advisory(self, crop: str, latitude: float, longitude: float, weather_hash: str, village_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        key = advisory_cache.get_key(crop, latitude, longitude, weather_hash, village_id)
        return self.get(key)

    def set_advisory(self, crop: str, latitude: float, longitude: float, weather_hash: str, advisory_data: Dict[str, Any], ttl: int = 43200, village_id: Optional[int] = None):
        key = advisory_cache.get_key(crop, latitude, longitude, weather_hash, village_id)
        self.set(key, advisory_data, ttl)

    def get_translation(self, english_json: Dict[str, Any], language: str) -> Optional[Dict[str, Any]]:
        key = translation_cache.get_key(english_json, language)
        return self.get(key)

    def set_translation(self, english_json: Dict[str, Any], language: str, translated_data: Dict[str, Any], ttl: int = 43200):
        key = translation_cache.get_key(english_json, language)
        self.set(key, translated_data, ttl)


# 5. AdvisoryCache
class AdvisoryCache:
    def __init__(self, manager: CacheManager):
        self.manager = manager

    def get_key(self, crop: str, latitude: float, longitude: float, weather_hash: str, village_id: Optional[int] = None) -> str:
        if village_id is not None:
            location_identifier = str(village_id)
        else:
            lat_grid = f"{latitude:.3f}"
            lon_grid = f"{longitude:.3f}"
            location_identifier = f"{lat_grid}_{lon_grid}"
        
        crop_clean = crop.lower().strip()
        return f"advisory:{location_identifier}:{crop_clean}:{weather_hash}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.manager.get(key)

    def set(self, key: str, advisory_data: Dict[str, Any], ttl: int = 43200):
        self.manager.set(key, advisory_data, ttl)

    def ttl(self, key: str) -> Optional[int]:
        return self.manager.ttl(key)


# 6. TranslationCache
class TranslationCache:
    def __init__(self, manager: CacheManager):
        self.manager = manager

    def get_key(self, english_json: Dict[str, Any], language: str) -> str:
        serialized = json_stable_hash(english_json)
        return f"trans:{serialized}:{language.lower().strip()}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.manager.get(key)

    def set(self, key: str, translated_data: Dict[str, Any], ttl: int = 43200):
        self.manager.set(key, translated_data, ttl)

    def ttl(self, key: str) -> Optional[int]:
        return self.manager.ttl(key)


# 7. LockManager for Async request deduplication
class LockManager:
    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._lock_of_locks = asyncio.Lock()

    async def get_lock(self, key: str) -> asyncio.Lock:
        async with self._lock_of_locks:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]

    async def release_lock(self, key: str):
        async with self._lock_of_locks:
            if key in self._locks:
                lock = self._locks[key]
                if not lock.locked():
                    del self._locks[key]


# Global Shared Instances
cache_manager = CacheManager()
advisory_cache = AdvisoryCache(cache_manager)
translation_cache = TranslationCache(cache_manager)
lock_manager = LockManager()

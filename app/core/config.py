import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

class Settings:
    # FastAPI
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    
    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    
    # LLM Providers Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    ENABLE_GROQ_FALLBACK: bool = os.getenv("ENABLE_GROQ_FALLBACK", "true").lower() in ("true", "1", "yes")
    
    # Gemini
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))

    # Groq
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # Caching
    CACHE_TYPE: str = os.getenv("CACHE_TYPE", "in_memory")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    ADVISORY_CACHE_TTL: int = int(os.getenv("ADVISORY_CACHE_TTL", "43200"))
    
    # Logging
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "TEXT")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()

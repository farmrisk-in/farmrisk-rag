import asyncio
import json
import httpx
from typing import Protocol, Type, TypeVar, Any, Optional
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.errors import APIError
from groq import AsyncGroq
from app.core.config import settings
from app.core.logging import logger

T = TypeVar("T", bound=BaseModel)

class LLMProvider(Protocol):
    async def generate_json(self, prompt: str, schema: Type[T], temperature: float = 0.2) -> T:
        """Generate structured JSON response conforming to the Pydantic schema."""
        ...

    async def generate_text(self, prompt: str, temperature: float = 0.2) -> str:
        """Generate raw plain text response."""
        ...

def is_transient_error(e: Exception) -> bool:
    """Determine if an exception is a transient error that should be retried."""
    # 1. APIError from google-genai SDK
    if isinstance(e, APIError):
        status_code = getattr(e, "code", getattr(e, "status_code", None))
        if status_code in [429, 500, 502, 503, 504]:
            return True
        # Check string representation if code/status_code not found
        err_str = str(e)
        for code in ["429", "500", "502", "503", "504"]:
            if code in err_str:
                return True
                
    # 2. Timeout and connection errors
    if isinstance(e, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError, TimeoutError, asyncio.TimeoutError)):
        return True
        
    # Check general error strings for transient issues
    err_str = str(e).lower()
    for term in ["timeout", "timed out", "connection", "network", "temporary", "unavailable", "rate limit", "resource exhausted", "deadline exceeded"]:
        if term in err_str:
            # Do NOT retry authentication or client-side errors
            if any(auth_term in err_str for auth_term in ["400", "401", "403", "unauthorized", "api_key", "invalid key", "invalid credential"]):
                return False
            return True
            
    return False

class GeminiProvider:
    def __init__(self):
        self.enabled = bool(settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY != "your_google_api_key")
        if self.enabled:
            self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        else:
            logger.warning("GOOGLE_API_KEY not configured. GeminiProvider is running in disabled/mock state.")

    async def generate_json(self, prompt: str, schema: Type[T], temperature: float = 0.2) -> T:
        if not self.enabled:
            raise ValueError("Gemini API key not configured or provider disabled.")
            
        attempts = 3
        backoffs = [1.0, 2.0, 4.0]
        
        for attempt in range(1, attempts + 1):
            try:
                logger.info(f"Gemini attempt {attempt}/{attempts}")
                response = self.client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=temperature,
                    )
                )
                data = json.loads(response.text)
                return schema.model_validate(data)
            except Exception as e:
                logger.warning(f"Gemini attempt {attempt} failed: {e}")
                
                # Check if this is a transient error and we have attempts remaining
                if attempt < attempts and is_transient_error(e):
                    wait_time = backoffs[attempt - 1]
                    logger.info(f"Gemini retrying after transient error {e}. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    if attempt == attempts:
                        logger.error("Gemini retries exhausted.")
                        if is_transient_error(e):
                            wait_time = backoffs[attempt - 1]
                            logger.info(f"Waiting {wait_time}s after final failure...")
                            await asyncio.sleep(wait_time)
                    raise e

    async def generate_text(self, prompt: str, temperature: float = 0.2) -> str:
        if not self.enabled:
            raise ValueError("Gemini API key not configured or provider disabled.")
            
        attempts = 3
        backoffs = [1.0, 2.0, 4.0]
        
        for attempt in range(1, attempts + 1):
            try:
                logger.info(f"Gemini attempt {attempt}/{attempts} (text generation)")
                response = self.client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                    )
                )
                return response.text
            except Exception as e:
                logger.warning(f"Gemini text attempt {attempt} failed: {e}")
                if attempt < attempts and is_transient_error(e):
                    wait_time = backoffs[attempt - 1]
                    logger.info(f"Gemini retrying after transient error {e}. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    if attempt == attempts:
                        logger.error("Gemini retries exhausted.")
                        if is_transient_error(e):
                            wait_time = backoffs[attempt - 1]
                            logger.info(f"Waiting {wait_time}s after final failure...")
                            await asyncio.sleep(wait_time)
                    raise e


class GroqProvider:
    def __init__(self):
        self.enabled = bool(settings.GROQ_API_KEY)
        if self.enabled:
            self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        else:
            logger.warning("GROQ_API_KEY not configured. GroqProvider is running in disabled/mock state.")

    async def generate_json(self, prompt: str, schema: Type[T], temperature: float = 0.2) -> T:
        if not self.enabled:
            raise ValueError("Groq API key not configured or provider disabled.")
            
        chat_completion = await self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=settings.GROQ_MODEL,
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        content = chat_completion.choices[0].message.content
        data = json.loads(content)
        return schema.model_validate(data)

    async def generate_text(self, prompt: str, temperature: float = 0.2) -> str:
        if not self.enabled:
            raise ValueError("Groq API key not configured or provider disabled.")
            
        chat_completion = await self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=settings.GROQ_MODEL,
            temperature=temperature,
        )
        return chat_completion.choices[0].message.content


def get_primary_provider() -> LLMProvider:
    """Factory to get the primary LLM provider based on settings."""
    if settings.LLM_PROVIDER.lower() == "groq":
        return GroqProvider()
    return GeminiProvider()

def get_fallback_provider() -> Optional[LLMProvider]:
    """Get the fallback provider (Groq) if configured and primary is Gemini."""
    if settings.LLM_PROVIDER.lower() == "gemini" and settings.ENABLE_GROQ_FALLBACK:
        return GroqProvider()
    return None

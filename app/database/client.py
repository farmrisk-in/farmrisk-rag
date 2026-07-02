import os
from supabase import create_client, Client
from app.core.config import settings

def get_supabase_client() -> Client:
    """Returns a Supabase client configured with the service role key."""
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise ValueError("Supabase URL and Service Role Key must be set in environment variables.")
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

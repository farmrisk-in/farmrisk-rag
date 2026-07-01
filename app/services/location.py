import json
from pathlib import Path
from typing import List, Optional, Protocol
import httpx
from app.models.schemas import LocationDetail
from app.core.config import settings

# Load canonical states at startup
BASE_DIR = Path(__file__).resolve().parent.parent.parent
with open(BASE_DIR / "config" / "states.json", "r", encoding="utf-8") as f:
    CANONICAL_STATES = json.load(f)

# Helper to normalize resolved state names to the canonical ones
def normalize_state(resolved_state: str) -> str:
    cleaned = resolved_state.lower().strip().replace("&", "and").replace(",", "")
    for state in CANONICAL_STATES:
        state_clean = state.lower().replace("&", "and").replace(",", "")
        if cleaned == state_clean or state_clean in cleaned or cleaned in state_clean:
            return state
    # Fallback if no match is found
    return resolved_state

class LocationResolver(Protocol):
    async def search(self, query: str) -> List[LocationDetail]:
        """Search location names and autocomplete to details."""
        ...

    async def reverse_geocode(self, latitude: float, longitude: float) -> LocationDetail:
        """Resolve coordinates to detailed location schema."""
        ...

class NominatimLocationResolver:
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=10.0,
            headers={"User-Agent": "FarmRiskAI-App/1.0 (contact@farmrisk.ai)"}
        )

    async def search(self, query: str) -> List[LocationDetail]:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": query,
            "format": "json",
            "countrycodes": "in",  # India only
            "addressdetails": "1",
            "accept-language": "en"
        }
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            results = response.json()
            
            locations = []
            for item in results:
                address = item.get("address", {})
                
                # Derive village/settlement
                village = (
                    address.get("village") or 
                    address.get("town") or 
                    address.get("suburb") or 
                    address.get("hamlet") or 
                    address.get("neighbourhood") or 
                    address.get("municipality") or 
                    address.get("city")
                )
                
                # Derive district
                district = address.get("district") or address.get("county") or address.get("state_district")
                
                # Derive state
                state_raw = address.get("state")
                if not state_raw:
                    continue
                state = normalize_state(state_raw)
                
                locations.append(LocationDetail(
                    village=village,
                    district=district,
                    state=state,
                    latitude=float(item["lat"]),
                    longitude=float(item["lon"])
                ))
            return locations
            
        except Exception as e:
            # Return empty list in case of errors
            print(f"Error calling Nominatim search: {e}")
            return []

    async def reverse_geocode(self, latitude: float, longitude: float) -> LocationDetail:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": latitude,
            "lon": longitude,
            "format": "json",
            "addressdetails": "1",
            "accept-language": "en"
        }
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            address = data.get("address", {})
            
            village = (
                address.get("village") or 
                address.get("town") or 
                address.get("suburb") or 
                address.get("hamlet") or 
                address.get("neighbourhood") or 
                address.get("municipality") or 
                address.get("city") or
                "Unknown Village"
            )
            
            district = address.get("district") or address.get("county") or address.get("state_district") or "Unknown District"
            
            state_raw = address.get("state") or "Rajasthan"  # Default fallback state
            state = normalize_state(state_raw)
            
            return LocationDetail(
                village=village,
                district=district,
                state=state,
                latitude=latitude,
                longitude=longitude
            )
            
        except Exception as e:
            # Fallback in case Nominatim reverse lookup fails
            print(f"Error calling Nominatim reverse geocode: {e}")
            return LocationDetail(
                village="Unknown Village",
                district="Unknown District",
                state="Rajasthan",  # Safe fallback for Pinecone query
                latitude=latitude,
                longitude=longitude
            )

    async def close(self):
        await self.client.aclose()

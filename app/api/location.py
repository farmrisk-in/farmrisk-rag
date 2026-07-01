from fastapi import APIRouter, Query, HTTPException
from typing import List
from app.models.schemas import LocationDetail
from app.services.location import NominatimLocationResolver

router = APIRouter(prefix="/api/location", tags=["Location"])
resolver = NominatimLocationResolver()

@router.get("/search", response_model=List[LocationDetail])
async def search_locations(q: str = Query(..., min_length=2, description="Search term for village/city/town")):
    """
    Search for villages and towns in India via Nominatim.
    Returns details (village, district, state, latitude, longitude)
    """
    try:
        results = await resolver.search(q)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Location search failed: {str(e)}")

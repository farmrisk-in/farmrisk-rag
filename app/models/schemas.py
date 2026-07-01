from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class LocationDetail(BaseModel):
    village: Optional[str] = Field(None, description="Name of the village, town, or settlement")
    district: Optional[str] = Field(None, description="Name of the district or county")
    state: str = Field(..., description="Standardized name of the Indian State")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")

class AdvisoryRequest(BaseModel):
    latitude: float = Field(..., description="Latitude coordinate of the farmer's location")
    longitude: float = Field(..., description="Longitude coordinate of the farmer's location")
    crop: str = Field(..., description="Name of the crop (e.g. Cotton, Rice)")
    language: str = Field(..., description="Target language (e.g. Gujarati, Hindi, Marathi, Tamil, English)")
    crop_stage: Optional[str] = Field(None, description="Optional current crop stage (e.g. vegetative, flowering, maturity)")

class AdvisoryResponse(BaseModel):
    advisory_summary: str = Field(..., description="Two-paragraph professional agrometeorological advisory summary.")

class TranslationResult(BaseModel):
    data: Dict[str, Any]
    translated: bool
    provider: Optional[str] = None



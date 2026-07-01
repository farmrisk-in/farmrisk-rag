import json
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.models.schemas import AdvisoryResponse
from app.llm.providers import get_primary_provider, get_fallback_provider
from app.core.logging import logger
from datetime import datetime

class AdvisoryEngine:
    def __init__(self):
        pass

    async def generate_advisory(
        self,
        crop: str,
        state: str,
        district: str,
        village: str,
        season: str,
        weather_data: Dict[str, Any],
        rag_context: List[Dict[str, Any]],
        crop_stage: Optional[str] = None
    ) -> AdvisoryResponse:
        """Generate a two-paragraph plain-text agrometeorological advisory using LLMProvider and RAG context."""
        
        # Format the context from Pinecone
        formatted_context = ""
        for idx, item in enumerate(rag_context, start=1):
            formatted_context += f"Source [{item['source']} page {item['page']}]:\n{item['content']}\n\n"
            
        # Format the weather data
        formatted_weather = json.dumps(weather_data["forecast"], indent=2)
        weather_risks = ", ".join(weather_data["risks"]) if weather_data["risks"] else "None"
        
        # Extract forecast period dates
        # start_date = weather_data["forecast"][0]["date"] if weather_data["forecast"] else "N/A"
        # end_date = weather_data["forecast"][-1]["date"] if weather_data["forecast"] else "N/A"
        start_date = datetime.strptime(weather_data["forecast"][0]["date"], "%Y-%m-%d").strftime("%d/%m/%Y")
        end_date = datetime.strptime(weather_data["forecast"][-1]["date"], "%Y-%m-%d").strftime("%d/%m/%Y")
        
        crop_stage_str = f"Crop Stage: {crop_stage}" if crop_stage else "Crop Stage: Not specified (infer from the current season and date)"

        prompt = f"""
You are an expert agrometeorologist assistant at FarmRisk.
Generate a professional, extension-style agricultural advisory bulletin for a farmer growing {crop} in {village}, {district}, {state} during the {season} season.

CROP PARAMETERS:
- Crop: {crop}
- {crop_stage_str}

WEATHER PARAMETERS (10-Day Forecast from {start_date} to {end_date}):
{formatted_weather}

WEATHER RISK ALERTS:
{weather_risks}

AGRICULTURAL ADVISORY CONTEXT (Retrieved ICAR Scientific guidelines):
{formatted_context}

INSTRUCTIONS:
Generate exactly one advisory consisting of exactly two paragraphs in plain text.
- Always use indian date stamp format (DD/MM/YYYY).
- Always highlight the important words and numbers using asterisks. Like this: *word* and *number*
- Do NOT return JSON.
- Do NOT use headings.
- Do NOT use bullet points.
- Do NOT use numbering.
- The total length of both paragraphs combined MUST be between 120 and 180 words.
- All text in the response must be written in English.

Paragraph 1: Weather and Crop Impact
- Must begin exactly with: "From {start_date} to {end_date}, "
- Describe expected weather conditions and crop/soil impacts over the 10-day period.
- Include cumulative rainfall (mm), rainfall pattern (light, moderate, or heavy), and rainfall-related risks.
- Include minimum and maximum temperature range.
- Include wind conditions, humidity, and expected soil moisture trend.
- Describe the expected impact on the selected crop ({crop}).
- This paragraph must describe ONLY weather conditions and crop impacts. Do NOT include recommendations, actions, or guidelines here.

Paragraph 2: Crop Advisory
- Provide practical crop-specific agricultural recommendations.
- Recommendations must be based ONLY on the weather forecast, crop stage, and the retrieved ICAR advisory context. Never invent recommendations or hallucinate info.
- Include guidance where applicable for: irrigation, sowing or transplanting, fertilizer timing, pesticide spraying (e.g., matching wind conditions), drainage management, pest/disease monitoring, and harvesting.
- End the paragraph with exactly one concluding sentence stating the overall agricultural outlook:
  - "Overall, the agricultural outlook for this period is Favorable."
  - "Overall, the agricultural outlook for this period is Cautionary."
  - "Overall, the agricultural outlook for this period is Unfavorable."
  Choose the single option that best matches the weather and crop impact.
"""

        primary_provider = get_primary_provider()
        fallback_provider = get_fallback_provider()
        
        try:
            raw_text = await primary_provider.generate_text(
                prompt=prompt,
                temperature=settings.TEMPERATURE
            )
            logger.info(f"{settings.LLM_PROVIDER} advisory succeeded")
            return AdvisoryResponse(advisory_summary=raw_text.strip())
        except Exception as primary_exc:
            logger.warning(f"Primary provider {settings.LLM_PROVIDER} failed to generate advisory: {primary_exc}")
            
            if fallback_provider:
                logger.info("Switching to Groq fallback for advisory generation")
                try:
                    raw_text = await fallback_provider.generate_text(
                        prompt=prompt,
                        temperature=settings.TEMPERATURE
                    )
                    logger.info("Groq advisory succeeded")
                    return AdvisoryResponse(advisory_summary=raw_text.strip())
                except Exception as fallback_exc:
                    logger.error(f"Fallback provider Groq failed to generate advisory: {fallback_exc}")
            
            logger.warning("All LLM providers failed to generate advisory. Using local English mock fallback.")
            return self._get_mock_advisory(crop, village, start_date, end_date)

    def _get_mock_advisory(self, crop: str, village: str, start_date: str, end_date: str) -> AdvisoryResponse:
        """Fallback mock advisory for local testing without Gemini credentials."""
        paragraph_1 = f"From {start_date} to {end_date}, the region of {village} is expected to experience a cumulative rainfall of 25 mm, characterized by a light and intermittent rainfall pattern that poses minimal immediate flood risks. Maximum temperatures will peak around 36°C while minimums drop to 23°C. These conditions will maintain moderate soil moisture trends, which is highly beneficial for the active vegetative growth phase of {crop} but may also encourage early weed emergence."
        paragraph_2 = f"Based on the weather forecast and standard guidelines, farmers should optimize irrigation schedules by pausing watering on days with light showers and ensuring active weeding. Apply nitrogenous fertilizers during dry breaks and monitor the crop closely for sucking pests and fungal leaf spots, ensuring that drainage channels are completely clear of debris. Overall, the agricultural outlook for this period is Favorable."
        
        advisory_text = f"{paragraph_1}\n\n{paragraph_2}"
        return AdvisoryResponse(advisory_summary=advisory_text)


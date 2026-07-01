from datetime import datetime
from typing import Protocol

class SeasonResolver(Protocol):
    def resolve_season(self, latitude: float, longitude: float, date: datetime = None) -> str:
        """Resolve agricultural season (Kharif or Rabi) for given coordinates and date."""
        ...

class MonthSeasonResolver:
    """Standard month-based agricultural season detector for India."""
    
    def resolve_season(self, latitude: float, longitude: float, date: datetime = None) -> str:
        if date is None:
            date = datetime.now()
            
        month = date.month
        
        # In India, Kharif season is June - October; Rabi season is November - May
        if 6 <= month <= 10:
            return "Kharif"
        else:
            return "Rabi"

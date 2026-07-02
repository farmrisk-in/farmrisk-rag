from typing import List, Dict, Any
from app.database.client import get_supabase_client

class PgVectorStore:
    def __init__(self, table_name: str = "advisories"):
        self.client = get_supabase_client()
        self.table_name = table_name

    def similarity_search(self, query_vector: List[float], filter_metadata: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Calls a Supabase RPC function for similarity search.
        The RPC should be named `match_advisories`.
        """
        response = self.client.rpc("match_advisories", {
            "query_embedding": query_vector,
            "match_threshold": -1.0,  # Include everything and rely on LIMIT
            "match_count": top_k,
            "p_crop": filter_metadata.get("crop"),
            "p_state": filter_metadata.get("state"),
            "p_season": filter_metadata.get("season")
        }).execute()

        return response.data if response.data else []

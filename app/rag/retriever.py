from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from app.database.pgvector import PgVectorStore

class AdvisoryRetriever:
    def __init__(self):
        # Local SentenceTransformer
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
        # Supabase pgvector store
        self.store = PgVectorStore()

    def retrieve(self, crop: str, state: str, season: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant context chunks from Supabase using metadata pre-filtering."""
        query_text = f"Advisory and recommendations for growing {crop} in {state} during {season} season."
        query_vector = self.model.encode(query_text, normalize_embeddings=True).tolist()
        
        # Stage 1: Strict metadata filter
        meta_filter = {
            "crop": crop,
            "state": state,
            "season": season
        }
        
        try:
            matches = self.store.similarity_search(query_vector, meta_filter, top_k)
            
            # Stage 2: Fallback (if no results for specific state, query national/wider region by crop & season)
            if not matches:
                fallback_filter = {
                    "crop": crop,
                    "state": None, # Setting state to None will trigger the fallback in SQL
                    "season": season
                }
                matches = self.store.similarity_search(query_vector, fallback_filter, top_k)
                
            results = []
            for match in matches:
                results.append({
                    "id": match.get("id"),
                    "score": match.get("similarity"), # Map similarity to score
                    "content": match.get("content", ""),
                    "state": match.get("state"),
                    "crop": match.get("crop"),
                    "season": match.get("season"),
                    "page": match.get("page"),
                    "source": match.get("source")
                })
            return results
            
        except Exception as e:
            print(f"Error querying Supabase index: {e}")
            return []

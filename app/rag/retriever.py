from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from app.core.config import settings

class AdvisoryRetriever:
    def __init__(self):
        # Local SentenceTransformer
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
        # Pinecone
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)

    def retrieve(self, crop: str, state: str, season: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant context chunks from Pinecone using metadata pre-filtering."""
        query_text = f"Advisory and recommendations for growing {crop} in {state} during {season} season."
        query_vector = self.model.encode(query_text, normalize_embeddings=True).tolist()
        
        # Stage 1: Strict metadata filter
        meta_filter = {
            "crop": crop,
            "state": state,
            "season": season
        }
        
        try:
            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=meta_filter,
                include_metadata=True
            )
            
            matches = response.get("matches", [])
            
            # Stage 2: Fallback (if no results for specific state, query national/wider region by crop & season)
            if not matches:
                # Fallback filter omitting the state to capture national guidelines
                fallback_filter = {
                    "crop": crop,
                    "season": season
                }
                response = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    filter=fallback_filter,
                    include_metadata=True
                )
                matches = response.get("matches", [])
                
            results = []
            for match in matches:
                metadata = match.get("metadata", {})
                results.append({
                    "id": match.get("id"),
                    "score": match.get("score"),
                    "content": metadata.get("content", ""),
                    "state": metadata.get("state"),
                    "crop": metadata.get("crop"),
                    "season": metadata.get("season"),
                    "page": metadata.get("page"),
                    "source": metadata.get("source")
                })
            return results
            
        except Exception as e:
            print(f"Error querying Pinecone index: {e}")
            return []

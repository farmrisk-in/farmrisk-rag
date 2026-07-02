import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

# Load Environment Variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

PARSED_DIR = BASE_DIR / "data" / "parsed"

def main():
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("Error: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is not set in environment variables.")
        return

    # Load Chunks
    chunks_file = PARSED_DIR / "chunks.json"
    if not chunks_file.exists():
        print(f"Chunks file {chunks_file} not found!")
        return

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("No chunks to upload.")
        return

    print(f"Loading SentenceTransformer model 'BAAI/bge-small-en-v1.5'...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    print("Connecting to Supabase...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    
    print(f"Embedding and uploading {len(chunks)} chunks in batches of 100...")
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # Prepare contents to embed
        texts = [item["content"] for item in batch]
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        upsert_data = []
        for idx, item in enumerate(batch):
            # Create a unique ID
            safe_state = item["state"].lower().replace(" ", "_").replace("&", "and")
            safe_crop = item["crop"].lower().replace(" ", "_")
            safe_season = item["season"].lower()
            content_hash = hashlib.md5(item['content'].encode('utf-8')).hexdigest()[:8]
            unique_id = f"{safe_state}_{safe_crop}_{safe_season}_{item.get('chunk_id', idx)}_{content_hash}"
            
            upsert_data.append({
                "id": unique_id,
                "content": item["content"],
                "embedding": embeddings[idx].tolist(),
                "crop": item["crop"],
                "state": item["state"],
                "season": item["season"],
                "source": item["source"],
                "page": item["page"],
                "metadata": {
                    "category": item["category"]
                }
            })
            
        # Insert into Supabase (using upsert to avoid duplicate key errors on re-runs)
        try:
            supabase.table("advisories").upsert(upsert_data).execute()
            print(f"  Upserted batch {i//batch_size + 1}/{len(chunks)//batch_size + 1} (items {i} to {i + len(batch)})")
        except Exception as e:
            print(f"  Error upserting batch {i//batch_size + 1}: {e}")

    print("Supabase upload complete!")

if __name__ == "__main__":
    main()

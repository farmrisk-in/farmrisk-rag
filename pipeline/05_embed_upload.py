import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Load Environment Variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

PARSED_DIR = BASE_DIR / "data" / "parsed"

def main():
    if not PINECONE_API_KEY or PINECONE_API_KEY == "your_pinecone_api_key":
        print("Error: PINECONE_API_KEY is not set in environment variables.")
        return
    if not PINECONE_INDEX_NAME or PINECONE_INDEX_NAME == "farmrisk-advisories":
        print("Error: PINECONE_INDEX_NAME is not set in environment variables.")
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
    
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if it does not exist
    existing_indexes = [index.name for index in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating a new index...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Index '{PINECONE_INDEX_NAME}' created successfully.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")

    index = pc.Index(PINECONE_INDEX_NAME)

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
            unique_id = f"{safe_state}_{safe_crop}_{safe_season}_{item['chunk_id']}"
            
            upsert_data.append((
                unique_id,
                embeddings[idx].tolist(),
                {
                    "crop": item["crop"],
                    "state": item["state"],
                    "season": item["season"],
                    "source": item["source"],
                    "page": item["page"],
                    "category": item["category"]
                }
            ))
            
        index.upsert(vectors=upsert_data)
        print(f"  Upserted batch {i//batch_size + 1}/{len(chunks)//batch_size + 1} (items {i} to {i + len(batch)})")

    print("Pinecone upload complete!")

if __name__ == "__main__":
    main()

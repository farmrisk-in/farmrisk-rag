import os
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

def download_models():
    print(f"Pre-downloading {EMBEDDING_MODEL} model for cache baking...")
    SentenceTransformer(EMBEDDING_MODEL)
    print("Model downloaded successfully.")

if __name__ == "__main__":
    download_models()


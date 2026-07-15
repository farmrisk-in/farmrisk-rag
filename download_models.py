from sentence_transformers import SentenceTransformer

def download_models():
    print("Pre-downloading BAAI/bge-small-en-v1.5 model for cache baking...")
    SentenceTransformer("BAAI/bge-small-en-v1.5")
    print("Model downloaded successfully.")

if __name__ == "__main__":
    download_models()

import json
import tiktoken
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PARSED_DIR = BASE_DIR / "data" / "parsed"

# Tokenizer
TOKENIZER_NAME = "cl100k_base"  # Standard GPT-4/Gemini cl100k_base
encoding = tiktoken.get_encoding(TOKENIZER_NAME)

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def chunk_text(text: str, target_size: int = 700, overlap: int = 100) -> list:
    """Split text into chunks of target_size with overlap using tokens."""
    tokens = encoding.encode(text)
    total_tokens = len(tokens)
    
    if total_tokens <= 800:
        return [text]
        
    chunks = []
    start = 0
    step = target_size - overlap
    
    while start < total_tokens:
        end = min(start + target_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # If we reached the end, break
        if end == total_tokens:
            break
            
        start += step
        
    return chunks

def main():
    input_file = PARSED_DIR / "valid_advisories.json"
    if not input_file.exists():
        print(f"Valid advisories file {input_file} not found!")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    all_chunks = []
    
    for r in records:
        content = r["content"]
        chunks = chunk_text(content, target_size=700, overlap=100)
        
        for idx, chunk_content in enumerate(chunks, start=1):
            chunk_record = {
                "season": r["season"],
                "source": r["source"],
                "state": r["state"],
                "category": r["category"],
                "crop": r["crop"],
                "page": r["page"],
                "chunk_id": idx,
                "content": chunk_content.strip(),
                "token_count": count_tokens(chunk_content)
            }
            all_chunks.append(chunk_record)
            
    output_file = PARSED_DIR / "chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
    print(f"Chunking summary:")
    print(f"  Total source advisories: {len(records)}")
    print(f"  Generated chunks: {len(all_chunks)}")
    print(f"  Saved to: {output_file}")

if __name__ == "__main__":
    main()

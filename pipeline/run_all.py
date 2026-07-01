import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def run_script(script_name: str):
    script_path = BASE_DIR / "pipeline" / script_name
    print(f"\n==========================================")
    print(f"Running: {script_name}")
    print(f"==========================================\n")
    
    result = subprocess.run([sys.executable, str(script_path)], cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"\nError: {script_name} failed with return code {result.returncode}")
        sys.exit(result.returncode)

def main():
    # Note: 01_extract.py is omitted as it has already been executed to dump raw text JSONs.
    run_script("02_parse.py")
    run_script("03_validate.py")
    run_script("04_chunk.py")
    run_script("05_embed_upload.py")
    print("\nFull Ingestion Pipeline completed successfully!")

if __name__ == "__main__":
    main()

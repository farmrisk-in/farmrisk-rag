"""
pipeline/01_extract.py
====================================================================
Extracts text page-by-page from raw PDF advisories in Advisories_PDF/
using PyMuPDF.

Features:
- Preserves existing extracted files (does not delete/overwrite old data).
- Automatically maps subfolder names (Gujarat, Punjab, Tamilnadu, India) to state metadata.
- Detects digital text vs scanned/empty pages.
- Saves structured JSON outputs into data/extracted/<State>/<filename>.json.
- Generates a comprehensive manifest.json report of the extraction.
"""

import json
import time
from pathlib import Path
import pymupdf
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "Advisories_PDF"
OUTPUT_DIR = BASE_DIR / "data" / "extracted"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_pdf(pdf_path: Path, output_path: Path, state: str) -> dict:
    """Extract page text from a single PDF and save structured JSON."""
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)

    pages = []
    text_pages = 0
    scanned_pages = 0
    total_chars = 0

    for page_idx, page in enumerate(doc, start=1):
        text = page.get_text("text")
        char_count = len(text.strip())
        is_empty = char_count < 10

        if is_empty:
            scanned_pages += 1
        else:
            text_pages += 1

        total_chars += char_count

        pages.append({
            "page": page_idx,
            "text": text,
            "char_count": char_count,
            "is_scanned_or_empty": is_empty
        })

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    status = "SUCCESS"
    if scanned_pages == total_pages:
        status = "SCANNED_NEEDS_OCR"
    elif scanned_pages > 0:
        status = "PARTIAL_TEXT"

    return {
        "file_name": pdf_path.name,
        "state": state,
        "rel_path": str(pdf_path.relative_to(PDF_DIR)),
        "output_file": str(output_path.relative_to(BASE_DIR)),
        "total_pages": total_pages,
        "text_pages": text_pages,
        "scanned_pages": scanned_pages,
        "total_chars": total_chars,
        "status": status
    }


def main():
    if not PDF_DIR.exists():
        print(f"Error: {PDF_DIR} directory not found.")
        return

    pdf_files = sorted(PDF_DIR.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}.")
        return

    print(f"==================================================")
    print(f"Starting Extraction of {len(pdf_files)} PDFs from {PDF_DIR.name}/")
    print(f"==================================================\n")

    manifest = []
    t0 = time.time()

    for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
        # Derive state from parent folder name
        rel_parts = pdf_path.relative_to(PDF_DIR).parts
        state = rel_parts[0] if len(rel_parts) > 1 else "Unknown"

        # Output path preserving state folder structure
        out_state_dir = OUTPUT_DIR / state
        output_file = out_state_dir / f"{pdf_path.stem}.json"

        result = extract_pdf(pdf_path, output_file, state)
        manifest.append(result)

    elapsed = time.time() - t0
    total_pages = sum(m["total_pages"] for m in manifest)
    total_chars = sum(m["total_chars"] for m in manifest)
    text_pages = sum(m["text_pages"] for m in manifest)
    scanned_pages = sum(m["scanned_pages"] for m in manifest)

    # Save extraction manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_files": len(manifest),
            "total_pages": total_pages,
            "text_pages": text_pages,
            "scanned_pages": scanned_pages,
            "total_characters": total_chars,
            "elapsed_seconds": round(elapsed, 2),
            "files": manifest
        }, f, indent=2, ensure_ascii=False)

    print("\n==================================================")
    print(f"Extraction Completed in {elapsed:.2f}s")
    print(f"Total PDFs Processed   : {len(manifest)}")
    print(f"Total Pages Extracted  : {total_pages} (Digital: {text_pages}, Scanned/Empty: {scanned_pages})")
    print(f"Total Characters       : {total_chars:,}")
    print(f"Manifest written to    : {manifest_path.relative_to(BASE_DIR)}")
    print(f"==================================================")


if __name__ == "__main__":
    main()

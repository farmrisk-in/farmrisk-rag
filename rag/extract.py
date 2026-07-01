from pathlib import Path
import fitz
import json
from tqdm import tqdm

PDF_FOLDER = Path("data/pdfs")
OUTPUT_FOLDER = Path("data/extracted")

OUTPUT_FOLDER.mkdir(exist_ok=True)

for pdf_file in PDF_FOLDER.glob("*.pdf"):

    print(f"Reading {pdf_file.name}")

    pdf = fitz.open(pdf_file)

    pages = []

    for page_number, page in enumerate(tqdm(pdf), start=1):

        text = page.get_text("text")

        pages.append({
            "page": page_number,
            "text": text
        })

    output_file = OUTPUT_FOLDER / f"{pdf_file.stem}.json"

    with open(output_file, "w", encoding="utf-8") as f:

        json.dump(
            pages,
            f,
            ensure_ascii=False,
            indent=2
        )

print("Done")
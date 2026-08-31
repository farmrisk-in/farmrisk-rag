"""
pipeline/02_parse.py
====================================================================
Unified Parsing and Metadata Tagging Engine.

Combines:
1. All-India ICAR Kharif & Rabi advisories (Baseline)
2. Punjab Agricultural University (PAU) Kharif, Rabi & Vegetable guides
3. Tamil Nadu Agricultural University (TNAU) Agriculture & Horticulture CPG
4. Gujarat Directorate of Agriculture Crop Manuals (with Gujarati->Canonical mapping)

Outputs: data/parsed/advisories.json
"""

import json
import re
from pathlib import Path
from tqdm import tqdm

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = BASE_DIR / "data" / "extracted"
PARSED_DIR = BASE_DIR / "data" / "parsed"
CONFIG_DIR = BASE_DIR / "config"

PARSED_DIR.mkdir(parents=True, exist_ok=True)

# Load Canonical Lists
with open(CONFIG_DIR / "crops.json", "r", encoding="utf-8") as f:
    CANONICAL_CROPS = json.load(f)

with open(CONFIG_DIR / "states.json", "r", encoding="utf-8") as f:
    CANONICAL_STATES = json.load(f)

# Gujarati Crop Name Mapping to Canonical English
GUJARATI_CROPS = {
    "ઘઉં": "Wheat",
    "ડાંગર": "Rice",
    "ડાાંગર": "Rice",
    "ચોખા": "Rice",
    "કપાસ": "Cotton",
    "શેરડી": "Sugarcane",
    "દિવેલા": "Castor",
    "મગફળી": "Groundnut",
    "તલ": "Sesame",
    "રાઈ": "Mustard",
    "રાય": "Mustard",
    "સોયાબીન": "Soybean",
    "મગ": "Moong",
    "અડદ": "Urad",
    "તુવેર": "Tur",
    "તૂવેર": "Tur",
    "ચણા": "Chickpea",
    "બાજરી": "Bajra",
    "મકાઈ": "Maize",
    "જુવાર": "Jowar",
    "વાલ": "Cowpea",
}

CATEGORIES = [
    "CEREAL CROPS", "CEREALS", "PULSES", "PULSE CROPS", "OILSEEDS", 
    "OILSEED CROPS", "FRUIT AND VEGETABLE CROPS", "FRUIT & VEGETABLE CROPS", 
    "VEGETABLE CROPS", "VEGETABLES", "FRUIT CROPS", "HORTICULTURE", 
    "COMMERCIAL CROPS", "CASH CROPS", "FIBRE CROPS", "FODDER CROPS",
    "SUGAR CROPS"
]


def clean_text(text: str) -> str:
    """Normalize ligatures, soft hyphens, and strip solitary page numbers."""
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\ufb01", "fi").replace("ϐ", "fi").replace("ϐield", "field")
    text = text.replace("\ufb02", "fl")
    text = text.replace("ﬀ", "ff").replace("ﬁ", "fi").replace("ﬂ", "fl")
    
    # Remove repeated ICAR headers
    text = re.sub(r"(ICAR\s+KHARIF\s+AGRO-ADVISORY|ICAR\s+RABI\s+AgRo-AdvIsoRy\s+foR\s+fARmeRs|AgRo-AdvIsoRy\s+foR\s+fARmeRs)", "", text, flags=re.I)
    
    lines = []
    for line in text.split("\n"):
        line_strip = line.strip()
        if line_strip.isdigit():
            continue
        if not line_strip:
            continue
        lines.append(line_strip)
    
    return "\n".join(lines)


def detect_state(line: str) -> str:
    line_clean = line.strip().lower().replace("&", "and").replace(",", "")
    for state in CANONICAL_STATES:
        state_clean = state.lower().replace("&", "and").replace(",", "")
        if line_clean == state_clean or line_clean.startswith(state_clean + " "):
            return state
    return None


def detect_crop(line: str) -> str:
    line_clean = line.strip().rstrip(":").strip().lower()
    
    # Direct exact matches
    for crop in CANONICAL_CROPS:
        if line_clean == crop.lower():
            return crop
            
    # Composite crop names e.g. "Mash (Black Gram)"
    for crop in CANONICAL_CROPS:
        crop_clean = crop.lower()
        if "(" in crop_clean:
            parts = re.findall(r'\b[a-z\s]+\b', crop_clean)
            for part in parts:
                part = part.strip()
                if len(part) > 3 and line_clean == part:
                    return crop
                    
    # Common synonyms
    synonyms = {
        "paddy": "Rice",
        "moong": "Moong",
        "moong (green gram)": "Moong",
        "green gram": "Green gram",
        "urad": "Urad",
        "black gram": "Black gram",
        "gram": "Chickpea",
        "bengalgram": "Chickpea",
        "redgram": "Tur",
        "pigeonpea": "Tur",
        "arhar": "Tur",
        "cumbu": "Bajra",
        "pearl millet": "Bajra",
        "sorghum": "Jowar",
        "groundnut": "Groundnut",
        "peanut": "Groundnut",
        "cotton": "Cotton",
        "sugarcane": "Sugarcane",
        "wheat": "Wheat",
        "mustard": "Mustard",
        "raya": "Mustard",
        "sarson": "Mustard",
        "sesame": "Sesame",
        "gingelly": "Sesame",
        "castor": "Castor",
        "sunflower": "Sunflower",
        "soybean": "Soybean",
        "maize": "Maize",
        "potato": "Potato",
        "tomato": "Tomato",
        "onion": "Onion",
        "chilli": "Chilli",
        "brinjal": "Brinjal",
        "eggplant": "Brinjal",
        "banana": "Banana",
        "mango": "Mango",
        "guava": "Guava",
        "grapes": "Grapes",
        "papaya": "Papaya",
        "pomegranate": "Pomegranate",
    }
    
    for syn, canonical in synonyms.items():
        if line_clean == syn or line_clean.startswith(syn + " ") or line_clean.endswith(" " + syn):
            return canonical
            
    return None


def detect_category(line: str) -> str:
    line_clean = line.strip().upper()
    for cat in CATEGORIES:
        if line_clean == cat or line_clean.endswith(" " + cat):
            return cat.title()
    return None


# ==============================================================================
# 1. PARSE ALL-INDIA ICAR (Preserving Baseline Kharif & Rabi)
# ==============================================================================
def parse_icar_kharif():
    print("Parsing ICAR Kharif advisories...")
    # Prefer new extracted path, fall back to baseline
    input_file = EXTRACTED_DIR / "India" / "ICAR En-Kharif Agro-Advisories for Farmers 2025.json"
    if not input_file.exists():
        input_file = EXTRACTED_DIR / "ICAR.json"
    if not input_file.exists():
        print(f"Extraction file for ICAR Kharif not found!")
        return []

    with open(input_file, "r", encoding="utf-8") as f:
        pages = json.load(f)

    records = []
    current_state = None
    current_category = None
    current_crop = None
    buffer = []
    current_page = None

    for page_data in pages:
        page_no = page_data["page"]
        text = clean_text(page_data["text"])
        lines = text.split("\n")

        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                continue

            state = detect_state(line_strip)
            if state:
                if current_crop and buffer:
                    records.append({
                        "season": "Kharif",
                        "source": "ICAR.pdf",
                        "state": current_state,
                        "category": current_category or "Kharif Crops",
                        "crop": current_crop,
                        "page": current_page,
                        "content": " ".join(buffer),
                        "language": "en"
                    })
                    buffer = []
                current_state = state
                current_crop = None
                continue

            category = detect_category(line_strip)
            if category:
                current_category = category
                continue

            crop = detect_crop(line_strip)
            if crop:
                if current_crop and buffer:
                    records.append({
                        "season": "Kharif",
                        "source": "ICAR.pdf",
                        "state": current_state,
                        "category": current_category or "Kharif Crops",
                        "crop": current_crop,
                        "page": current_page,
                        "content": " ".join(buffer),
                        "language": "en"
                    })
                    buffer = []
                current_crop = crop
                current_page = page_no
                continue

            if current_crop:
                buffer.append(line_strip)

    if current_crop and buffer:
        records.append({
            "season": "Kharif",
            "source": "ICAR.pdf",
            "state": current_state,
            "category": current_category or "Kharif Crops",
            "crop": current_crop,
            "page": current_page,
            "content": " ".join(buffer),
            "language": "en"
        })

    return records


def parse_icar_rabi():
    print("Parsing ICAR Rabi advisories...")
    input_file = EXTRACTED_DIR / "India" / "Rabi-Agro-Advisory-2021-22.json"
    if not input_file.exists():
        input_file = EXTRACTED_DIR / "Rabi-Agro-Advisory-2021-22.json"
    if not input_file.exists():
        print(f"Extraction file for ICAR Rabi not found!")
        return []

    with open(input_file, "r", encoding="utf-8") as f:
        pages = json.load(f)

    records = []
    current_zone = None
    current_state = None
    current_crop = None
    buffer = []
    current_page = None

    for page_data in pages:
        page_no = page_data["page"]
        text = clean_text(page_data["text"])
        lines = text.split("\n")

        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                continue

            zone_match = re.search(r'\b(Zone-\w+|\bZone\s+\w+)\b', line_strip, re.I)
            if zone_match:
                current_zone = zone_match.group(1).title()

            state = detect_state(line_strip)
            if state:
                if current_crop and buffer:
                    records.append({
                        "season": "Rabi",
                        "source": "Rabi-Agro-Advisory-2021-22.pdf",
                        "state": current_state,
                        "category": "Rabi Crops",
                        "crop": current_crop,
                        "page": current_page,
                        "content": " ".join(buffer),
                        "zone": current_zone,
                        "language": "en"
                    })
                    buffer = []
                current_state = state
                current_crop = None
                continue

            crop = detect_crop(line_strip)
            if crop:
                if current_crop and buffer:
                    records.append({
                        "season": "Rabi",
                        "source": "Rabi-Agro-Advisory-2021-22.pdf",
                        "state": current_state,
                        "category": "Rabi Crops",
                        "crop": current_crop,
                        "page": current_page,
                        "content": " ".join(buffer),
                        "zone": current_zone,
                        "language": "en"
                    })
                    buffer = []
                current_crop = crop
                current_page = page_no
                continue

            if current_crop:
                buffer.append(line_strip)

    if current_crop and buffer:
        records.append({
            "season": "Rabi",
            "source": "Rabi-Agro-Advisory-2021-22.pdf",
            "state": current_state,
            "category": "Rabi Crops",
            "crop": current_crop,
            "page": current_page,
            "content": " ".join(buffer),
            "zone": current_zone,
            "language": "en"
        })

    return records


# ==============================================================================
# 2. PARSE PUNJAB (PAU Kharif, Rabi, Vegetables)
# ==============================================================================
def parse_punjab():
    print("Parsing Punjab Agricultural University (PAU) advisories...")
    punjab_dir = EXTRACTED_DIR / "Punjab"
    if not punjab_dir.exists():
        return []

    files_config = [
        ("pp_kharif.json", "Kharif", "pp_kharif.pdf", "Kharif Crops"),
        ("pp_rabi.json", "Rabi", "pp_rabi.pdf", "Rabi Crops"),
        ("pp_veg.json", "Annual", "pp_veg.pdf", "Vegetable Crops")
    ]

    records = []

    for fname, season, source, default_category in files_config:
        fpath = punjab_dir / fname
        if not fpath.exists():
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            pages = json.load(f)

        current_category = default_category
        current_crop = None
        current_page = None
        buffer = []

        for p in pages:
            page_no = p["page"]
            text = clean_text(p["text"])
            lines = text.split("\n")

            for line in lines:
                line_strip = line.strip()
                if not line_strip:
                    continue

                cat = detect_category(line_strip)
                if cat:
                    current_category = cat
                    continue

                # In PAU, crop titles are often bold/prominent in first few lines of page
                crop = detect_crop(line_strip)
                if crop:
                    if current_crop and buffer:
                        records.append({
                            "season": season,
                            "source": source,
                            "state": "Punjab",
                            "category": current_category,
                            "crop": current_crop,
                            "page": current_page,
                            "content": " ".join(buffer),
                            "language": "en"
                        })
                        buffer = []
                    current_crop = crop
                    current_page = page_no
                    continue

                if current_crop:
                    buffer.append(line_strip)

        if current_crop and buffer:
            records.append({
                "season": season,
                "source": source,
                "state": "Punjab",
                "category": current_category,
                "crop": current_crop,
                "page": current_page,
                "content": " ".join(buffer),
                "language": "en"
            })

    return records


# ==============================================================================
# 3. PARSE TAMIL NADU (TNAU Agriculture & Horticulture CPG)
# ==============================================================================
def parse_tamilnadu():
    print("Parsing Tamil Nadu Agricultural University (TNAU) advisories...")
    tn_dir = EXTRACTED_DIR / "Tamilnadu"
    if not tn_dir.exists():
        return []

    files_config = [
        ("Agriculture-CPG-2020.json", "Annual", "Agriculture-CPG-2020.pdf", "Field Crops"),
        ("HORTICULTURE.json", "Annual", "HORTICULTURE.pdf", "Horticulture Crops")
    ]

    records = []

    for fname, season, source, default_category in files_config:
        fpath = tn_dir / fname
        if not fpath.exists():
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            pages = json.load(f)

        current_category = default_category
        current_crop = None
        current_page = None
        buffer = []

        for p in pages:
            page_no = p["page"]
            text = clean_text(p["text"])
            lines = text.split("\n")

            for line in lines:
                line_strip = line.strip()
                if not line_strip:
                    continue

                cat = detect_category(line_strip)
                if cat:
                    current_category = cat
                    continue

                crop = detect_crop(line_strip)
                if crop:
                    if current_crop and buffer:
                        records.append({
                            "season": season,
                            "source": source,
                            "state": "Tamil Nadu",
                            "category": current_category,
                            "crop": current_crop,
                            "page": current_page,
                            "content": " ".join(buffer),
                            "language": "en"
                        })
                        buffer = []
                    current_crop = crop
                    current_page = page_no
                    continue

                if current_crop:
                    buffer.append(line_strip)

        if current_crop and buffer:
            records.append({
                "season": season,
                "source": source,
                "state": "Tamil Nadu",
                "category": current_category,
                "crop": current_crop,
                "page": current_page,
                "content": " ".join(buffer),
                "language": "en"
            })

    return records


# ==============================================================================
# 4. PARSE GUJARAT (Directorate of Agriculture Crop Manuals)
# ==============================================================================
def parse_gujarat():
    print("Parsing Gujarat Directorate of Agriculture advisories...")
    guj_dir = EXTRACTED_DIR / "Gujarat"
    if not guj_dir.exists():
        return []

    files_config = [
        ("Cash_Crops.json", "Cash Crops", "Kharif"),
        ("Cereals.json", "Cereal Crops", "Kharif-Rabi"),
        ("Oilseeds.json", "Oilseed Crops", "Kharif-Rabi"),
        ("Pulses.json", "Pulse Crops", "Kharif-Rabi"),
        ("Summer_Crops.json", "Summer Crops", "Summer"),
        ("Schematic_Book_of_Agriculture.json", "Agricultural Schemes", "Annual")
    ]

    records = []

    for fname, category, default_season in files_config:
        fpath = guj_dir / fname
        if not fpath.exists():
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            pages = json.load(f)

        current_crop = None
        current_page = None
        buffer = []

        for p in pages:
            page_no = p["page"]
            text = clean_text(p["text"])
            lines = text.split("\n")

            for line in lines:
                line_strip = line.strip()
                if not line_strip:
                    continue

                # Check Gujarati crop names
                matched_crop = None
                for guj_term, canonical in GUJARATI_CROPS.items():
                    if guj_term in line_strip:
                        matched_crop = canonical
                        break

                if matched_crop:
                    if current_crop and buffer:
                        records.append({
                            "season": default_season,
                            "source": f"{fname.replace('.json', '.pdf')}",
                            "state": "Gujarat",
                            "category": category,
                            "crop": current_crop,
                            "page": current_page,
                            "content": " ".join(buffer),
                            "language": "gu"
                        })
                        buffer = []
                    current_crop = matched_crop
                    current_page = page_no
                    buffer.append(line_strip)
                    continue

                if current_crop:
                    buffer.append(line_strip)

        if current_crop and buffer:
            records.append({
                "season": default_season,
                "source": f"{fname.replace('.json', '.pdf')}",
                "state": "Gujarat",
                "category": category,
                "crop": current_crop,
                "page": current_page,
                "content": " ".join(buffer),
                "language": "gu"
            })

    return records


# ==============================================================================
# MAIN COMPILE
# ==============================================================================
def main():
    kharif_records = parse_icar_kharif()
    rabi_records = parse_icar_rabi()
    punjab_records = parse_punjab()
    tamilnadu_records = parse_tamilnadu()
    gujarat_records = parse_gujarat()

    all_records = (
        kharif_records + 
        rabi_records + 
        punjab_records + 
        tamilnadu_records + 
        gujarat_records
    )

    output_file = PARSED_DIR / "advisories.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print("\n==================================================")
    print("PARSING COMPLETE")
    print("==================================================")
    print(f"ICAR Kharif Records     : {len(kharif_records)}")
    print(f"ICAR Rabi Records       : {len(rabi_records)}")
    print(f"Punjab (PAU) Records    : {len(punjab_records)}")
    print(f"Tamil Nadu (TNAU) Recs  : {len(tamilnadu_records)}")
    print(f"Gujarat Records         : {len(gujarat_records)}")
    print(f"--------------------------------------------------")
    print(f"TOTAL UNIFIED RECORDS   : {len(all_records)}")
    print(f"Saved to                : {output_file.relative_to(BASE_DIR)}")
    print("==================================================")


if __name__ == "__main__":
    main()

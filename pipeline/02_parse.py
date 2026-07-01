import json
import re
from pathlib import Path

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

# Helper to normalize ligatures and spaces
def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\ufb01", "fi").replace("ϐ", "fi").replace("ϐield", "field")
    text = text.replace("\ufb02", "fl")
    text = text.replace("ﬀ", "ff").replace("ﬁ", "fi").replace("ﬂ", "fl")
    
    # Remove repeated headers/footers
    text = re.sub(r"(ICAR\s+KHARIF\s+AGRO-ADVISORY|ICAR\s+RABI\s+AgRo-AdvIsoRy\s+foR\s+fARmeRs|AgRo-AdvIsoRy\s+foR\s+fARmeRs)", "", text, flags=re.I)
    
    # Remove line numbers or solitary numbers (page numbers) at start/end of lines
    lines = []
    for line in text.split("\n"):
        line_strip = line.strip()
        # Skip solitary page numbers
        if line_strip.isdigit():
            continue
        # Skip empty lines
        if not line_strip:
            continue
        lines.append(line)
    
    return "\n".join(lines)

# Check if line matches a state
def detect_state(line: str) -> str:
    line_clean = line.strip().lower().replace("&", "and").replace(",", "")
    for state in CANONICAL_STATES:
        state_clean = state.lower().replace("&", "and").replace(",", "")
        # Match exact line or boundary
        if line_clean == state_clean or line_clean.startswith(state_clean + " "):
            return state
    return None

# Check if line matches a crop
def detect_crop(line: str) -> str:
    line_clean = line.strip().rstrip(":").strip().lower()
    
    # Handle direct exact matches
    for crop in CANONICAL_CROPS:
        if line_clean == crop.lower():
            return crop
            
    # Handle composite crop names, e.g. "Mash (Black Gram)" matching black gram
    for crop in CANONICAL_CROPS:
        crop_clean = crop.lower()
        if "(" in crop_clean:
            # Extract name and bracketed parts
            parts = re.findall(r'\b[a-z\s]+\b', crop_clean)
            for part in parts:
                part = part.strip()
                if len(part) > 3 and line_clean == part:
                    return crop
                    
    # Handle common sub-crop titles
    if line_clean == "paddy":
        return "Paddy"
    if line_clean == "moong" or line_clean == "moong (green gram)":
        return "Moong"
    if line_clean == "urad" or line_clean == "black gram":
        return "Black gram"
        
    return None

# Categories
CATEGORIES = [
    "CEREAL CROPS", "PULSES", "OILSEEDS", "FRUIT AND VEGETABLE CROPS", 
    "FRUIT & VEGETABLE CROPS", "VEGETABLE CROPS", "FRUIT CROPS", 
    "ROOT AND TUBER CROPS", "LIVESTOCK", "POULTRY", "FISHERIES", 
    "COMMERCIAL CROPS", "FODDER CROPS", "PULSE CROPS", "OILSEED CROPS"
]

def detect_category(line: str) -> str:
    line_clean = line.strip().upper()
    for cat in CATEGORIES:
        if line_clean == cat:
            return cat.title()
    return None

def parse_kharif():
    print("Parsing Kharif advisories...")
    input_file = EXTRACTED_DIR / "ICAR.json"
    if not input_file.exists():
        print(f"Extraction file {input_file} not found!")
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

            # Detect state change
            state = detect_state(line_strip)
            if state:
                # Save previous crop if exists
                if current_crop and buffer:
                    records.append({
                        "season": "Kharif",
                        "source": "ICAR.pdf",
                        "state": current_state,
                        "category": current_category,
                        "crop": current_crop,
                        "page": current_page,
                        "content": " ".join(buffer)
                    })
                    buffer = []
                current_state = state
                current_crop = None
                continue

            # Detect category change
            category = detect_category(line_strip)
            if category:
                current_category = category
                continue

            # Detect crop change
            crop = detect_crop(line_strip)
            if crop:
                if current_crop and buffer:
                    records.append({
                        "season": "Kharif",
                        "source": "ICAR.pdf",
                        "state": current_state,
                        "category": current_category,
                        "crop": current_crop,
                        "page": current_page,
                        "content": " ".join(buffer)
                    })
                    buffer = []
                current_crop = crop
                current_page = page_no
                continue

            # Accumulate content if inside a crop
            if current_crop:
                buffer.append(line_strip)

    # Save last crop
    if current_crop and buffer:
        records.append({
            "season": "Kharif",
            "source": "ICAR.pdf",
            "state": current_state,
            "category": current_category,
            "crop": current_crop,
            "page": current_page,
            "content": " ".join(buffer)
        })

    return records

def parse_rabi():
    print("Parsing Rabi advisories...")
    input_file = EXTRACTED_DIR / "Rabi-Agro-Advisory-2021-22.json"
    if not input_file.exists():
        print(f"Extraction file {input_file} not found!")
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

            # Detect zone pattern, e.g. "Zone-III" or "Zone III"
            zone_match = re.search(r'\b(Zone-\w+|\bZone\s+\w+)\b', line_strip, re.I)
            if zone_match:
                current_zone = zone_match.group(1).title()

            # Detect state mention
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
                        "zone": current_zone
                    })
                    buffer = []
                current_state = state
                current_crop = None
                continue

            # Detect crop change
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
                        "zone": current_zone
                    })
                    buffer = []
                current_crop = crop
                current_page = page_no
                continue

            # Accumulate content
            if current_crop:
                buffer.append(line_strip)

    # Save last crop
    if current_crop and buffer:
        records.append({
            "season": "Rabi",
            "source": "Rabi-Agro-Advisory-2021-22.pdf",
            "state": current_state,
            "category": "Rabi Crops",
            "crop": current_crop,
            "page": current_page,
            "content": " ".join(buffer),
            "zone": current_zone
        })

    return records

def main():
    kharif_records = parse_kharif()
    rabi_records = parse_rabi()
    
    all_records = kharif_records + rabi_records
    
    output_file = PARSED_DIR / "advisories.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully compiled {len(all_records)} raw records.")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()

import json
import hashlib
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PARSED_DIR = BASE_DIR / "data" / "parsed"
QUARANTINE_DIR = BASE_DIR / "data" / "quarantine"
CONFIG_DIR = BASE_DIR / "config"

QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

# Load Canonical Lists
with open(CONFIG_DIR / "crops.json", "r", encoding="utf-8") as f:
    CANONICAL_CROPS = set(crop.lower() for crop in json.load(f))

with open(CONFIG_DIR / "states.json", "r", encoding="utf-8") as f:
    CANONICAL_STATES = set(state.lower() for state in json.load(f))

def main():
    input_file = PARSED_DIR / "advisories.json"
    if not input_file.exists():
        print(f"Parsed advisories file {input_file} not found!")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    valid_records = []
    quarantined_records = []
    seen_hashes = set()
    
    for r in records:
        state = r.get("state")
        crop = r.get("crop")
        season = r.get("season")
        content = r.get("content", "").strip()
        
        errors = []
        
        if not state:
            errors.append("Missing state")
        elif state.lower() not in CANONICAL_STATES:
            errors.append(f"State '{state}' is not in canonical list")
            
        if not crop:
            errors.append("Missing crop")
        elif crop.lower() not in CANONICAL_CROPS:
            errors.append(f"Crop '{crop}' is not in canonical list")
            
        if not season:
            errors.append("Missing season")
            
        if not content:
            errors.append("Empty content")
        elif len(content) < 50:
            errors.append(f"Content too short ({len(content)} chars)")
            
        # Deduplication
        if not errors:
            # Generate SHA-256 hash
            content_hash = hashlib.sha256(
                f"{state.lower()}|{crop.lower()}|{season.lower()}|{content}".encode("utf-8")
            ).hexdigest()
            
            if content_hash in seen_hashes:
                errors.append("Duplicate advisory record")
            else:
                seen_hashes.add(content_hash)
                
        if errors:
            r_failed = r.copy()
            r_failed["validation_errors"] = errors
            quarantined_records.append(r_failed)
        else:
            valid_records.append(r)
            
    # Save valid records
    valid_output = PARSED_DIR / "valid_advisories.json"
    with open(valid_output, "w", encoding="utf-8") as f:
        json.dump(valid_records, f, indent=2, ensure_ascii=False)
        
    # Save quarantined records
    quarantine_output = QUARANTINE_DIR / "failed_advisories.json"
    with open(quarantine_output, "w", encoding="utf-8") as f:
        json.dump(quarantined_records, f, indent=2, ensure_ascii=False)
        
    print(f"Validation summary:")
    print(f"  Valid records: {len(valid_records)}")
    print(f"  Quarantined records: {len(quarantined_records)}")
    print(f"  Outputs saved to: {valid_output} and {quarantine_output}")

if __name__ == "__main__":
    main()

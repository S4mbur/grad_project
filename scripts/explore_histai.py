#!/usr/bin/env python3
"""
Download and analyze HISTAI metadata for skin cancer WSI selection.
Filters cases by diagnosis/ICD-10 for our 5-superclass project.
"""
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).parent.parent
HISTAI_DIR = PROJECT_ROOT / "data" / "histai_metadata"
HISTAI_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Download metadata
print("=" * 60)
print("[1/3] Downloading HISTAI metadata (59.6 MB)...")
print("=" * 60)

meta_path = HISTAI_DIR / "metadata.json"
if not meta_path.exists():
    local = hf_hub_download(
        repo_id="histai/HISTAI-metadata",
        filename="metadata.json",
        repo_type="dataset",
        local_dir=str(HISTAI_DIR),
    )
    print(f"  Downloaded: {local}")
else:
    print(f"  Already exists: {meta_path}")

# Step 2: Parse and analyze
print("\n" + "=" * 60)
print("[2/3] Analyzing metadata...")
print("=" * 60)

with open(meta_path, "r") as f:
    metadata = json.load(f)

print(f"  Total entries: {len(metadata)}")
print(f"  Type: {type(metadata)}")

# Explore structure
if isinstance(metadata, dict):
    keys = list(metadata.keys())
    print(f"  Top keys ({len(keys)}): {keys[:5]}...")
    
    # Check first entry
    first_key = keys[0]
    first_val = metadata[first_key]
    print(f"\n  Sample entry [{first_key}]:")
    if isinstance(first_val, dict):
        for k, v in first_val.items():
            val_str = str(v)[:100]
            print(f"    {k}: {val_str}")
    else:
        print(f"    {str(first_val)[:200]}")

elif isinstance(metadata, list):
    print(f"  First entry: {json.dumps(metadata[0], indent=2, default=str)[:500]}")

# Step 3: Filter skin cases
print("\n" + "=" * 60)
print("[3/3] Filtering skin-related cases...")
print("=" * 60)

# ICD-10 codes for skin
SKIN_ICD10 = {
    "L57": "Actinic Keratosis",
    "D04": "Bowen / Carcinoma in situ",
    "C43": "Melanoma",
    "C44": "BCC / SCC (non-melanoma skin cancer)",
    "D22": "Melanocytic Nevus",
    "D23": "Benign skin neoplasm",
    "L82": "Seborrheic Keratosis",
    "L85": "Epidermal thickening",
    "D17": "Lipoma",
    "D18": "Hemangioma",
    "L72": "Follicular cyst",
    "L98": "Other skin disorders",
}

icd10_counts = {}
diagnosis_counts = {}
skin_cases = {}
dataset_counts = {}

entries = metadata.values() if isinstance(metadata, dict) else metadata
entry_keys = list(metadata.keys()) if isinstance(metadata, dict) else range(len(metadata))

for key, entry in zip(entry_keys, entries):
    if not isinstance(entry, dict):
        continue
    
    icd10 = str(entry.get("icd10", "")).strip()
    diagnosis = str(entry.get("diagnosis", "")).strip().lower()
    case_mapping = entry.get("case_mapping", {})
    
    # Count ICD-10
    icd_prefix = icd10[:3] if icd10 else "N/A"
    icd10_counts[icd_prefix] = icd10_counts.get(icd_prefix, 0) + 1
    
    # Check if skin-related
    is_skin = False
    for prefix in SKIN_ICD10:
        if icd10.upper().startswith(prefix):
            is_skin = True
            break
    
    if not is_skin:
        skin_keywords = [
            "melanom", "nevus", "nev", "basal cell", "squamous",
            "keratosis", "actinic", "bowen", "skin", "derm",
            "epiderm", "cutane", "bcc", "scc", "iec"
        ]
        for kw in skin_keywords:
            if kw in diagnosis:
                is_skin = True
                break
    
    if is_skin:
        skin_cases[key] = entry
        
        # Track dataset
        if isinstance(case_mapping, dict):
            for ds_name in case_mapping.values():
                if isinstance(ds_name, str):
                    dataset_counts[ds_name] = dataset_counts.get(ds_name, 0) + 1

# Print ICD-10 summary (only skin-related)
print("\n  Skin-related ICD-10 codes found:")
for prefix, label in sorted(SKIN_ICD10.items()):
    count = icd10_counts.get(prefix, 0)
    if count > 0:
        print(f"    {prefix} ({label}): {count} cases")

print(f"\n  Total skin cases: {len(skin_cases)}")

# Categorize into 5 superclasses
superclasses = {
    "0_melanocytic_benign": [],
    "1_melanocytic_malignant": [],
    "2_nonmelano_benign": [],
    "3_nonmelano_indeterminate": [],
    "4_nonmelano_malignant": [],
}

for key, entry in skin_cases.items():
    icd10 = str(entry.get("icd10", "")).upper()
    diagnosis = str(entry.get("diagnosis", "")).lower()
    
    if icd10.startswith("D22") or "nevus" in diagnosis or "nev" in diagnosis:
        superclasses["0_melanocytic_benign"].append(key)
    elif icd10.startswith("C43") or "melanoma" in diagnosis:
        superclasses["1_melanocytic_malignant"].append(key)
    elif icd10.startswith(("L82", "D23", "D17", "D18", "L72")):
        superclasses["2_nonmelano_benign"].append(key)
    elif icd10.startswith(("L57", "D04")) or "actinic" in diagnosis or "bowen" in diagnosis:
        superclasses["3_nonmelano_indeterminate"].append(key)
    elif icd10.startswith("C44") or "basal cell" in diagnosis or "squamous" in diagnosis:
        superclasses["4_nonmelano_malignant"].append(key)
    else:
        # Try to classify by keywords
        if any(w in diagnosis for w in ["benign", "cyst", "fibroma"]):
            superclasses["2_nonmelano_benign"].append(key)
        elif any(w in diagnosis for w in ["carcinoma", "malignant"]):
            superclasses["4_nonmelano_malignant"].append(key)

print("\n  5-Superclass distribution:")
for sc, cases in superclasses.items():
    print(f"    {sc}: {len(cases)} cases")

# Show dataset distribution
if dataset_counts:
    print("\n  Dataset distribution:")
    for ds, count in sorted(dataset_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {ds}: {count}")

# Save filtered results
output = {
    "summary": {sc: len(cases) for sc, cases in superclasses.items()},
    "superclasses": {sc: cases[:50] for sc, cases in superclasses.items()},
    "total_skin_cases": len(skin_cases),
}
output_path = HISTAI_DIR / "skin_analysis.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Analysis saved: {output_path}")

# Show sample cases for class 3 (Non-melanocytic Indeterminate = AK)
ak_cases = superclasses["3_nonmelano_indeterminate"]
if ak_cases:
    print(f"\n  Sample AK/Indeterminate cases ({len(ak_cases)} total):")
    for cid in ak_cases[:5]:
        entry = skin_cases[cid]
        print(f"    Case: {cid}")
        print(f"      ICD-10: {entry.get('icd10', 'N/A')}")
        print(f"      Diagnosis: {str(entry.get('diagnosis', 'N/A'))[:80]}")
        cm = entry.get("case_mapping", {})
        if cm:
            print(f"      Dataset: {cm}")
        print()

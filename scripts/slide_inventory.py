#!/usr/bin/env python3
"""Complete slide inventory across all datasets."""

print("=" * 60)
print("COMPLETE SLIDE INVENTORY")
print("=" * 60)

# 1. COBRA BCC Group (WSL)
cobra_normal = 374
cobra_bcc = 201
print(f"\n[COBRA BCC Group - WSL]")
print(f"  Normal:  {cobra_normal}")
print(f"  BCC:     {cobra_bcc}")

# 2. COBRA OOD Group (D:)
ood = {
    "SCC": 497,
    "BCC": 170,
    "Lymphoma": 167,
    "Melanoma": 107,
    "Cutaneous metastases": 77,
    "Benign": 71,
    "Benign sebaceous": 57,
    "Melanoma in situ": 38,
    "Merkel cell": 30,
    "No abnormalities": 14,
    "Sebaceous carcinoma": 12,
    "Cylindroma": 11,
    "Microcystic adnexal": 9,
    "Skin adnexal carcinoma": 8,
}
print(f"\n[COBRA OOD Group - D:]")
for k, v in ood.items():
    print(f"  {k:25s}: {v}")

# 3. TCGA-SKCM
tcga_wsl = 37
tcga_d = 28
print(f"\n[TCGA-SKCM - Melanoma]")
print(f"  WSL (existing):  {tcga_wsl}")
print(f"  D: (new):        {tcga_d}")

# === 5 SUPERCLASS MAPPING ===
print(f"\n{'=' * 60}")
print("5-SUPERCLASS MAPPING")
print("=" * 60)

mel_benign = 0
mel_malign = 107 + 38 + 37 + 28  # Melanoma + in situ + TCGA
nonmel_benign = 374 + 71 + 57 + 14 + 11  # Normal, Benign, seb.benign, no abnorm, cylindroma
nonmel_indet = 0
nonmel_malign = 201 + 497 + 170 + 30 + 12 + 9 + 8  # BCC, SCC, Merkel, seb.ca, etc.

print(f"\n  1. Melanocytic Benign (Nevus):          {mel_benign:5d}  <-- NO DATA")
print(f"  2. Melanocytic Malignant (Melanoma):    {mel_malign:5d}")
print(f"     - COBRA OOD Melanoma:     107")
print(f"     - COBRA OOD Mel in situ:   38")
print(f"     - TCGA-SKCM (WSL):         37")
print(f"     - TCGA-SKCM (D:):          28")
print(f"  3. Non-mel Benign (Normal):            {nonmel_benign:5d}")
print(f"     - COBRA Normal:           374")
print(f"     - COBRA OOD Benign:        71")
print(f"     - Benign sebaceous:        57")
print(f"     - No abnormalities:        14")
print(f"     - Cylindroma:              11")
print(f"  4. Non-mel Indeterminate (AK):         {nonmel_indet:5d}  <-- NO DATA")
print(f"  5. Non-mel Malignant (BCC/SCC):        {nonmel_malign:5d}")
print(f"     - COBRA BCC:              201")
print(f"     - COBRA OOD BCC:          170")
print(f"     - COBRA OOD SCC:          497")
print(f"     - Merkel cell:             30")
print(f"     - Other malignant:         29")

# Balance analysis
print(f"\n{'=' * 60}")
print("BALANCE ANALYSIS (3 active classes)")
print("=" * 60)

classes = {
    "Mel. Malignant (Melanoma)": mel_malign,
    "Non-mel Benign (Normal)": nonmel_benign,
    "Non-mel Malignant (BCC/SCC)": nonmel_malign,
}

max_val = max(classes.values())
for k, v in sorted(classes.items(), key=lambda x: -x[1]):
    bar = "#" * int(v / max_val * 40)
    print(f"  {k:30s}: {v:5d}  {bar}")

gap_to_benign = nonmel_benign - mel_malign
gap_to_malign = nonmel_malign - mel_malign

print(f"\n  Melanoma = {mel_malign} (smallest class)")
print(f"  Gap to Non-mel Benign ({nonmel_benign}):    need {gap_to_benign} more")
print(f"  Gap to Non-mel Malignant ({nonmel_malign}): need {gap_to_malign} more")
print(f"\n  TCGA-SKCM has {305} primary (01Z) melanoma slides available.")
print(f"  Already downloaded (01Z): 17")
print(f"\n  OPTIONS:")
print(f"  A) Add ~320 more TCGA → total ~530 mel → close to benign ({nonmel_benign})")
print(f"  B) Add ~100 more TCGA → total ~310 mel → reasonable balance")
print(f"  C) Keep current {mel_malign} → cap other classes to ~210 for training")

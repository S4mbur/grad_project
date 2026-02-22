# WSI Data Sources Reference
# ===========================
# Last updated: 2026-02-14

## Currently Available Data
- COBRA (normal + BCC): Already have tiles
- TCGA-SKCM (melanoma): 37 slides downloaded via GDC API
- Histo-Seg (BCC + SCC + IEC): Downloaded from Mendeley, exploring now

## High Priority - Ready to Download
1. CPTAC-CM (Cutaneous Melanoma)
   - URL: https://www.cancerimagingarchive.net/collection/cptac-cm/
   - Content: Cutaneous melanoma pathology WSI + radiology
   - Size: 107 GB total (pathology subset smaller)
   - Access: OPEN (CC BY 3.0)
   - Download: https://www.cancerimagingarchive.net/wp-content/uploads/TCIA-CPTAC-CM_v11_20240429.tcia
   - Pathology search: https://pathdb.cancerimagingarchive.net/eaglescope/dist/...

2. TCGA-HNSC (Head-Neck SCC via GDC)
   - URL: https://www.cancerimagingarchive.net/collection/tcga-hnsc/
   - Content: 472 diagnostic SVS slides (Head-Neck Squamous Cell Carcinoma)
   - Access: OPEN via GDC API (same as TCGA-SKCM download method)
   - Use case: SCC class - similar morphology to cutaneous SCC
   - Note: Mukozal epitel kaynaklı, stroma farklı olabilir. HPV+ olanlar farklı morfoloji.

3. CPTAC-HNSCC (Head-Neck SCC via CPTAC)
   - URL: https://www.cancerimagingarchive.net/collection/cptac-hnscc/
   - Content: Head-Neck SCC pathology WSI
   - Size: 97 GB
   - Access: OPEN (CC BY 3.0) - pathology portal accessible
   - Download: Aspera or pathology portal

## Medium Priority - Other SCC Types (cross-organ transfer)
4. TCGA-LUSC (Lung SCC via GDC)
   - Diagnostic slides available, open access
   - Lung squamous cell carcinoma

5. TCGA-CESC (Cervical SCC via GDC)
   - Diagnostic slides available, open access

6. TCGA-ESCA (Esophageal SCC via GDC)
   - Diagnostic slides available, open access

## Pending Access
7. HISTAI-skin-b1 / HISTAI-skin-b2 (Hugging Face)
   - URL: https://huggingface.co/datasets/histai/HISTAI-skin-b1
   - Content: 7,710 / 43,757 WSI with metadata (diagnosis, ICD-10)
   - Contains: AK, Melanoma, BCC, SCC, Nevus, Seb Keratosis
   - Access: PENDING REVIEW (requested 2026-02-14)
   - HF token: saved to WSL

## Access Blocked (NIH Policy Change)
- Anti-PD-1_MELANOMA: Radiology only + access blocked
- CMB-MEL: Has histopathology but access blocked
- HNSCC (MDACC): CT radiology only + access blocked
- CPTAC-LSCC: Lung SCC but access blocked

## Not Useful
- ~130+ other TCIA collections (brain, breast, prostate, colon, kidney, liver, etc.)
- Kaggle skin datasets: Dermoscopic images, not histopathology WSI
- UQ NMSC: Access restricted to UQ accounts

## Head-Neck SCC Usability Notes
- SCC morphology is similar across organs (confirmed by literature)
- Tumor/stroma ratio, immune infiltration, tumor budding show no significant differences
- Deep learning models trained on HNSCC transfer to cutaneous SCC
- CAUTION: Surrounding stroma is different (mucosal vs keratinized epithelium)
- CAUTION: HPV+ HNSCC has different morphology (basaloid, poorly differentiated)
- RECOMMENDATION: Use H&N SCC as supplementary data for SCC class, not sole source

#!/usr/bin/env python3

import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / 'results'
OUT_DIR = RESULTS_DIR / 'phase1_hard_case_bank'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_OUT = OUT_DIR / 'all_test_predictions.csv'
HARD_OUT = OUT_DIR / 'hard_case_bank.csv'
SUMMARY_OUT = OUT_DIR / 'summary.txt'

run_dirs = sorted([p for p in RESULTS_DIR.glob('mil_4class_*_v3*') if p.is_dir()])
all_rows = []
hard_rows = []

for run_dir in run_dirs:
    pred_csv = run_dir / 'phase1_test_predictions.csv'
    hard_csv = run_dir / 'phase1_hard_cases.csv'

    if pred_csv.exists():
        with open(pred_csv, newline='') as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row['result_dir'] = str(run_dir)
                all_rows.append(row)

    if hard_csv.exists():
        with open(hard_csv, newline='') as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row['result_dir'] = str(run_dir)
                hard_rows.append(row)

# Deduplicate by result_dir + slide_id
seen = set()
dedup_all = []
for row in all_rows:
    key = (row.get('result_dir', ''), row.get('slide_id', ''))
    if key in seen:
        continue
    seen.add(key)
    dedup_all.append(row)

seen = set()
dedup_hard = []
for row in hard_rows:
    key = (row.get('result_dir', ''), row.get('slide_id', ''))
    if key in seen:
        continue
    seen.add(key)
    dedup_hard.append(row)

if dedup_all:
    with open(TEST_OUT, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dedup_all[0].keys()))
        writer.writeheader()
        writer.writerows(dedup_all)

if dedup_hard:
    with open(HARD_OUT, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dedup_hard[0].keys()))
        writer.writeheader()
        writer.writerows(dedup_hard)

with open(SUMMARY_OUT, 'w') as f:
    f.write('Phase 1 Hard Case Bank Summary\n')
    f.write('=' * 40 + '\n')
    f.write(f'Runs scanned: {len(run_dirs)}\n')
    f.write(f'Test prediction rows: {len(dedup_all)}\n')
    f.write(f'Hard-case rows: {len(dedup_hard)}\n')
    f.write(f'All predictions CSV: {TEST_OUT if dedup_all else "not generated"}\n')
    f.write(f'Hard-case bank CSV: {HARD_OUT if dedup_hard else "not generated"}\n')

print(f'Runs scanned: {len(run_dirs)}')
print(f'Test prediction rows: {len(dedup_all)}')
print(f'Hard-case rows: {len(dedup_hard)}')
print(f'Summary: {SUMMARY_OUT}')

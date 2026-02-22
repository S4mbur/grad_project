#!/usr/bin/env python3
"""Check multi-class tile manifest statistics."""
import pandas as pd

df = pd.read_csv('data/manifests/multiclass_tile_manifest.csv')

print('=== MULTI-CLASS TILE MANIFEST SUMMARY ===')
print(f'Total tiles: {len(df):,}')
print()

print('Class distribution:')
for label in sorted(df['label'].unique()):
    name = df[df['label'] == label]['label_name'].iloc[0]
    count = len(df[df['label'] == label])
    pct = count / len(df) * 100
    print(f'  {label} = {name}: {count:,} tiles ({pct:.1f}%)')

print()
print('Split distribution:')
for split in ['train', 'val', 'test']:
    if split in df['split'].values:
        count = len(df[df['split'] == split])
        print(f'  {split}: {count:,} tiles')

print()
print('Source distribution:')
for source in df['source'].unique():
    count = len(df[df['source'] == source])
    print(f'  {source}: {count:,} tiles')

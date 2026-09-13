from pathlib import Path
from src.fare_pipeline import read_ids_entries, merge_ids_entries

entries = read_ids_entries([Path('dataset/ids_text/ids.txt'), Path('dataset/ids_text/ids-cdp.txt')])
print('entries', len(entries))
merged = merge_ids_entries(entries)
print('merged', len(merged))
print('sample keys', list(merged.keys())[:50])

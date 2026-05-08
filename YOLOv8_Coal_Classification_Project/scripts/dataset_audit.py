import csv
from pathlib import Path

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
OUT1 = ROOT / 'data' / 'dataset_counts.csv'
OUT2 = ROOT / 'reports' / 'dataset_counts.csv'

rows = []

for split in ['train', 'val']:
    split_dir = DATA_DIR / split
    if not split_dir.exists():
        print(f"Warning: {split_dir} does not exist")
        continue
    for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        cnt = 0
        for ext in VALID_EXTS:
            cnt += sum(1 for _ in class_dir.glob(f'*{ext}'))
        rows.append({'split': split, 'class': class_dir.name, 'count': cnt})

# write CSVs
headers = ['split', 'class', 'count']
for out in (OUT1, OUT2):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

# print a short summary
print('Dataset counts:')
for r in rows:
    print(f"{r['split']}/{r['class']}: {r['count']}")

print(f"\nWrote: {OUT1}\n       {OUT2}")

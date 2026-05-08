from pathlib import Path

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
ROOT = Path(__file__).resolve().parents[1]
BAL = ROOT / 'data' / 'train_balanced'

if not BAL.exists():
    print(f"No balanced train dir found at {BAL}")
else:
    for class_dir in sorted([p for p in BAL.iterdir() if p.is_dir()]):
        cnt = 0
        for ext in VALID_EXTS:
            cnt += len(list(class_dir.glob(f'*{ext}')))
        print(f"{class_dir.name}: {cnt}")

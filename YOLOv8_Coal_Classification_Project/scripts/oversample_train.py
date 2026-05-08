"""
Simple oversampling script: duplicates images from minority classes into a new folder
`data/train_balanced/` until each class reaches `target_count`.

Usage:
    python scripts/oversample_train.py --target 200

This script does not modify the original data. It writes copies into data/train_balanced.
"""

import argparse
import random
import shutil
from pathlib import Path

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / 'data' / 'train'
BALANCED_DIR = ROOT / 'data' / 'train_balanced'


def collect_files(class_dir: Path):
    files = []
    for ext in VALID_EXTS:
        files.extend(list(class_dir.glob(f'*{ext}')))
    return files


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def oversample(target_count: int = 200, random_seed: int = 42):
    random.seed(random_seed)
    if not TRAIN_DIR.exists():
        print(f"Train directory not found: {TRAIN_DIR}")
        return

    ensure_dir(BALANCED_DIR)

    class_dirs = [p for p in TRAIN_DIR.iterdir() if p.is_dir()]
    if not class_dirs:
        print(f"No class directories found in {TRAIN_DIR}")
        return

    summary = []

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        orig_files = collect_files(class_dir)
        orig_count = len(orig_files)

        dest_class_dir = BALANCED_DIR / class_name
        ensure_dir(dest_class_dir)

        # Copy originals
        for f in orig_files:
            shutil.copy2(f, dest_class_dir / f.name)

        # If no originals, skip
        if orig_count == 0:
            print(f"Warning: class {class_name} has 0 original images; skipping augmentation")
            summary.append((class_name, 0, 0, 0))
            continue

        # Duplicate random originals until reach target_count
        copies = 0
        current = orig_count
        dup_idx = 0
        while current < target_count:
            src = random.choice(orig_files)
            dup_idx += 1
            new_name = f"{src.stem}_dup{dup_idx}{src.suffix}"
            dest_path = dest_class_dir / new_name
            shutil.copy2(src, dest_path)
            copies += 1
            current += 1

        summary.append((class_name, orig_count, copies, current))

    # Print summary
    print("Oversampling summary (class, original, copies added, final):")
    for cls, orig, copies, final in summary:
        print(f"  {cls}: {orig} -> +{copies} = {final}")

    print(f"Balanced training data written to: {BALANCED_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=200, help='Target images per class')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    oversample(target_count=args.target, random_seed=args.seed)

#!/usr/bin/env python3
"""
Create a compact share ZIP (<2GB) containing:
- reports/
- coal_classification_runs/coal_yolov8_m_balanced2/
- scripts/
- model/best.pt
- data/classes.txt
- sample_images/ (up to N per class, prefer val then train_balanced)
Usage: python scripts/create_small_share_package.py --per_class 20
"""
from pathlib import Path
import argparse
import shutil
import zipfile

root = Path('.')
exp = 'coal_yolov8_m_balanced2'
reports = root / 'reports'
exp_dir = root / 'coal_classification_runs' / exp
model_file = root / 'model' / 'best.pt'
scripts_dir = root / 'scripts'

def collect_samples(per_class=20):
    val_dir = root / 'data' / 'val'
    train_dir = root / 'data' / 'train_balanced'
    exts = ['*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff']
    samples = []
    if val_dir.exists():
        for class_dir in sorted([d for d in val_dir.iterdir() if d.is_dir()]):
            imgs = []
            for e in exts:
                imgs.extend(list(class_dir.glob(e)))
            imgs = imgs[:per_class]
            samples.extend(imgs)
    # if not enough samples, pull from train_balanced
    if train_dir.exists():
        # count per class currently collected
        from collections import defaultdict
        counts = defaultdict(int)
        for p in samples:
            # infer class from parent name
            counts[p.parent.name]+=1
        for class_dir in sorted([d for d in train_dir.iterdir() if d.is_dir()]):
            needed = per_class - counts.get(class_dir.name, 0)
            if needed<=0:
                continue
            imgs = []
            for e in exts:
                imgs.extend(list(class_dir.glob(e)))
            # skip those already in samples
            to_add = [i for i in imgs if i not in samples][:needed]
            samples.extend(to_add)
    return samples


def create_zip(per_class=20):
    samples = collect_samples(per_class)
    out_zip = root / f"{exp}_small_package.zip"
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        # add reports
        if reports.exists():
            for p in reports.rglob('*'):
                z.write(p, Path('reports')/p.relative_to(reports))
        # add experiment folder
        if exp_dir.exists():
            for p in exp_dir.rglob('*'):
                # skip large files if any beyond weights we want
                z.write(p, Path('experiment')/p.relative_to(exp_dir))
        # add scripts
        if scripts_dir.exists():
            for p in scripts_dir.rglob('*.py'):
                z.write(p, Path('scripts')/p.relative_to(scripts_dir))
        # add model
        if model_file.exists():
            z.write(model_file, Path('model')/model_file.name)
        # add classes.txt
        cfile = root / 'data' / 'classes.txt'
        if cfile.exists():
            z.write(cfile, Path('data')/cfile.name)
        # add samples
        sample_base = Path('sample_images')
        for p in samples:
            arcname = sample_base / p.parent.name / p.name
            z.write(p, arcname)
    print('Created small package:', out_zip)
    return out_zip

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--per_class', type=int, default=20, help='Max sample images per class')
    args = parser.parse_args()
    create_zip(args.per_class)

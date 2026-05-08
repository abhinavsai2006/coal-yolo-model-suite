#!/usr/bin/env python3
"""
Assemble a small share package with dataset counts, model results, and sample images for a paper.
Produces: coal_yolov8_m_balanced2_share_package.zip at repo root.
"""
from pathlib import Path
import shutil
import zipfile
import csv

root = Path('.')
share_dir = root / 'share_package_coal_yolov8_m_balanced2'
if share_dir.exists():
    shutil.rmtree(share_dir)
share_dir.mkdir()

# Files to include (if they exist)
reports = root / 'reports'
exp = 'coal_yolov8_m_balanced2'
exp_dir = root / 'coal_classification_runs' / exp

candidates = []
# dataset counts
for p in [reports / 'dataset_image_counts.csv', reports / 'dataset_image_counts.txt']:
    if p.exists():
        candidates.append(p)
# model results
for p in [reports / f'{exp}_metrics.csv', reports / f'{exp}_metrics_summary.csv', reports / f'{exp}_report.pdf', reports / 'coal_classification_metrics.csv', reports / 'coal_classification_metrics_summary.csv', reports / 'confusion_matrix_advanced.png']:
    if p.exists():
        candidates.append(p)
# training logs and args
for p in [exp_dir / 'results.csv', exp_dir / 'args.yaml']:
    if p.exists():
        candidates.append(p)

# copy candidates into share_dir/docs
docs_dir = share_dir / 'docs'
docs_dir.mkdir()
for p in candidates:
    shutil.copy2(p, docs_dir / p.name)

# collect sample images: pick one example per class from data/val
val_dir = root / 'data' / 'val'
sample_dir = share_dir / 'sample_images'
sample_dir.mkdir()
img_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
if val_dir.exists():
    for class_dir in sorted([d for d in val_dir.iterdir() if d.is_dir()]):
        found = None
        for ext in img_exts:
            files = list(class_dir.glob(ext))
            if files:
                found = files[0]
                break
        if found:
            # copy and prefix with class name to avoid collisions
            dest = sample_dir / f"{class_dir.name}_{found.name}"
            shutil.copy2(found, dest)

# include small README
readme = share_dir / 'README_share.txt'
with open(readme, 'w') as f:
    f.write('Share package for experiment: {}\n'.format(exp))
    f.write('\nIncluded files:\n')
    for p in (docs_dir).iterdir():
        f.write(f"- docs/{p.name}\n")
    f.write('\nSample images (one per validation class) in sample_images/\n')
    # include counts summary if available
    counts = reports / 'dataset_image_counts.csv'
    if counts.exists():
        f.write('\nDataset counts:\n')
        with open(counts,'r', newline='') as cf:
            reader = csv.reader(cf)
            for row in reader:
                f.write(','.join(row) + '\n')

# create zip
zip_name = root / f"{exp}_share_package.zip"
if zip_name.exists():
    zip_name.unlink()

with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    # add docs
    for p in docs_dir.rglob('*'):
        z.write(p, p.relative_to(share_dir))
    # add sample images
    for p in sample_dir.rglob('*'):
        z.write(p, p.relative_to(share_dir))
    # add README
    z.write(readme, readme.relative_to(share_dir))

print('Created share ZIP:', zip_name)

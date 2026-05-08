# Coal YOLO Model Suite

Professional, code-only monorepo for coal-image classification experiments across multiple YOLO generations.

This repository is intentionally curated to keep only source code, configuration files, and lightweight documentation. Large datasets, image assets, trained weights, generated reports, and other heavy artifacts are excluded.

## Repository Scope

Included projects:
- `YOLOv8_Coal_Classification_Project`
- `YOLOv10_Coal_Classification_Project`
- `YOLOv11_Coal_Classification_Project`
- `Yolov12`

Excluded artifacts:
- raw and split image datasets
- screenshots and sample image dumps
- model checkpoints (`*.pt`, `*.pth`, `*.onnx`, `*.engine`)
- generated charts/reports (`*.png`, `*.pdf`, `*.csv`, `*.json`)
- run outputs and cache folders (`runs/`, `__pycache__/`, etc.)

## Dataset Source

Primary dataset reference:
- https://www.scidb.cn/en/detail?dataSetId=5654e40ae6d14ade84bac79cb0753852&version=V1

Archive groups referenced from the source page:
- `01.Non-destructive coal.zip`
- `02.Destructive coal.zip`
- `03.Strongly destructive coal.zip`
- `04.Pulverized coal.zip`
- `05.Fully pulverized coal.zip`

For reproducibility notes, see `DATASET.md`.

## Project Breakdown

### 1) YOLOv8_Coal_Classification_Project
Purpose:
- Baseline and extended YOLOv8 classification workflows with data prep, training, inference, and reporting scripts.

Key paths:
- `scripts/` for train/evaluation/report pipelines
- `data/` for lightweight dataset metadata (`classes.txt`, YAML)
- `requirements.txt` for dependencies

### 2) YOLOv10_Coal_Classification_Project
Purpose:
- YOLOv10-based coal classification pipeline focused on diversified training and production inference utilities.

Key paths:
- `train_diversified_model.py`
- `comprehensive_evaluation.py`
- `production_classifier.py`
- `coal_classification.yaml`

### 3) YOLOv11_Coal_Classification_Project
Purpose:
- YOLOv11 workflows including standard training/evaluation and advanced hybrid variants.

Key paths:
- `train_model.py`, `train_model_regularized.py`
- `evaluate_model.py`, `evaluate_all_metrics.py`
- `yolo11_a2mada/` for A2MADA attention-alignment variant
- `yolo11_hybridmodel/` for hybrid architecture, training, inference, and utilities

### 4) Yolov12
Purpose:
- Experimental YOLOv12-style model implementation and training/inference scripts.

Key paths:
- `yolov12_model.py`
- `train_yolov12.py`
- `infer_yolov12.py`
- `organize_yolo_dataset.py`

## Quick Start

1. Clone repository
```bash
git clone https://github.com/abhinavsai2006/coal-yolo-model-suite.git
cd coal-yolo-model-suite
```

2. Choose a project folder and create an environment
```bash
cd YOLOv11_Coal_Classification_Project
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Run the selected workflow (example)
```bash
python train_model.py
python evaluate_model.py
```

## Reproducibility and Data Policy

- This repository does not store dataset images.
- Download dataset archives from SciDB using the URL above.
- Keep extracted datasets outside this repository or in ignored local paths.
- If required, update local dataset paths in each project's YAML/config files.

## Recommended Repository Hygiene

- Keep large outputs local and untracked.
- Commit only source, configs, and essential docs.
- Pin dependency versions in each project's `requirements.txt` when preparing a release.

## License and Attribution

- Project-level licenses and notices are retained where available (for example in `YOLOv8_Coal_Classification_Project/LICENSE`).
- Respect original dataset terms of use from SciDB.

---

Maintained for clear, reproducible, and professional coal-classification model development.
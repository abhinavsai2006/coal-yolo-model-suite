# Coal YOLO Model Suite

Industry-grade, code-only monorepo for coal image classification research and production-oriented workflows across YOLOv8, YOLOv10, YOLOv11, and a YOLOv12-style experimental implementation.

This repository is structured for reproducibility, portability, and clean version control hygiene.

## Executive Summary

Coal YOLO Model Suite consolidates multiple model generations and training strategies into a single professional workspace:

- Baseline and advanced classification pipelines
- Dataset preparation and metadata-driven configuration
- Evaluation and reporting utilities
- Hybrid-model experimentation for YOLOv11
- Code-only packaging suitable for GitHub sharing and collaboration

## Key Highlights

- Multi-version model stack: YOLOv8, YOLOv10, YOLOv11, and YOLOv12-style experimentation
- Curated source control policy: excludes heavy artifacts and sensitive bulk data
- Unified setup: root-level `requirements.txt` plus project-specific requirements
- Reproducible data workflow with documented dataset source and ingestion guidance
- Clear module boundaries for training, inference, evaluation, and utilities

## Repository Scope

Included projects:
- `YOLOv8_Coal_Classification_Project`
- `YOLOv10_Coal_Classification_Project`
- `YOLOv11_Coal_Classification_Project`
- `Yolov12`

Excluded artifacts by design:
- raw and split image datasets
- screenshots and sample image dumps
- trained checkpoints and binary model outputs (`*.pt`, `*.pth`, `*.onnx`, `*.engine`)
- generated reports and visual outputs (`*.png`, `*.pdf`, `*.csv`, `*.json`)
- run outputs and caches (`runs/`, `__pycache__/`, and related folders)

## Dataset Information

Primary dataset source:
- https://www.scidb.cn/en/detail?dataSetId=5654e40ae6d14ade84bac79cb0753852&version=V1

Referenced archive groups:
- `01.Non-destructive coal.zip`
- `02.Destructive coal.zip`
- `03.Strongly destructive coal.zip`
- `04.Pulverized coal.zip`
- `05.Fully pulverized coal.zip`

Detailed data handling guidance is available in `DATASET.md`.

## Project Architecture

```text
coal-yolo-model-suite/
|- README.md
|- DATASET.md
|- requirements.txt
|- YOLOv8_Coal_Classification_Project/
|  |- scripts/
|  |- data/
|  |- model/
|  |- requirements.txt
|- YOLOv10_Coal_Classification_Project/
|  |- train_diversified_model.py
|  |- comprehensive_evaluation.py
|  |- production_classifier.py
|  |- coal_classification.yaml
|  |- requirements.txt
|- YOLOv11_Coal_Classification_Project/
|  |- train_model.py
|  |- train_model_regularized.py
|  |- evaluate_model.py
|  |- yolo11_a2mada/
|  |- yolo11_hybridmodel/
|  |- requirements.txt
|- Yolov12/
	|- yolov12_model.py
	|- train_yolov12.py
	|- infer_yolov12.py
	|- organize_yolo_dataset.py
```

## Module-Level Details

### YOLOv8_Coal_Classification_Project
Primary purpose:
- Baseline and extended YOLOv8 classification workflow with preparation, training, inference, and utility scripts.

Important files:
- `scripts/train_coal_classifier.py`
- `scripts/evaluate_model.py`
- `scripts/prepare_dataset.py`
- `data/balanced_data.yaml`

### YOLOv10_Coal_Classification_Project
Primary purpose:
- Diversified training and evaluation workflow with production classification entry points.

Important files:
- `train_diversified_model.py`
- `comprehensive_evaluation.py`
- `production_classifier.py`
- `coal_classification.yaml`

### YOLOv11_Coal_Classification_Project
Primary purpose:
- Standard training/evaluation plus advanced hybrid and attention-alignment experiments.

Important files:
- `train_model.py`
- `train_model_regularized.py`
- `evaluate_model.py`
- `evaluate_all_metrics.py`

Advanced submodules:
- `yolo11_a2mada/` (attention alignment model variant)
- `yolo11_hybridmodel/` (hybrid architecture, ensemble training, inference, utilities)

### Yolov12
Primary purpose:
- Experimental lightweight YOLOv12-like model stack for custom training and inference flows.

Important files:
- `yolov12_model.py`
- `train_yolov12.py`
- `infer_yolov12.py`
- `organize_yolo_dataset.py`

## Installation

### Option 1: Unified root environment (recommended)

```bash
git clone https://github.com/abhinavsai2006/coal-yolo-model-suite.git
cd coal-yolo-model-suite
python -m venv .venv
# PowerShell
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: Per-project environment

```bash
cd YOLOv11_Coal_Classification_Project
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage Workflows

### YOLOv11 baseline example

```bash
cd YOLOv11_Coal_Classification_Project
python train_model.py
python evaluate_model.py
```

### YOLOv10 diversified pipeline example

```bash
cd YOLOv10_Coal_Classification_Project
python train_diversified_model.py
python comprehensive_evaluation.py
```

### YOLOv8 baseline training example

```bash
cd YOLOv8_Coal_Classification_Project
python scripts/train_coal_classifier.py
python scripts/evaluate_model.py
```

### Yolov12 experiment example

```bash
cd Yolov12
python train_yolov12.py
python infer_yolov12.py path/to/image.jpg
```

## Data and Reproducibility Policy

- Dataset images are intentionally excluded from source control.
- Download dataset archives from SciDB using the documented URL.
- Keep extracted data in local, ignored paths.
- Update YAML/data paths to match your machine and environment.
- Commit only source code, configuration templates, and lightweight documentation.

## Dependency Strategy

- Root `requirements.txt` provides a consolidated dependency baseline.
- Project-level `requirements.txt` files preserve module-specific constraints.
- For production work, freeze exact versions after environment validation.

## Quality and Maintenance Guidance

- Use one experiment directory per run outside tracked source paths.
- Store generated metrics and charts in ignored output directories.
- Keep training scripts deterministic where possible (seed control and fixed splits).
- Document any architecture or hyperparameter changes in project-level notes.

## License and Attribution

- Existing project-level licenses are preserved where provided.
- Dataset ownership and terms remain with the source provider.
- Ensure compliance with all upstream model and dataset licenses before redistribution.

---

Built for professional coal classification research, reproducible experimentation, and clean deployment handoff.
<div align="center">

# 🏭 Coal Classification AI Model Suite 🏭

**A comprehensive, monorepo suite of advanced YOLO (You Only Look Once) deep learning architectures dedicated to the highly accurate classification of distinct coal deformation states for mining safety and geological analysis.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO-Ultralytics-1C1C1C?style=for-the-badge&logo=yolo&logoColor=white)](https://ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![JSON](https://img.shields.io/badge/JSON-Metadata-000000?style=for-the-badge&logo=json&logoColor=white)](https://json.org)
[![CSV](https://img.shields.io/badge/CSV-Metrics-43B02A?style=for-the-badge&logo=microsoftexcel&logoColor=white)](https://en.wikipedia.org/wiki/Comma-separated_values)
[![ONNX](https://img.shields.io/badge/ONNX-Export-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnx.ai/)

</div>

---

## 📖 Executive Summary
This repository houses the entire evolutionary pipeline of advanced deep-learning vision models trained on microscopic and macroscopic geological samples of coal structures. The suite encompasses experiments across **YOLOv8**, **YOLOv10**, **YOLOv11**, and prototype **YOLOv12**, systematically comparing structural prediction metrics (Precision, Recall, mAP).

As a direct integration of both **code and heavy artifacts**, this repository tracks complete reproducibility pipelines including model weights (`*.pt`, `*.onnx`), evaluation metrics (`*.csv`, `*.json`), and generated analytical charts (`*.png`, `*.pdf`)—meaning an engineer can clone, load a tensor graph, and infer instantly without secondary artifact stores. 

---

## 🗃️ Coal Taxonomy & Dataset Schema
The core of the models revolves around distinguishing six highly specific geological phenomena. Understanding the states of destructive coal is critical for predicting mining hazards such as gas outbursts. 

| Icon | Classification Category | Description | Data Composition / Visual Features |
|:---:|:---|:---|:---|
| 🪨 | **01. Non-destructive coal** | Native coal structure with primary porosity preserved. | Intact banding, smooth fracture surfaces. |
| 🏚️ | **02. Destructive coal** | Initial fracturing observed under tectonic stress. | Mild fragmentation, visible micro-fissures. |
| 💥 | **03. Strongly destructive coal** | Highly fractured structure with distinct mylonitic zones. | Widespread fracture networks, loose aggregates. |
| 🌫️ | **04. Pulverized coal**| Coal crushed to a coarse powder state. | Granular texture, lack of cohesive macro-structure. |
| 💨 | **05. Fully pulverized coal** | Complete structural breakdown into fine dust/powder. | Homogenous fine particulate mass, high surface area. |
| 🚫 | **Not coal** | Background noise, non-coal geological formations (shale/sandstone) or equipment. | Tool marks, extraneous minerals, metallic background. |

---

## ⚙️ Data Preprocessing Pipeline 
In order to handle varying ambient illumination, scale, and noise from the geological lab equipment, the pipeline incorporates the following steps before hitting the Neural Network:

1. **Dimensional Normalization**: Images are uniformly scaled to `[224x224]` or `[640x640]` depending on the model tier to conserve GPU memory while retaining granular structural data.
2. **Channel Normalization**: Z-score normalization utilizing ImageNet `mean` and `standard deviation` arrays to stabilize gradient descent.
3. **Augmentation Injection (A2MADA)**: Heavily utilized in the `YOLOv11` branches, generating synthetic orientations (rotate, flip) and lighting variants (hsv_s, hsv_v) to mimic different microscopic lighting environments.
4. **Data Splitting**: Strict `80/20` Train/Validation splits defined in `coal_*.yaml` structures directly integrated via PyTorch DataLoaders.

---

## 🧠 Model Architectures & Evolution

This project evaluates the timeline of YOLO's classification capabilities, capitalizing on architectural leaps from Ultralytics and community forks:

### 1️⃣ YOLOv8 (The Baseline)
- **Architecture Base**: CSPDarknet53 backbone with an anchor-free detection/classification head.
- **Focus**: Establishing baselines for mAP50-95 in coal fragmentation.
- **Artifacts**: Find baseline `yolo11n.pt` and `inference_results.csv` within the `YOLOv8_Coal_Classification_Project` directory.

### 2️⃣ YOLOv10 (Diversified Focus)
- **Architecture Base**: Introduces NMS-free training via dual assignments and extensive focus on reducing computational overhead during inference.
- **Focus**: Deployments on constrained Edge Devices (production classifiers).
- **Artifacts**: Weights stored in `/yolov10_diversified/` alongside evaluation pipelines.

### 3️⃣ YOLOv11 (A2MADA / Hybrid Models)
- **Architecture Base**: Enhanced feature extraction networks (C3k2 blocks and SPPF) bringing parameter efficiency. 
- **Focus**: The pinnacle of the project's accuracy. Utilizing complex hyperparameter regularization (`train_model_regularized.py`) to prevent overfitting on the minor variations between "Pulverized" vs "Fully pulverized".
- **Artifacts**: Contains highly detailed `report_figures/` and robust `.pt` exports. See `report_summary.md`.

### 4️⃣ YOLOv12 (Prototype / Experimental)
- **Architecture Base**: Implementing the bleeding-edge components of the YOLO trajectory.
- **Focus**: Evaluating if newer blocks yield better sub-class feature differentiation on geological materials.
- **Artifacts**: Standalone `yolov12_coal.pth` standard PyTorch tensor files.

---

## 📊 Artifacts & Model Checkpoints Included

By policy, this repository **tracks** the generated state artifacts to ensure an instantly verifiable development lifecycle. The following data files are committed alongside the code:

*   **Model Weights**: `*.pt` (PyTorch state dictionaries), `*.pth`, `*.onnx` (Open Neural Network Exchange), `*.engine` (TensorRT optimized formats).
*   **Run Outputs (`runs/`)**: Complete Ultralytics output directories containing `args.yaml`, loss visualizations, and confusion matrices.
*   **Generated Dashboards**: `*.png` visualizations of the loss functions over epochs, and PR (Precision-Recall) curves.
*   **Detailed Analytics**: `.csv` aggregations of raw predictions versus ground truth, and comprehensive `.pdf` project reports generated post-evaluation.

---

## 🏗️ Project Structure
```text
coal-yolo-model-suite/
│
├── DATASET.md                     # Central instructions on dataset acquisition/handling
├── requirements.txt               # Global PyPi Dependencies
│
├── YOLOv8_Coal_Classification_Project/   # Baseline implementations
│   ├── coal_classification_metrics.csv
│   └── inference_results_not_coal.csv
│
├── YOLOv10_Coal_Classification_Project/  # Edge Inference implementations
│   ├── production_classifier.py
│   ├── train_diversified_model.py
│   ├── comprehensive_evaluation/         # Output reports (.json, .csv, .png)
│   └── yolov10_diversified/              # Model Checkpoints (*.pt)
│
├── YOLOv11_Coal_Classification_Project/  # Heavy Regularization & Best Accuracy
│   ├── train_model_regularized.py
│   ├── generate_detailed_report.py
│   ├── report_figures/                   # Visual output artifacts
│   ├── runs/                             # Ultralytics train/val logs and plots
│   ├── yolo11_a2mada/                    # Experimental Weights
│   └── yolo11_hybridmodel/               # Production Weights (*.pt)
│
└── Yolov12/                              # Next-generation Architectures
    ├── train_yolov12.py
    └── yolov12_coal.pth                  # Bare PyTorch weights for YOLOv12 config
```

---

## 🚀 Quick Start & Inference

**1. Clone the repository natively (including weights):**
```bash
git clone https://github.com/abhinavsai2006/coal-yolo-model-suite.git
cd coal-yolo-model-suite
```

**2. Hydrate the environment:**
*(We recommend Conda/Miniconda to isolate PyTorch GPU binaries)*
```bash
pip install -r requirements.txt
```

**3. Run an Inference with YOLOv11 (Best Model):**
```python
from ultralytics import YOLO

# Load the tracked weights directly from the repo
model = YOLO('YOLOv11_Coal_Classification_Project/yolo11_hybridmodel/weights/best.pt')

# Infer on a new lab image
results = model.predict('path/to/some_microscope_image.jpg')

# Process results
print(f"Predicted class: {results[0].names[results[0].probs.top1]}")
```

## 📜 License
Dual-licensed inherently. Scripts bound under `LICENSE` provided. Checkpoint `.pt` licenses inherit the AGPL-3.0 strictly derived from Ultralytics LLC standards unless exclusively retrained entirely from scratch without pre-trained backbones.

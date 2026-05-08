# YOLOv10 Coal Classification Project 🔥

A complete implementation of **true YOLOv10 classification** for coal type identification using the official THU-MIG YOLOv10 repository.

## 🎯 Project Overview

This project implements authentic YOLOv10 classification to identify 6 different types of coal with **97.91% accuracy**:

1. **destructive_coal** - Coal with destructive properties
2. **fully_pulverized_coal** - Completely pulverized coal
3. **non_coal** - Non-coal materials  
4. **non_destructive_coal** - Non-destructive coal
5. **pulverized_coal** - Partially pulverized coal
6. **strongly_destructive_coal** - Highly destructive coal

## 📊 Model Performance

- **Overall Accuracy**: 97.91%
- **F1-Score**: 97.93%
- **Inference Speed**: 94.63 ms/image (10.6 FPS)
- **Total Dataset**: 2,536 images
- **Test Accuracy**: 97.91% on 478 test images

### Per-Class Performance
| Class | Accuracy | Test Images |
|-------|----------|-------------|
| fully_pulverized_coal | 100.0% | 70/70 |
| non_coal | 100.0% | 88/88 |
| destructive_coal | 97.9% | 94/96 |
| non_destructive_coal | 97.4% | 76/78 |
| strongly_destructive_coal | 97.4% | 74/76 |
| pulverized_coal | 94.3% | 66/70 |

## 🏗️ Project Structure

```
YOLOv10_Coal_Classification_Project/
├── 📁 data/                          # Original raw dataset
├── 📁 dataset/                       # Processed train/val/test splits
├── 📁 yolov10_gpu_classification/    # Trained model weights
├── 📁 evaluation_plots/              # Performance visualizations
├── 📁 evaluation_results/            # Detailed evaluation reports
├── 📁 random_test_results/           # Random testing outputs
├── 📄 data_preparation.py            # Dataset preparation script
├── 📄 yolov10_gpu_classification.py  # Main training script
├── 📄 evaluate_yolov10_model.py      # Model evaluation script
├── 📄 test_random_images.py          # Random image testing
├── 📄 coal_classification.yaml       # Dataset configuration
├── 📄 requirements.txt               # Python dependencies
└── 📄 README.md                      # Project documentation
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create conda environment
conda create -n yolov9_gpu python=3.9
conda activate yolov9_gpu

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset
```bash
python data_preparation.py
```

### 3. Train Model
```bash
python yolov10_gpu_classification.py
```

### 4. Evaluate Model
```bash
python evaluate_yolov10_model.py
```

### 5. Test Random Images
```bash
python test_random_images.py
```

## 📋 Requirements

- Python 3.9+
- PyTorch with CUDA support
- Ultralytics YOLOv10
- OpenCV
- Matplotlib
- Scikit-learn
- Pandas
- NumPy
- PIL

## 🎯 Model Details

- **Architecture**: YOLOv10 Classification (YOLOv8n-cls backbone)
- **Framework**: Official Ultralytics YOLOv10
- **Repository**: THU-MIG/yolov10 (official)
- **Training**: 50 epochs with GPU acceleration
- **Input Size**: 224x224 pixels
- **Optimizer**: AdamW with cosine learning rate scheduling

## 📈 Results

The model achieves exceptional performance:

- **Training Accuracy**: 98.5% (validation)
- **Test Accuracy**: 97.91%
- **Perfect Classification**: 100% on `fully_pulverized_coal` and `non_coal`
- **Real-time Capable**: 10.6 FPS on GPU
- **Industrial Ready**: Suitable for production coal classification

## 🔧 Usage Examples

### Evaluate Model
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('yolov10_gpu_classification/coal_gpu_n/weights/best.pt')

# Predict on image
results = model('path/to/coal_image.jpg')
print(f"Prediction: {results[0].probs.top1}")
```

### Random Testing
```python
# Test on random images
python test_random_images.py

# Choose option 1-5:
# 1. Test 10 random images
# 2. Test 15 random images  
# 3. Visualize 6 predictions
# 4. Test specific classes
# 5. Run all tests
```

## 📊 Files Description

| File | Purpose |
|------|---------|
| `data_preparation.py` | Organizes raw data into train/val/test splits (70%/20%/10%) |
| `yolov10_gpu_classification.py` | Main training script with GPU optimization |
| `evaluate_yolov10_model.py` | Comprehensive model evaluation and reporting |
| `test_random_images.py` | Random image testing with visualization |
| `coal_classification.yaml` | Dataset configuration for YOLO training |

## 🏆 Key Features

- ✅ **Authentic YOLOv10**: Uses official THU-MIG implementation
- ✅ **High Accuracy**: 97.91% classification accuracy
- ✅ **GPU Accelerated**: CUDA-optimized training and inference
- ✅ **Comprehensive Evaluation**: Detailed performance analysis
- ✅ **Visualization**: Confusion matrices and performance plots
- ✅ **Random Testing**: Test on random samples with confidence scores
- ✅ **Industrial Ready**: Real-time classification capability

## 📞 Support

This implementation uses the official YOLOv10 repository from THU-MIG (Tsinghua University) ensuring authenticity and state-of-the-art performance for coal classification tasks.

**Model Repository**: [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)

---
*Successfully implemented authentic YOLOv10 classification with 97.91% accuracy on coal dataset* 🎉
- Training history saving (JSON format)

### Inference Features
- Single image prediction with confidence scores
- Batch inference on image folders
- Top-K predictions support
- Class distribution analysis
- Results export to JSON

## Output Files

After training, the following files are generated:

- `models/best_model.pth` - Best model checkpoint
- `models/training_history.json` - Training metrics
- `models/training_history.png` - Training plots
- `models/test_results.json` - Test evaluation results
- `models/confusion_matrix.png` - Confusion matrix visualization

## Performance Metrics

The model provides comprehensive evaluation:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1-Score**: Per-class and averaged metrics
- **Confusion Matrix**: Visual representation of predictions
- **Class-wise Performance**: Detailed breakdown by coal type

## Key Differences from Detection YOLOs

This is a **pure classification model** that:
- ❌ Does NOT use bounding boxes
- ❌ Does NOT detect object locations
- ✅ Classifies entire images into coal categories
- ✅ Uses YOLOv10-inspired efficient architecture
- ✅ Optimized for classification accuracy

## Example Results

```
Classification Report:
                        precision    recall  f1-score   support

       destructive_coal       0.95      0.93      0.94        67
  fully_pulverized_coal       0.92      0.94      0.93        52
              non_coal       0.97      0.98      0.98        89
   non_destructive_coal       0.90      0.88      0.89        73
        pulverized_coal       0.89      0.91      0.90        58
strongly_destructive_coal     0.94      0.92      0.93        61

              accuracy                           0.93       400
             macro avg       0.93      0.93      0.93       400
          weighted avg       0.93      0.93      0.93       400
```

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended (NVIDIA GTX 1060+ or better)
- **RAM**: 8GB+ system RAM
- **Storage**: 2GB+ free space for dataset and models

## Troubleshooting

1. **CUDA out of memory**: Reduce batch_size in config
2. **Dataset not found**: Run `data_preparation.py` first
3. **Import errors**: Install all required packages
4. **Slow training**: Ensure CUDA is properly installed

## License

This project is for educational and research purposes.
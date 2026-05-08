# YOLOv11 Coal Classification Project

This project uses YOLOv11 to classify different types of coal from images.

## Dataset Structure
The dataset contains 5 classes of coal:
- `destructive_coal`
- `fully_pulverized_coal`
- `non_destructive_coal`
- `pulverized_coal`
- `strongly_destructive_coal`

## Project Structure
```
├── data/                   # Dataset directory
│   ├── train/             # Training images
│   ├── val/               # Validation images
│   └── test/              # Test images
├── coal_data.yaml         # Dataset configuration
├── train_model.py         # Training script
├── evaluate_model.py      # Evaluation script
├── predict_samples.py     # Prediction script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model
```bash
python train_model.py
```
This will:
- Download the YOLOv11 nano classification model
- Train on your coal dataset for 100 epochs
- Save the best model to `runs/classify/coal_classification/weights/best.pt`
- Generate training plots and metrics

### 2. Evaluate the Model
```bash
python evaluate_model.py
```
This will:
- Load the trained model
- Evaluate on the test dataset
- Display accuracy metrics and per-class performance

### 3. Make Predictions
```bash
python predict_samples.py
```
This will:
- Load the trained model
- Make predictions on random test images
- Show predicted vs actual classes with confidence scores

## Model Configuration

The training uses these default parameters:
- **Model**: YOLOv11 nano classification (`yolo11n-cls.pt`)
- **Image Size**: 224x224 pixels
- **Batch Size**: 16
- **Epochs**: 100
- **Learning Rate**: 0.01
- **Early Stopping**: 10 epochs patience

You can modify these parameters in `train_model.py`.

## Results

After training, you'll find:
- Trained model: `runs/classify/coal_classification/weights/best.pt`
- Training plots: `runs/classify/coal_classification/`
- Validation metrics during training

## GPU Support

To use GPU for faster training, change the device parameter in `train_model.py`:
```python
device='cuda:0'  # instead of 'cpu'
```

Make sure you have CUDA and PyTorch GPU support installed.
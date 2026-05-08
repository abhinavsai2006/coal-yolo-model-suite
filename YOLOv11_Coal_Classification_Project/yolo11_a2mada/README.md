# A2MADA-YOLO for Coal Classification

**Attention Alignment Multiscale Adversarial Domain Adaptation YOLO**

Based on research: [A2MADA-YOLO GitHub Repository](https://github.com/HaoxingZhou/A2MADA-YOLO)

## 🎯 Overview

A2MADA-YOLO is an advanced object detection/classification model that combines:
- **Attention Alignment**: CBAM-based channel and spatial attention
- **Multiscale Processing**: Features processed at multiple scales (1x, 2x, 4x)
- **Adversarial Domain Adaptation**: Gradient reversal for robust feature learning
- **YOLO Architecture**: Fast and efficient backbone

## 🏗️ Architecture Features

### 1. **Attention Alignment Module**
- Channel attention (squeeze-excitation)
- Spatial attention (7x7 convolution)
- CBAM (Convolutional Block Attention Module)
- Domain classifier for alignment

### 2. **Multiscale Attention Alignment**
- Processes features at scales: 1x, 2x, 4x
- Independent attention for each scale
- Fusion layer to combine multiscale features

### 3. **Adversarial Attention**
- Gradient Reversal Layer (GRL)
- Domain discriminator
- Task-specific class predictor
- Adversarial training for robust features

### 4. **Backbone**
- YOLO-style stem and stages
- C3 blocks with bottlenecks
- SPPF (Spatial Pyramid Pooling Fast)
- Attention at each stage

## 📊 Model Specifications

```
Total Parameters: ~8-10M
Model Size: ~30-40 MB
Input Size: 224x224
Classes: 6 (coal types)
```

## 🚀 Quick Start

### 1. Test the Model

```bash
cd yolo11_a2mada
python a2mada_model.py
```

Expected output:
```
Testing A2MADA-YOLO Model...
Total parameters: X,XXX,XXX
Trainable parameters: X,XXX,XXX
Model size: XX.XX MB
✓ A2MADA-YOLO model test passed!
```

### 2. Train the Model

```bash
python train_a2mada.py --epochs 50 --batch-size 32 --lr 0.001
```

Training features:
- **Adversarial training** with gradient reversal
- **Multiscale domain adaptation**
- **Dynamic alpha scheduling** (gradually increases)
- **Cosine annealing** learning rate
- **Gradient clipping** for stability

### 3. Evaluate the Model

```bash
python evaluate_a2mada.py --weights runs/a2mada_XXXXXX/weights/best.pt --test-dir ../data/test
```

## 📈 Training Strategy

### Loss Components

1. **Classification Loss**: Cross-entropy for coal type prediction
2. **Domain Loss**: BCE for domain adaptation (main)
3. **Multiscale Domain Loss**: BCE for each attention scale

Total Loss = Classification Loss + 0.1 × Domain Loss + 0.05 × Multiscale Domain Loss

### Alpha Scheduling

The gradient reversal weight (alpha) gradually increases:

```python
alpha = 2.0 / (1.0 + exp(-10 * epoch / total_epochs)) - 1.0
```

This allows the model to:
- Learn basic features early (low alpha)
- Improve domain invariance later (high alpha)

## 🎓 Key Concepts

### Adversarial Domain Adaptation

The model learns features that are:
- **Discriminative** for classification (high class accuracy)
- **Invariant** to domain shifts (robust to variations)

### Gradient Reversal Layer (GRL)

During backpropagation:
- **Forward pass**: Identity function (x → x)
- **Backward pass**: Reverses gradients (∂L/∂x → -α × ∂L/∂x)

This forces the feature extractor to learn domain-invariant features.

## 📁 Project Structure

```
yolo11_a2mada/
├── modules/
│   └── attention_alignment.py    # Attention modules
├── a2mada_model.py               # Model architecture
├── train_a2mada.py               # Training script
├── evaluate_a2mada.py            # Evaluation script
├── README.md                     # This file
└── runs/                         # Training outputs
    └── a2mada_XXXXXX/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        └── train_history.json
```

## 🔧 Configuration

### Default Parameters

```python
epochs = 50
batch_size = 32
learning_rate = 0.001
img_size = 224
optimizer = AdamW (weight_decay=0.01)
scheduler = CosineAnnealingLR
```

### Augmentations

- Random crop (1.1x resize)
- Random horizontal flip (0.5)
- Random vertical flip (0.3)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)

## 📊 Expected Performance

Based on adversarial training and attention mechanisms:
- **Training Time**: ~25-30 minutes (50 epochs on GPU)
- **Target Accuracy**: 90-93%
- **Robustness**: Better generalization than standard models

## 🔍 Training Modes

### 1. Adversarial Mode (Training)

```python
output, domain_pred, domain_outputs = model(x, alpha=1.0, use_adversarial=True)
```

Returns:
- Class predictions
- Domain predictions
- Multiscale domain outputs

### 2. Inference Mode (Testing)

```python
output = model(x, use_adversarial=False)
```

Returns:
- Class predictions only

## 🎯 Advantages

1. **Domain Robustness**: Adversarial training improves generalization
2. **Multiscale Features**: Better handling of scale variations
3. **Attention Mechanisms**: Focuses on important regions
4. **Fast Training**: 50 epochs to convergence
5. **YOLO Efficiency**: Fast inference speed

## 📚 References

- Original A2MADA-YOLO: https://github.com/HaoxingZhou/A2MADA-YOLO
- CBAM: Convolutional Block Attention Module
- Domain Adaptation: Gradient Reversal Layer
- YOLOv11: Modern YOLO architecture

## 🚀 Tips for Best Results

1. **Start with 50 epochs** - Usually sufficient for convergence
2. **Monitor alpha values** - Should gradually increase to ~0.76
3. **Watch domain loss** - Should stabilize around 0.69 (BCE loss for 0.5 target)
4. **Use GPU** - Much faster than CPU
5. **Adjust alpha weight** - Increase if need more robustness

## ✅ Success Indicators

- Training accuracy: 90-95%
- Validation accuracy: 88-92%
- Domain loss: ~0.69 (optimal for 0.5 target)
- Smooth loss curves
- No severe overfitting

---

**Created for Coal Classification Project**
**Target: 92%+ Accuracy with Domain Robustness**

"""Evaluate a single model with multi-scale test-time augmentation.

Scales used: 224, 256, 288 (center-crop / resize strategy)
We average softmax probabilities across scales to potentially improve
robustness and accuracy without retraining.

Usage (PowerShell):
  & "E:\Yolo\yolov9_gpu\python.exe" \
    "E:\Yolo\YOLOv11_Coal_Classification_Project\yolo11_hybridmodel\inference\evaluate_single_multiscale.py" \
    --weights "E:\Yolo\YOLOv11_Coal_Classification_Project\yolo11_hybridmodel\runs\ensemble_seed789_20251109_035230\weights\best.pt" \
    --output evaluation_single_multiscale_seed789

"""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
import sys
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from models.hybrid_model_pretrained import create_pretrained_hybrid_model


def build_loader(scale: int, batch_size: int = 32):
    # Resize to scale then center-crop (scale, scale) (effectively identical)
    transform = transforms.Compose([
        transforms.Resize((scale, scale)),
        transforms.CenterCrop((scale, scale)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data_root = Path(__file__).parent.parent.parent / 'data'
    test_dataset = datasets.ImageFolder(data_root / 'test', transform=transform)
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return loader, test_dataset.classes


def load_model(weight_path: str, num_classes: int, device: torch.device):
    """Load checkpoint and auto-detect backbone (resnet50 vs efficientnet_b3)."""
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    keys = list(state_dict.keys())
    if any(k.startswith('backbone.features') for k in keys):
        backbone = 'efficientnet_b3'
    elif any(k.startswith('backbone.conv1') for k in keys):
        backbone = 'resnet50'
    else:
        backbone = 'resnet50'
    print(f'Auto-detected backbone: {backbone}')
    wrapper = create_pretrained_hybrid_model(num_classes=num_classes, backbone_type=backbone, pretrained=True, dropout=0.6)
    model = wrapper.build_model().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_multiscale(model, scales, device, batch_size=32):
    # Build loaders per scale once
    loaders = []
    for s in scales:
        l, classes = build_loader(s, batch_size)
        loaders.append(l)
    classes = loaders[0].dataset.classes
    num_samples = len(loaders[0].dataset)

    # Collect probabilities (num_samples, num_classes) per scale
    all_scale_probs = []
    with torch.no_grad():
        for scale_loader in loaders:
            scale_probs = []
            for inputs, _ in scale_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                scale_probs.append(probs)
            all_scale_probs.append(np.concatenate(scale_probs, axis=0))

    # Average across scales
    avg_probs = np.mean(all_scale_probs, axis=0)
    predictions = np.argmax(avg_probs, axis=1)
    labels = np.array([label for _, label in loaders[0].dataset.samples])  # integer labels
    return labels, predictions, avg_probs, classes


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Single Model Multi-Scale Evaluation')
    ap.add_argument('--weights', required=True, help='Path to model checkpoint')
    ap.add_argument('--output', default='evaluation_single_multiscale', help='Output folder name')
    ap.add_argument('--scales', type=int, nargs='+', default=[224, 256, 288], help='List of scales (square)')
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 70)
    print(' SINGLE MODEL MULTI-SCALE EVALUATION')
    print('=' * 70)
    print(f'Checkpoint: {args.weights}')
    print(f'Scales: {args.scales}')
    print(f'Device: {device}')
    print('=' * 70)

    # Load classes (from smallest scale loader) to get num_classes
    tmp_loader, classes = build_loader(args.scales[0], batch_size=args.batch_size)
    num_classes = len(classes)
    del tmp_loader  # Will rebuild anyway

    # Load model
    model = load_model(args.weights, num_classes, device)

    # Evaluate
    labels, predictions, probs, classes = evaluate_multiscale(model, args.scales, device, batch_size=args.batch_size)
    accuracy = 100.0 * (predictions == labels).sum() / len(labels)
    report = classification_report(labels, predictions, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, predictions)

    # Output directory
    out_dir = Path(__file__).parent.parent / args.output
    out_dir.mkdir(exist_ok=True)

    results = {
        'checkpoint': args.weights,
        'scales': args.scales,
        'test_accuracy': float(accuracy),
        'num_test_samples': len(labels),
        'classes': classes,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    with open(out_dir / 'multiscale_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plots
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Single Model Multi-Scale - Confusion Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(out_dir / 'multiscale_confusion_matrix.png', dpi=300)
    plt.close()

    # Per-class accuracy
    per_class_acc = []
    for i, cname in enumerate(classes):
        mask = labels == i
        acc_i = 100.0 * (predictions[mask] == labels[mask]).sum() / mask.sum()
        per_class_acc.append(acc_i)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(classes)), per_class_acc, color='mediumseagreen')
    plt.axhline(y=np.mean(per_class_acc), color='red', linestyle='--', label=f'Mean {np.mean(per_class_acc):.2f}%')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.ylabel('Accuracy (%)')
    plt.title('Single Model Multi-Scale - Per-Class Accuracy', fontsize=16)
    for i, b in enumerate(bars):
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h + 1, f'{per_class_acc[i]:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'multiscale_class_accuracy.png', dpi=300)
    plt.close()

    print('\n' + '=' * 70)
    print(f'Test Accuracy (multi-scale): {accuracy:.2f}%')
    print('=' * 70)
    if accuracy >= 92.0:
        print('🎉 SUCCESS! Reached 92%+ test accuracy!')
    else:
        print(f'Gap to target: {92.0 - accuracy:.2f}%')
    print('=' * 70)
    print(f'Results saved to {out_dir}')


if __name__ == '__main__':
    main()

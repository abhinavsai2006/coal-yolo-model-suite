"""
Evaluate Ensemble of Models on Test Set

Loads multiple models and averages their predictions for final classification.
This typically improves accuracy by 1-2% over single models.

Author: YOLOv11 Hybrid Model Project
Date: November 2025
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
import json
import sys
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.hybrid_model_pretrained import create_pretrained_hybrid_model

# --- Helper for TenCrop TTA (must be top-level for Windows pickling) ---
def apply_tencrop(crops):
    """Convert PIL crops from TenCrop into a stacked normalized tensor.

    This is defined at module scope so DataLoader workers can pickle it on Windows.
    """
    return torch.stack([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])(
            transforms.ToTensor()(c)
        ) for c in crops
    ])


def get_test_loader(batch_size=32, tta: str = 'none'):
    """Create test data loader with validation-style preprocessing.

    Use Resize(256) + CenterCrop(224) to better match training distribution
    and reduce aspect-ratio distortion versus direct 224x224 resize.
    """
    if tta == 'tencrop':
        # TenCrop returns 10 images (4 corners + center + their horizontal flips)
        # Convert to tensor and normalize each crop -> (10, C, H, W)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.TenCrop(224),
            transforms.Lambda(apply_tencrop),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    data_root = Path(__file__).parent.parent.parent / 'data'
    test_dataset = datasets.ImageFolder(data_root / 'test', transform=transform)
    # For TenCrop we disable multiprocessing workers on Windows to avoid
    # edge-case pickling overhead with large tensors (optional, safe).
    num_workers = 0 if tta == 'tencrop' else 4
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return test_loader, test_dataset.classes


def load_model(weight_path, num_classes, device):
    """Load a single model from checkpoint using the wrapper factory.

    Auto-detects the backbone type (resnet50 vs efficientnet_b3) from the
    state_dict keys to ensure correct architecture restoration.
    """
    # Load checkpoint first to inspect keys
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Heuristic backbone detection from state_dict keys
    keys = list(state_dict.keys())
    if any(k.startswith('backbone.features') for k in keys):
        detected_backbone = 'efficientnet_b3'
    elif any(k.startswith('backbone.conv1') for k in keys):
        detected_backbone = 'resnet50'
    else:
        # Fallback to resnet50 if uncertain
        detected_backbone = 'resnet50'

    print(f"    ↳ Detected backbone: {detected_backbone}")

    # Build the appropriate model
    wrapper = create_pretrained_hybrid_model(
        num_classes=num_classes,
        backbone_type=detected_backbone,
        pretrained=True,
        dropout=0.6,
    )
    model = wrapper.build_model()

    # Load weights (strict to catch mismatches early)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model


def ensemble_predict(models, test_loader, device, tta: str = 'none', model_weights=None):
    """Get ensemble predictions with optional test-time augmentation (TTA).

    tta: 'none' | 'hflip'
      - none: single forward pass per model
      - hflip: average predictions from original and horizontal-flipped image
            - tencrop: average predictions across 10 fixed crops
    """
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            # For TenCrop, inputs will be (B, 10, C, H, W)
            is_tencrop = (tta == 'tencrop') and (inputs.dim() == 5)
            if is_tencrop:
                B, NC, C, H, W = inputs.shape  # NC=10
                inputs = inputs.view(B * NC, C, H, W).to(device)
            else:
                inputs = inputs.to(device)
            
            # Collect predictions from all models
            batch_probs = []
            for model in models:
                if is_tencrop:
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    probs = probs.view(-1, 10, probs.size(-1)).mean(dim=1)
                elif tta == 'hflip':
                    # Original
                    outputs1 = model(inputs)
                    # Horizontal flip
                    inputs_flipped = torch.flip(inputs, dims=[3])
                    outputs2 = model(inputs_flipped)
                    outputs = (outputs1 + outputs2) / 2.0
                else:
                    outputs = model(inputs)
                if not is_tencrop:
                    probs = torch.softmax(outputs, dim=1)
                batch_probs.append(probs.cpu().numpy())
            
            # Average probabilities across models (weighted if provided)
            if model_weights is not None:
                avg_probs = np.average(batch_probs, axis=0, weights=model_weights)
            else:
                avg_probs = np.mean(batch_probs, axis=0)
            
            all_probs.extend(avg_probs)
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    predictions = np.argmax(all_probs, axis=1)
    
    return all_labels, predictions, all_probs


def plot_confusion_matrix(cm, classes, output_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Ensemble Model - Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_accuracy(labels, predictions, classes, output_path):
    """Plot per-class accuracy"""
    accuracies = []
    for i, class_name in enumerate(classes):
        mask = labels == i
        class_acc = 100 * (predictions[mask] == labels[mask]).sum() / mask.sum()
        accuracies.append(class_acc)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(classes)), accuracies, color='steelblue')
    plt.axhline(y=np.mean(accuracies), color='red', linestyle='--', 
                label=f'Mean: {np.mean(accuracies):.2f}%')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Ensemble Model - Per-Class Accuracy', fontsize=16, pad=20)
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.ylim([0, 105])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{accuracies[i]:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Ensemble on Test Set')
    parser.add_argument('--weights', nargs='+', required=True,
                       help='Paths to model weights (space-separated)')
    parser.add_argument('--output', type=str, default='evaluation_ensemble',
                       help='Output directory name')
    parser.add_argument('--tta', type=str, default='none', choices=['none', 'hflip', 'tencrop'],
                        help='Apply test-time augmentation to boost accuracy')
    parser.add_argument('--weighted', action='store_true',
                        help='Use validation-accuracy-weighted ensemble averaging')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("  ENSEMBLE EVALUATION ON TEST SET")
    print("=" * 70)
    print(f"Number of models: {len(args.weights)}")
    print(f"Device: {device}")
    print("\nModel weights:")
    for i, w in enumerate(args.weights, 1):
        print(f"  {i}. {w}")
    print("=" * 70)
    
    # Load test data
    print("\nLoading test dataset...")
    test_loader, classes = get_test_loader(batch_size=32, tta=args.tta)
    num_classes = len(classes)
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    print(f"✓ Classes: {classes}")
    print(f"✓ TTA: {args.tta}")
    
    # Optionally compute model weights from validation accuracy
    model_weights = None
    if args.weighted:
        print("\nComputing model weights from validation accuracy...")
        weights = []
        for w in args.weights:
            try:
                ckpt = torch.load(w, map_location='cpu', weights_only=False)
                val_acc = ckpt.get('val_acc', None)
                if val_acc is None:
                    # Try reading training history
                    run_dir = Path(w).parent.parent
                    hist = run_dir / 'train_history.json'
                    if hist.exists():
                        with open(hist, 'r') as f:
                            hist_obj = json.load(f)
                        val_acc = max((e.get('acc', 0.0) for e in hist_obj.get('val', [])), default=0.0)
                if val_acc is None:
                    print(f"  - WARN: No val_acc found for {w}. Using 1.0 as fallback.")
                    val_acc = 1.0
                weights.append(float(val_acc))
            except Exception as e:
                print(f"  - WARN: Failed to read val_acc from {w}: {e}. Using 1.0.")
                weights.append(1.0)
        weights = np.array(weights, dtype=np.float32)
        if weights.sum() == 0:
            model_weights = None
            print("  - WARN: All weights zero; reverting to uniform averaging.")
        else:
            model_weights = (weights / weights.sum()).tolist()
            print("  Model weights (normalized):")
            for path, w in zip(args.weights, model_weights):
                print(f"    {path} -> {w:.4f}")

    # Load all models
    print("\nLoading models...")
    models = []
    for i, weight_path in enumerate(args.weights, 1):
        print(f"  Loading model {i}/{len(args.weights)}...")
        model = load_model(weight_path, num_classes, device)
        models.append(model)
    print(f"✓ All {len(models)} models loaded!")
    
    # Get ensemble predictions
    print("\nEvaluating ensemble...")
    labels, predictions, probs = ensemble_predict(models, test_loader, device, tta=args.tta, model_weights=model_weights)
    
    # Calculate accuracy
    accuracy = 100 * (predictions == labels).sum() / len(labels)
    
    # Generate classification report
    report = classification_report(labels, predictions, 
                                   target_names=classes,
                                   output_dict=True,
                                   zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Print results
    print("\n" + "=" * 70)
    print("  ENSEMBLE RESULTS")
    print("=" * 70)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("=" * 70)
    
    print("\nPer-Class Metrics:")
    print(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    print("-" * 80)
    
    for class_name in classes:
        metrics = report[class_name]
        print(f"{class_name:<30} {metrics['precision']*100:>10.2f}%  "
              f"{metrics['recall']*100:>10.2f}%  {metrics['f1-score']*100:>10.2f}%  "
              f"{int(metrics['support']):>7}")
    
    print("-" * 80)
    print(f"{'AVERAGE':<30} {report['macro avg']['precision']*100:>10.2f}%  "
          f"{report['macro avg']['recall']*100:>10.2f}%  "
          f"{report['macro avg']['f1-score']*100:>10.2f}%  "
          f"{len(labels):>7}")
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / args.output
    output_dir.mkdir(exist_ok=True)
    
    # Save results
    results = {
        'ensemble_size': len(models),
        'model_weights': args.weights,
        'weights_normalized': model_weights,
        'test_accuracy': float(accuracy),
        'num_test_samples': len(labels),
        'classes': classes,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'tta': args.tta,
    }
    
    with open(output_dir / 'ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrix
    print("\nGenerating visualizations...")
    plot_confusion_matrix(cm, classes, output_dir / 'ensemble_confusion_matrix.png')
    plot_class_accuracy(labels, predictions, classes, output_dir / 'ensemble_class_accuracy.png')
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - ensemble_results.json")
    print(f"  - ensemble_confusion_matrix.png")
    print(f"  - ensemble_class_accuracy.png")
    
    print("\n" + "=" * 70)
    if accuracy >= 92.0:
        print("🎉 SUCCESS! Reached 92%+ test accuracy!")
    else:
        print(f"📊 Current: {accuracy:.2f}% | Target: 92.00% | Gap: {92.0 - accuracy:.2f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()

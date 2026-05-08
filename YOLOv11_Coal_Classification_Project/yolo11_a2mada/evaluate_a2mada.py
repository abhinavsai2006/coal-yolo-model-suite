"""
Evaluation Script for A2MADA-YOLO
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent))
from a2mada_model import create_a2mada_model


class CoalDataset(Dataset):
    """Custom Dataset for Coal Classification"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            for img_path in class_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size=224):
    """Get evaluation transforms"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def evaluate_model(model, dataloader, device):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            outputs = model(inputs, use_adversarial=False)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, classes, output_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - A2MADA-YOLO')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate A2MADA-YOLO')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--test-dir', type=str, default='../data/test')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--output-dir', type=str, default='./evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("A2MADA-YOLO Evaluation")
    print("=" * 60)
    print(f"Using device: {args.device}")
    print(f"Weights: {args.weights}")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    test_dataset = CoalDataset(args.test_dir, get_transforms(args.img_size))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"\nTest samples: {len(test_dataset)}")
    print(f"Classes: {test_dataset.classes}")
    
    # Load model
    print("\nLoading model...")
    model = create_a2mada_model(num_classes=len(test_dataset.classes))
    
    checkpoint = torch.load(args.weights, map_location=args.device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(args.device)
    print("Model loaded successfully!")
    
    # Evaluate
    print("\nEvaluating model...")
    predictions, labels = evaluate_model(model, test_loader, args.device)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    cm = confusion_matrix(labels, predictions)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print("=" * 60)
    
    print("\nPer-Class Metrics:")
    print(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score'}")
    print("-" * 66)
    
    for i, class_name in enumerate(test_dataset.classes):
        print(f"{class_name:<30} {precision[i]*100:>10.2f}%  {recall[i]*100:>10.2f}%  {f1[i]*100:>10.2f}%")
    
    # Save results
    results = {
        'accuracy': float(accuracy * 100),
        'per_class_metrics': {
            class_name: {
                'precision': float(precision[i] * 100),
                'recall': float(recall[i] * 100),
                'f1': float(f1[i] * 100)
            }
            for i, class_name in enumerate(test_dataset.classes)
        },
        'confusion_matrix': cm.tolist()
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'evaluation_results.json'}")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, test_dataset.classes, output_dir / 'confusion_matrix.png')
    print(f"Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()

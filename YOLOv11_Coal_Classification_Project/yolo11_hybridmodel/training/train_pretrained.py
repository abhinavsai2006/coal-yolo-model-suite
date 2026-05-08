"""
Fast Training with Pretrained Backbone
Quick fine-tuning for 92%+ accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import sys
from datetime import datetime
from collections import Counter

sys.path.append(str(Path(__file__).parent))
from hybrid_model_pretrained import create_pretrained_hybrid_model


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


def get_transforms(img_size=224, is_training=True):
    """Get data transforms"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_class_weights(train_dataset):
    """Calculate class weights"""
    labels = [label for _, label in train_dataset.samples]
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    weights = torch.zeros(len(class_counts))
    for cls, count in class_counts.items():
        weights[cls] = total_samples / (len(class_counts) * count)
    
    return weights


def get_weighted_sampler(train_dataset):
    """Create weighted sampler"""
    labels = [label for _, label in train_dataset.samples]
    class_counts = Counter(labels)
    
    weights = [1.0 / class_counts[label] for label in labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    return sampler


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Training loop for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validation loop"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description='Train Pretrained Hybrid Model')
    parser.add_argument('--train-dir', type=str, default='../data/train')
    parser.add_argument('--val-dir', type=str, default='../data/val')
    parser.add_argument('--epochs', type=int, default=30, help='Quick fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001, help='Lower LR for fine-tuning')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'efficientnet_b3'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Pretrained Hybrid Model - Fast Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Backbone: {args.backbone}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 60)
    
    # Create datasets
    train_dataset = CoalDataset(args.train_dir, get_transforms(args.img_size, True))
    val_dataset = CoalDataset(args.val_dir, get_transforms(args.img_size, False))
    
    num_classes = len(train_dataset.classes)
    print(f"\nDataset Info:")
    print(f"Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Class weights and sampler
    class_weights = get_class_weights(train_dataset).to(args.device)
    weighted_sampler = get_weighted_sampler(train_dataset)
    
    print(f"\nClass Weights: {class_weights}")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=weighted_sampler,
        num_workers=4,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # Create model with pretrained backbone
    print(f"\nCreating model with pretrained {args.backbone} backbone...")
    wrapper = create_pretrained_hybrid_model(
        num_classes=num_classes,
        backbone_type=args.backbone,
        pretrained=True,
        dropout=0.5
    )
    model = wrapper.build_model()
    
    info = wrapper.get_model_info()
    print(f"Total parameters: {info['total_parameters']:,}")
    print(f"Trainable parameters: {info['trainable_parameters']:,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Optimizer with different LRs for backbone and classifier
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            classifier_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},  # Lower LR for pretrained backbone
        {'params': classifier_params, 'lr': args.lr}  # Higher LR for classifier
    ], weight_decay=0.01)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training history
    history = {'train': [], 'val': []}
    best_val_acc = 0.0
    
    # Create run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path('runs') / f'pretrained_{args.backbone}_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = run_dir / 'weights'
    weights_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving results to: {run_dir}")
    print("\nStarting training...\n")
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, args.device)
        
        # Update scheduler
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save history
        history['train'].append({'epoch': epoch + 1, 'loss': train_loss, 'acc': train_acc})
        history['val'].append({'epoch': epoch + 1, 'loss': val_loss, 'acc': val_acc})
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'backbone_type': args.backbone
            }, weights_dir / 'best.pt')
            print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Save last model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'backbone_type': args.backbone
        }, weights_dir / 'last.pt')
        
        print()
    
    # Save training history
    with open(run_dir / 'train_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("=" * 60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Results saved to: {run_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

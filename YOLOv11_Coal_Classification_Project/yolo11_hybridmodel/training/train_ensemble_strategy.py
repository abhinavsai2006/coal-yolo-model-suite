"""
Train Multiple Models for Ensemble - Strategy B to Reach 92%

This script trains new models with different random seeds for ensemble.
Each model will have slightly different learned patterns.

Author: YOLOv11 Hybrid Model Project
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.hybrid_model_pretrained import create_pretrained_hybrid_model


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(batch_size=32):
    """Create data loaders with moderate augmentation"""
    
    # Training augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/test transform
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    data_root = Path(__file__).parent.parent.parent / 'data'
    
    train_dataset = datasets.ImageFolder(data_root / 'train', transform=train_transform)
    val_dataset = datasets.ImageFolder(data_root / 'val', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, len(train_dataset.classes)


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train Models for Ensemble')
    parser.add_argument('--seed', type=int, required=True,
                       help='Random seed for this model')
    parser.add_argument('--epochs', type=int, default=40,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available! This script requires GPU.")
        sys.exit(1)
    
    print("=" * 70)
    print("  TRAINING MODEL FOR ENSEMBLE")
    print("=" * 70)
    print(f"Random seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Patience: {args.patience}")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print("=" * 70)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent.parent / 'runs' / f'ensemble_seed{args.seed}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / 'weights'
    weights_dir.mkdir(exist_ok=True)
    
    # Get data loaders
    print("\nLoading dataset...")
    train_loader, val_loader, num_classes = get_data_loaders(args.batch_size)
    print(f"✓ Dataset loaded!")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Classes: {num_classes}")
    
    # Create model
    print("\nCreating model...")
    wrapper = create_pretrained_hybrid_model(num_classes=num_classes, backbone_type='resnet50', pretrained=True, dropout=0.6)
    model = wrapper.build_model()
    model = model.to(device)
    print(f"✓ Model created!")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                       factor=0.5, patience=5, verbose=True)
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda')
    
    print("\n✓ Training setup complete!")
    print(f"  Optimizer: AdamW (lr={args.lr}, weight_decay=0.0001)")
    print(f"  Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    print(f"  Loss: CrossEntropyLoss (label_smoothing=0.15)")
    print(f"  Mixed Precision: Enabled (FP16)")
    print(f"  Gradient Clipping: max_norm=1.0")
    
    # Training history
    history = {'train': [], 'val': []}
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\n" + "=" * 70)
    print("  STARTING TRAINING")
    print("=" * 70)
    print(f"Target: Train robust model for ensemble")
    print(f"Output: {output_dir}")
    print("=" * 70 + "\n")
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                            optimizer, device, scaler)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save history
        history['train'].append({'epoch': epoch, 'loss': train_loss, 'acc': train_acc})
        history['val'].append({'epoch': epoch, 'loss': val_loss, 'acc': val_acc})
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.8f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'seed': args.seed
            }
            torch.save(checkpoint, weights_dir / 'best.pt')
            print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.patience}")
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch time: {epoch_time:.1f}s\n")
        
        # Early stopping
        if patience_counter >= args.patience:
            print("=" * 70)
            print(f"Early stopping triggered after {patience_counter} epochs without improvement")
            print("=" * 70)
            break
    
    # Save training history
    with open(output_dir / 'train_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved: {weights_dir / 'best.pt'}")
    print(f"Training history: {output_dir / 'train_history.json'}")
    print("=" * 70)


if __name__ == '__main__':
    main()

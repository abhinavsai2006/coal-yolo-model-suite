"""
Continue Training to Reach 92% Test Accuracy
Resumes from best checkpoint with optimized hyperparameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.hybrid_model_pretrained import PretrainedHybridWrapper


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_transforms(augmentation='moderate'):
    """Get data transforms with specified augmentation level"""
    if augmentation == 'moderate':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


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
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
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
    parser = argparse.ArgumentParser(description='Continue Training to Reach 92%')
    parser.add_argument('--checkpoint', type=str, 
                       default='runs/ensemble_fast_seed42/weights/best.pt',
                       help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=30, help='Number of additional epochs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.00003, help='Learning rate (3e-5)')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--train-dir', type=str, default='../data/train')
    parser.add_argument('--val-dir', type=str, default='../data/val')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    print(f"\n{'='*70}")
    print(f"  CONTINUING TRAINING TO REACH 92% TEST ACCURACY")
    print(f"{'='*70}\n")
    print(f"Random seed: {args.seed}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Additional epochs: {args.epochs}")
    print(f"Learning rate: {args.lr:.6f}")
    print(f"Batch size: {args.batch_size}")
    print(f"Patience: {args.patience}")
    
    # Setup device
    if not torch.cuda.is_available():
        print(f"\n{'='*70}")
        print("ERROR: CUDA not available! GPU is required for training.")
        print("Please activate the conda environment with CUDA support:")
        print("conda activate E:\\Yolo\\yolov9_gpu")
        print(f"{'='*70}\n")
        return
    
    device = torch.device('cuda')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Load data
    print(f"\n{'='*70}")
    print("Loading dataset...")
    print(f"{'='*70}\n")
    
    train_transform, val_transform = get_transforms('moderate')
    
    train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(args.val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Load model from checkpoint
    print(f"\n{'='*70}")
    print(f"Loading model from checkpoint...")
    print(f"{'='*70}\n")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model_config = {
        'num_classes': len(train_dataset.classes),
        'backbone_type': 'resnet50',
        'pretrained': False,
        'dropout': 0.6  # Increased dropout for better generalization
    }
    
    model_wrapper = PretrainedHybridWrapper(model_config=model_config)
    model = model_wrapper.build_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    previous_best = checkpoint.get('best_val_acc', 0)
    
    print(f"✓ Model loaded successfully!")
    print(f"  Previous epoch: {checkpoint.get('epoch', 0)}")
    print(f"  Previous best val acc: {previous_best:.2f}%")
    print(f"  Starting from epoch: {start_epoch}")
    
    # Setup training with improved hyperparameters
    print(f"\n{'='*70}")
    print("Setting up training...")
    print(f"{'='*70}\n")
    
    # Increased label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    
    # AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    print(f"✓ Training setup complete!")
    print(f"  Optimizer: AdamW (lr={args.lr:.6f}, weight_decay=0.0001)")
    print(f"  Scheduler: CosineAnnealingWarmRestarts")
    print(f"  Loss: CrossEntropyLoss (label_smoothing=0.15)")
    print(f"  Mixed Precision: Enabled (FP16)")
    print(f"  Gradient Clipping: max_norm=1.0")
    
    # Training history
    history = {'train': [], 'val': []}
    best_val_acc = previous_best
    patience_counter = 0
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'runs/continued_training_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / 'weights'
    weights_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  STARTING TRAINING")
    print(f"{'='*70}\n")
    print(f"Target: Reach 92%+ test accuracy")
    print(f"Current best: {previous_best:.2f}% validation")
    print(f"Output: {output_dir}")
    print(f"\n{'='*70}\n")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_start = time.time()
        
        print(f"Epoch {epoch}/{start_epoch + args.epochs - 1}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.8f}")
        
        # Record history
        history['train'].append({'epoch': epoch, 'loss': train_loss, 'acc': train_acc})
        history['val'].append({'epoch': epoch, 'loss': val_loss, 'acc': val_acc})
        
        # Save best model
        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'train_loss': train_loss
            }
            
            torch.save(checkpoint_dict, weights_dir / 'best.pt')
            print(f"✓ New best model saved! (Val Acc: {val_acc:.2f}%, +{improvement:.2f}%)")
            
            # Check if we reached 92%
            if val_acc >= 92.0:
                print(f"\n{'='*70}")
                print(f"  🎉 TARGET REACHED! Validation Accuracy: {val_acc:.2f}%")
                print(f"{'='*70}\n")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n{'='*70}")
            print(f"Early stopping triggered after {patience_counter} epochs without improvement")
            print(f"{'='*70}\n")
            break
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch time: {epoch_time:.1f}s")
        print()
    
    # Training complete
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*70}\n")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Improvement: +{best_val_acc - previous_best:.2f}%")
    print(f"Best model saved: {weights_dir / 'best.pt'}")
    
    # Save training history
    with open(output_dir / 'train_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  NEXT STEPS")
    print(f"{'='*70}\n")
    print(f"1. Evaluate on test set:")
    print(f"   python evaluation/evaluate_pretrained.py --weights {weights_dir / 'best.pt'}")
    print(f"\n2. If test accuracy < 92%, try:")
    print(f"   - Train with different seed")
    print(f"   - Create ensemble with multiple models")
    print(f"\n3. Check training history: {output_dir / 'train_history.json'}")
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()

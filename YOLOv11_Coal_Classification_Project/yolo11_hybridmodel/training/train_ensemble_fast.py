"""
Fast Ensemble Member Training - Fine-tune from existing checkpoint
Quick 10-15 epoch fine-tuning with different seeds/augmentation to create diversity
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import random
from datetime import datetime

from hybrid_model_pretrained import PretrainedHybridWrapper


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CoalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in class_dir.glob(ext):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(augmentation_type='moderate', img_size=224):
    """Get data transforms based on augmentation type"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    if augmentation_type == 'light':
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize
        ])
    elif augmentation_type == 'moderate':
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalize
        ])
    else:  # strong
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='Fast Ensemble Member Training')
    parser.add_argument('--resume', type=str, required=True, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--augmentation', type=str, default='moderate',
                       choices=['light', 'moderate', 'strong'])
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--train-dir', type=str, default='../data/train')
    parser.add_argument('--val-dir', type=str, default='../data/val')
    parser.add_argument('--output-name', type=str, required=True)
    parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision training')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path('runs') / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / 'weights'
    weights_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_transform, val_transform = get_transforms(args.augmentation)
    
    train_dataset = CoalDataset(args.train_dir, transform=train_transform)
    val_dataset = CoalDataset(args.val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Load model from checkpoint
    print(f"\nLoading model from {args.resume}...")
    checkpoint = torch.load(args.resume, map_location=device)
    
    model_config = {
        'num_classes': len(train_dataset.classes),
        'backbone_type': args.backbone,
        'pretrained': False,
        'dropout': 0.6
    }
    
    model_wrapper = PretrainedHybridWrapper(model_config=model_config)
    model = model_wrapper.build_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"\nModel loaded! Starting epoch: {checkpoint.get('epoch', 0) + 1}")
    print(f"Previous best val acc: {checkpoint.get('best_val_acc', 0):.2f}%")
    print(f"Augmentation: {args.augmentation}")
    print(f"Mixed Precision: {args.mixed_precision}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    # Training history
    history = {'train': [], 'val': []}
    best_val_acc = 0.0
    start_epoch = checkpoint.get('epoch', 0) + 1
    
    print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    print("="*60)
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f"\nEpoch {epoch}/{start_epoch + args.epochs - 1}")
        print("-"*60)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save history
        history['train'].append({'epoch': epoch, 'loss': train_loss, 'acc': train_acc})
        history['val'].append({'epoch': epoch, 'loss': val_loss, 'acc': val_acc})
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'seed': args.seed,
                'augmentation': args.augmentation
            }
            torch.save(checkpoint_data, weights_dir / 'best.pt')
            print(f"✓ New best model saved! Val Acc: {val_acc:.2f}%")
    
    # Save final checkpoint and history
    torch.save(checkpoint_data, weights_dir / 'last.pt')
    
    with open(output_dir / 'train_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("Fast fine-tuning completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

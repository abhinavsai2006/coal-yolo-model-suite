"""
Training Script for A2MADA-YOLO
Fast training with attention alignment and adversarial domain adaptation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import sys
from datetime import datetime

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


def get_transforms(img_size=224, is_training=True):
    """Get data transforms"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def train_epoch(model, dataloader, criterion, domain_criterion, optimizer, device, alpha):
    """Training loop for one epoch with adversarial training"""
    model.train()
    running_loss = 0.0
    running_class_loss = 0.0
    running_domain_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with adversarial training
        class_pred, domain_pred, domain_outputs = model(
            inputs, alpha=alpha, use_adversarial=True
        )
        
        # Classification loss
        class_loss = criterion(class_pred, labels)
        
        # Domain adaptation loss (source domain = 0, target = 1)
        # For single domain training, use 0.5 as target
        domain_target = torch.full_like(domain_pred, 0.5)
        domain_loss = domain_criterion(domain_pred, domain_target)
        
        # Multi-scale domain loss
        multiscale_domain_loss = 0
        for d_out in domain_outputs:
            d_target = torch.full_like(d_out, 0.5)
            multiscale_domain_loss += domain_criterion(d_out, d_target)
        multiscale_domain_loss /= len(domain_outputs)
        
        # Total loss
        total_loss = class_loss + 0.1 * domain_loss + 0.05 * multiscale_domain_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += total_loss.item()
        running_class_loss += class_loss.item()
        running_domain_loss += (domain_loss.item() + multiscale_domain_loss.item())
        
        _, predicted = class_pred.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_class_loss = running_class_loss / len(dataloader)
    epoch_domain_loss = running_domain_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_class_loss, epoch_domain_loss, epoch_acc


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
            
            # Use simple inference mode
            outputs = model(inputs, use_adversarial=False)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description='Train A2MADA-YOLO')
    parser.add_argument('--train-dir', type=str, default='../data/train')
    parser.add_argument('--val-dir', type=str, default='../data/val')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("A2MADA-YOLO Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 60)
    
    # Create datasets
    train_dataset = CoalDataset(args.train_dir, get_transforms(args.img_size, True))
    val_dataset = CoalDataset(args.val_dir, get_transforms(args.img_size, False))
    
    num_classes = len(train_dataset.classes)
    print(f"\nClasses: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True if args.device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True if args.device == 'cuda' else False
    )
    
    # Create model
    print("\nCreating A2MADA-YOLO model...")
    model = create_a2mada_model(num_classes=num_classes)
    model = model.to(args.device)
    
    # Loss functions
    criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # Training history
    history = {'train': [], 'val': []}
    best_val_acc = 0.0
    
    # Create run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path('runs') / f'a2mada_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = run_dir / 'weights'
    weights_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving results to: {run_dir}")
    print("\nStarting training...\n")
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("-" * 60)
        
        # Calculate alpha for gradient reversal (gradually increase)
        alpha = 2.0 / (1.0 + np.exp(-10 * epoch / args.epochs)) - 1.0
        
        # Train
        train_loss, class_loss, domain_loss, train_acc = train_epoch(
            model, train_loader, criterion, domain_criterion,
            optimizer, args.device, alpha
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, args.device)
        
        # Update scheduler
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f} (Class: {class_loss:.4f}, Domain: {domain_loss:.4f})")
        print(f"Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f} | Alpha: {alpha:.4f}")
        
        # Save history
        history['train'].append({
            'epoch': epoch + 1,
            'loss': train_loss,
            'class_loss': class_loss,
            'domain_loss': domain_loss,
            'acc': train_acc,
            'alpha': alpha
        })
        history['val'].append({
            'epoch': epoch + 1,
            'loss': val_loss,
            'acc': val_acc
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_acc': train_acc
            }, weights_dir / 'best.pt')
            print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Save last model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc
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

"""Train EfficientNet-B3 Hybrid Variant to add diversity and reach 92%.

This trains the same hybrid head on an EfficientNet-B3 backbone.

Recommended run (PowerShell):
  & "E:\Yolo\yolov9_gpu\python.exe" \
    "E:\Yolo\YOLOv11_Coal_Classification_Project\yolo11_hybridmodel\training\train_efficientnet_b3.py" \
    --epochs 30 --batch-size 24 --lr 1e-4 --patience 12 --seed 1337
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
import argparse
import json
import time
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.hybrid_model_pretrained import create_pretrained_hybrid_model


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_loaders(batch_size=24, workers=4):
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.15, 0.15, 0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    root = Path(__file__).parent.parent.parent / 'data'
    train_ds = datasets.ImageFolder(root / 'train', transform=train_tf)
    val_ds = datasets.ImageFolder(root / 'val', transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, len(train_ds.classes)


def train_epoch(model, loader, device, optimizer, scaler=None):
    model.train()
    crit = nn.CrossEntropyLoss(label_smoothing=0.10)
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        loss_sum += loss.item()
        total += y.size(0)
        correct += (out.argmax(1) == y).sum().item()
    return loss_sum / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        loss = crit(out, y)
        loss_sum += loss.item()
        total += y.size(0)
        correct += (out.argmax(1) == y).sum().item()
    return loss_sum / len(loader), 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser(description='Train EfficientNet-B3 hybrid classifier')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=24)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('ERROR: CUDA required')
        return 1

    print('='*70)
    print(' TRAIN EfficientNet-B3 HYBRID')
    print('='*70)
    print(f'Seed: {args.seed}  Epochs: {args.epochs}  Batch: {args.batch_size}  LR: {args.lr}')
    print(f'Device: {device}  GPU: {torch.cuda.get_device_name(0)}  CUDA: {torch.version.cuda}')

    train_loader, val_loader, num_classes = get_loaders(args.batch_size, args.workers)
    print(f'Dataset -> Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Classes: {num_classes}')

    # Build model
    wrapper = create_pretrained_hybrid_model(num_classes=num_classes, backbone_type='efficientnet_b3', pretrained=True, dropout=0.5)
    model = wrapper.build_model().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, verbose=True)
    scaler = torch.amp.GradScaler('cuda')

    # Output dirs
    out_dir = Path(__file__).parent.parent / 'runs' / f'b3_seed{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_dir / 'weights'
    weights_dir.mkdir(exist_ok=True)

    best_val = -1.0
    wait = 0
    history = {'train': [], 'val': []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, device, optimizer, scaler)
        va_loss, va_acc = evaluate(model, val_loader, device)
        scheduler.step(va_acc)
        history['train'].append({'epoch': epoch, 'loss': tr_loss, 'acc': tr_acc})
        history['val'].append({'epoch': epoch, 'loss': va_loss, 'acc': va_acc})
        print(f'Epoch {epoch}/{args.epochs}  train {tr_loss:.4f}/{tr_acc:.2f}%  val {va_loss:.4f}/{va_acc:.2f}%  lr={optimizer.param_groups[0]["lr"]:.2e}  time={time.time()-t0:.1f}s')

        if va_acc > best_val:
            best_val = va_acc
            wait = 0
            torch.save({'model_state_dict': model.state_dict(), 'val_acc': va_acc, 'epoch': epoch}, weights_dir / 'best.pt')
            print(f'  ✓ Saved best @ val_acc={va_acc:.2f}% -> {weights_dir / "best.pt"}')
        else:
            wait += 1
            print(f'  no improve ({wait}/{args.patience})')
            if wait >= args.patience:
                print('Early stopping')
                break

    with open(out_dir / 'train_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print('='*70)
    print(f'Best Val Acc: {best_val:.2f}%  Weights: {weights_dir / "best.pt"}')
    print('='*70)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

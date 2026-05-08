"""High-resolution fine-tune for the best checkpoint to push past 92%.

Strategy:
- Load best model checkpoint (seed 789 recommended)
- Increase input resolution to 320x320 (train/val)
- Freeze backbone for first few epochs; train attention + classifier
- Unfreeze last ResNet block for final epochs (optional)
- Very low LR (5e-6) with ReduceLROnPlateau on val acc
- Early stopping patience 5

Usage (PowerShell):
  & "E:\Yolo\yolov9_gpu\python.exe" \
    "E:\Yolo\YOLOv11_Coal_Classification_Project\yolo11_hybridmodel\training\fine_tune_highres.py" \
    --weights "E:\Yolo\YOLOv11_Coal_Classification_Project\yolo11_hybridmodel\runs\ensemble_seed789_20251109_035230\weights\best.pt" \
    --epochs 10 --batch-size 16 --lr 5e-6 --unfreeze-last True

"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import sys
import json
import time

sys.path.append(str(Path(__file__).parent.parent))
from models.hybrid_model_pretrained import create_pretrained_hybrid_model


def build_loaders(batch_size=16, workers=2):
    """Construct train/val loaders.

    workers: number of DataLoader worker processes. On Windows high worker counts
    can sometimes hang; allow user override. Use pin_memory for CUDA speed.
    """
    train_tf = transforms.Compose([
        transforms.Resize((352, 352)),
        transforms.RandomCrop((320, 320)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.CenterCrop((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    root = Path(__file__).parent.parent.parent / 'data'
    train_ds = datasets.ImageFolder(root / 'train', transform=train_tf)
    val_ds = datasets.ImageFolder(root / 'val', transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    print(f"[Data] Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Classes: {len(train_ds.classes)}", flush=True)
    return train_loader, val_loader, len(train_ds.classes)


def load_model_from_ckpt(ckpt_path: str, num_classes: int, device: torch.device):
    wrapper = create_pretrained_hybrid_model(num_classes=num_classes, backbone_type='resnet50', pretrained=True, dropout=0.6)
    model = wrapper.build_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    return model


def freeze_backbone(model, freeze: bool = True):
    # Freeze all except classifier and attention modules (heuristic based on naming)
    for name, p in model.named_parameters():
        if freeze:
            trainable = any(k in name for k in ['classifier', 'attention', 'fusion'])
            p.requires_grad = trainable
        else:
            p.requires_grad = True


def unfreeze_last_resnet_block(model):
    for name, p in model.named_parameters():
        if any(k in name for k in ['layer4']):
            p.requires_grad = True


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = crit(out, y)
            loss_sum += loss.item()
            pred = out.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return loss_sum / len(loader), 100.0 * correct / total


def train_epoch(model, loader, device, optimizer, scaler=None, clip=1.0, log_interval=10):
    model.train()
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_sum = 0.0
    correct = 0
    total = 0
    num_batches = len(loader)
    start = time.time()
    for batch_idx, (x, y) in enumerate(loader, 1):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        loss_sum += loss.item()
        total += y.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        if batch_idx % log_interval == 0 or batch_idx == num_batches:
            pct = 100.0 * batch_idx / num_batches
            avg_loss = loss_sum / batch_idx
            avg_acc = 100.0 * correct / total
            elapsed = time.time() - start
            print(f"  [Train] Batch {batch_idx}/{num_batches} ({pct:5.1f}%) loss={avg_loss:.4f} acc={avg_acc:.2f}% time={elapsed:.1f}s", flush=True)
    return loss_sum / num_batches, 100.0 * correct / total


def main():
    import argparse
    ap = argparse.ArgumentParser(description='High-resolution fine-tune')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=5e-6)
    ap.add_argument('--unfreeze-last', type=bool, default=True)
    ap.add_argument('--workers', type=int, default=2, help='DataLoader workers (Windows may prefer 0-2)')
    ap.add_argument('--log-interval', type=int, default=10, help='Train batch logging interval')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('ERROR: CUDA required for fine-tune')
        return 1

    print('[Setup] Building data loaders...', flush=True)
    train_loader, val_loader, num_classes = build_loaders(args.batch_size, workers=args.workers)
    model = load_model_from_ckpt(args.weights, num_classes, device)

    # Phase 1: Freeze backbone
    freeze_backbone(model, True)
    # Report parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {total_params:,} | Trainable (phase1): {trainable_params:,}", flush=True)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    scaler = torch.amp.GradScaler('cuda')

    out_dir = Path(__file__).parent.parent / 'runs' / f'highres_ft_{Path(args.weights).parent.parent.name}'
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / 'best.pt'

    best_val = -1.0
    patience = 5
    wait = 0

    print('=' * 70)
    print(' HIGH-RES FINE-TUNE (320x320)')
    print('=' * 70)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        tr_loss, tr_acc = train_epoch(model, train_loader, device, optimizer, scaler, log_interval=args.log_interval)
        va_loss, va_acc = evaluate(model, val_loader, device)
        scheduler.step(va_acc)
        print(f'[Epoch {epoch}] train_loss={tr_loss:.4f} train_acc={tr_acc:.2f}% val_loss={va_loss:.4f} val_acc={va_acc:.2f}% lr={optimizer.param_groups[0]["lr"]:.2e}', flush=True)
        if va_acc > best_val:
            best_val = va_acc
            wait = 0
            torch.save({'model_state_dict': model.state_dict(), 'val_acc': va_acc, 'epoch': epoch}, best_path)
            print(f'  ✓ Saved best: {va_acc:.2f}%', flush=True)
        else:
            wait += 1
            print(f'  no improve ({wait}/{patience})', flush=True)
        if args.unfreeze_last and epoch == max(2, args.epochs // 2):
            # Unfreeze last block for a tiny gain
            unfreeze_last_resnet_block(model)
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
            # Recompute trainable params
            trainable_params2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'  → Unfroze last resnet block. New trainable params: {trainable_params2:,}', flush=True)
        if wait >= patience:
            print('Early stopping')
            break
        print(f'  epoch time {time.time() - t0:.1f}s', flush=True)

    print('=' * 70)
    print(f'Best val acc: {best_val:.2f}% at {best_path}')
    print('=' * 70)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

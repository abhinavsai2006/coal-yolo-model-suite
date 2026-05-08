"""Predict class for one or more images using the final hybrid model.

Usage (PowerShell):
  & "E:\Yolo\yolov9_gpu\python.exe" \
    "E:\Yolo\YOLOv11_Coal_Classification_Project\yolo11_hybridmodel\inference\predict_single.py" \
    --weights "E:\Yolo\YOLOv11_Coal_Classification_Project\yolo11_hybridmodel\runs\b3_seed1337\weights\best.pt" \
    --images "E:\path\to\img1.jpg" "E:\path\to\img2.jpg" \
    --tta hflip
"""
from __future__ import annotations
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import argparse
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.hybrid_model_pretrained import create_pretrained_hybrid_model

CLASS_NAMES = [
    'destructive_coal',
    'fully_pulverized_coal',
    'non_destructive_coal',
    'not_coal',
    'pulverized_coal',
    'strongly_destructive_coal',
]


def build_transform(size=224):
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(weight_path: str, num_classes: int, device: torch.device):
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    keys = list(state_dict.keys())
    if any(k.startswith('backbone.features') for k in keys):
        backbone = 'efficientnet_b3'
    else:
        backbone = 'resnet50'
    wrapper = create_pretrained_hybrid_model(num_classes=num_classes, backbone_type=backbone, pretrained=True, dropout=0.6)
    model = wrapper.build_model().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_images(model, images, device, tta: str = 'none'):
    tf = build_transform(224)
    results = []
    with torch.no_grad():
        for img_path in images:
            img = Image.open(img_path).convert('RGB')
            x = tf(img).unsqueeze(0).to(device)
            if tta == 'hflip':
                x_flip = torch.flip(x, dims=[3])
                logits = (model(x) + model(x_flip)) / 2.0
            else:
                logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())
            results.append({
                'image': str(img_path),
                'pred_class': CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else pred_idx,
                'probs': {CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i): float(probs[i]) for i in range(len(probs))}
            })
    return results


def main():
    ap = argparse.ArgumentParser(description='Predict images with final hybrid model')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--images', nargs='+', required=True)
    ap.add_argument('--tta', choices=['none', 'hflip'], default='none')
    ap.add_argument('--save', type=str, default=None, help='Optional JSON path to save predictions')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.weights, num_classes=len(CLASS_NAMES), device=device)
    preds = predict_images(model, [Path(p) for p in args.images], device, tta=args.tta)

    print(json.dumps(preds, indent=2))
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(preds, f, indent=2)
        print(f'\nSaved predictions -> {out}')


if __name__ == '__main__':
    main()

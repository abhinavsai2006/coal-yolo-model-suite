"""
Inference Script for YOLO11 Hybrid Model
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).parent))
from hybrid_model import create_hybrid_model


class HybridPredictor:
    """Predictor for YOLO11 Hybrid Model"""
    
    def __init__(self, weights_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Create model
        model_wrapper = create_hybrid_model(num_classes=self.config['num_classes'])
        self.model = model_wrapper.model
        self.model.load_state_dict(checkpoint['model'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully on {self.device}")
    
    def predict(self, image_path, top_k=3):
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file
            top_k: Return top k predictions
        
        Returns:
            Dictionary with predictions and probabilities
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
        
        # Get top k predictions
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(probabilities)))
        
        results = {
            'predictions': [],
            'image_path': str(image_path)
        }
        
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            results['predictions'].append({
                'class_id': int(idx),
                'probability': float(prob),
                'confidence': float(prob * 100)
            })
        
        return results
    
    def predict_batch(self, image_paths, top_k=3):
        """Predict for multiple images"""
        results = []
        for img_path in image_paths:
            result = self.predict(img_path, top_k)
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description='YOLO11 Hybrid Model Inference')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image-dir', type=str, help='Path to directory of images')
    parser.add_argument('--top-k', type=int, default=3, help='Return top k predictions')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--classes', type=str, nargs='+', 
                       default=['destructive_coal', 'fully_pulverized_coal', 'non_destructive_coal', 
                               'not_coal', 'pulverized_coal', 'strongly_destructive_coal'],
                       help='Class names')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = HybridPredictor(args.weights, args.device)
    
    # Collect images
    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
    elif args.image_dir:
        img_dir = Path(args.image_dir)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(img_dir.glob(ext))
            image_paths.extend(img_dir.glob(ext.upper()))
    else:
        print("Please provide --image or --image-dir")
        return
    
    if not image_paths:
        print("No images found!")
        return
    
    print(f"\nProcessing {len(image_paths)} image(s)...\n")
    
    # Predict
    for img_path in image_paths:
        print(f"{'='*60}")
        print(f"Image: {img_path.name}")
        print(f"{'='*60}")
        
        result = predictor.predict(img_path, args.top_k)
        
        for i, pred in enumerate(result['predictions'], 1):
            class_name = args.classes[pred['class_id']] if pred['class_id'] < len(args.classes) else f"Class_{pred['class_id']}"
            print(f"{i}. {class_name:<30} - Confidence: {pred['confidence']:.2f}%")
        print()


if __name__ == "__main__":
    main()

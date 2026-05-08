from ultralytics import YOLO
import cv2
import argparse
import sys
from pathlib import Path

def predict_coal_type(model_path: str, image_path: str):
    """
    Predict coal type from an image
    
    Args:
        model_path: Path to trained YOLOv8 model
        image_path: Path to image for prediction
    """
    
    # Load model
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    # Class names
    class_names = [
        'destructive_coal',
        'fully_pulverized_coal', 
        'non_destructive_coal',
        'pulverized_coal',
        'strongly_destructive_coal'
    ]
    
    try:
        # Make prediction
        results = model(image_path)
        result = results[0]
        
        # Get prediction details
        probs = result.probs
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        
        predicted_class = class_names[top1_idx]
        
        # Display results
        print(f"\n🔮 Coal Type Prediction")
        print("="*40)
        print(f"📷 Image: {Path(image_path).name}")
        print(f"🏷️  Predicted Type: {predicted_class.replace('_', ' ').title()}")
        print(f"📊 Confidence: {top1_conf:.4f} ({top1_conf*100:.2f}%)")
        
        # Show top 3 predictions
        top5_indices = probs.top5[:3]  # Get top 3
        top5_conf = probs.top5conf[:3]
        
        print(f"\n📈 Top 3 Predictions:")
        for i, (idx, conf) in enumerate(zip(top5_indices, top5_conf)):
            class_name = class_names[idx]
            print(f"   {i+1}. {class_name.replace('_', ' ').title()}: {conf:.4f} ({conf*100:.2f}%)")
        
        return predicted_class, top1_conf
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Coal Type Classification using YOLOv8')
    parser.add_argument('--model', '-m', type=str, 
                       default='coal_classification_runs/coal_yolov8_nano/weights/best.pt',
                       help='Path to trained YOLOv8 model')
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Path to image for prediction')
    
    args = parser.parse_args()
    
    print("🔥 YOLOv8 Coal Type Classifier")
    print("="*40)
    
    result = predict_coal_type(args.model, args.image)
    
    if result:
        predicted_class, confidence = result
        print(f"\n✅ Prediction completed successfully!")
    else:
        print(f"\n❌ Prediction failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
YOLOv11 Prediction Script
Make predictions on individual images using the trained model.
"""

from ultralytics import YOLO
import os
import random
from pathlib import Path

def predict_single_image(model_path, image_path):
    """Predict class for a single image"""
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    # Load model
    model = YOLO(model_path)
    
    # Make prediction
    results = model(image_path)
    
    # Get prediction details
    for result in results:
        probs = result.probs
        if probs is not None:
            top1_idx = probs.top1
            top1_conf = probs.top1conf.item()
            class_names = ['destructive_coal', 'fully_pulverized_coal', 
                          'non_destructive_coal', 'not_coal', 'pulverized_coal', 'strongly_destructive_coal']
            
            predicted_class = class_names[top1_idx] if top1_idx < len(class_names) else f"class_{top1_idx}"
            
            print(f"Image: {os.path.basename(image_path)}")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {top1_conf:.4f}")
            
            return predicted_class, top1_conf
    
    return None, None

def predict_random_samples(model_path='runs/classify/coal_classification/weights/best.pt', 
                          num_samples=5):
    """Predict on random samples from test data"""
    
    test_dir = Path('data/test')
    if not test_dir.exists():
        print("Test directory not found!")
        return
    
    # Get all image files from test directory
    image_files = []
    for class_dir in test_dir.iterdir():
        if class_dir.is_dir():
            for img_file in class_dir.glob('*.jpg'):
                image_files.append(img_file)
            for img_file in class_dir.glob('*.jpeg'):
                image_files.append(img_file)
            for img_file in class_dir.glob('*.png'):
                image_files.append(img_file)
    
    if not image_files:
        print("No image files found in test directory!")
        return
    
    # Select random samples
    random_images = random.sample(image_files, min(num_samples, len(image_files)))
    
    print(f"\nPredicting on {len(random_images)} random test images:")
    print("="*60)
    
    for img_path in random_images:
        true_class = img_path.parent.name
        predicted_class, confidence = predict_single_image(model_path, str(img_path))
        
        if predicted_class:
            correct = "✓" if predicted_class == true_class else "✗"
            print(f"{correct} True: {true_class} | Predicted: {predicted_class} | Conf: {confidence:.3f}")
        print("-" * 60)

if __name__ == "__main__":
    model_path = 'runs/classify/coal_classification/weights/best.pt'
    
    if os.path.exists(model_path):
        predict_random_samples(model_path, num_samples=10)
    else:
        print("Model not found! Please train the model first using train_model.py")
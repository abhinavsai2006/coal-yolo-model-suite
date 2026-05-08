"""
YOLOv11 Coal Classification Training Script with Regularization
Enhanced version to prevent overfitting with data augmentation and regularization.
"""

from ultralytics import YOLO
import torch
import os

def train_regularized_model():
    """Train YOLOv11 model with regularization techniques to prevent overfitting"""
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load YOLOv11 nano model
    model = YOLO('yolo11n-cls.pt')
    
    print("Starting YOLOv11 Coal Classification Training with Regularization...")
    print("Dataset: 6-class coal classification")
    print("Classes: destructive_coal, fully_pulverized_coal, non_destructive_coal, not_coal, pulverized_coal, strongly_destructive_coal")
    
    # Training parameters with regularization
    results = model.train(
        data='data',
        epochs=50,           # Reduced epochs to prevent overfitting
        imgsz=224,
        batch=16,            # Smaller batch size for better generalization
        device=device,
        patience=10,         # Earlier stopping
        save=True,
        plots=True,
        
        # Data Augmentation (Anti-overfitting)
        hsv_h=0.015,         # Hue augmentation
        hsv_s=0.7,           # Saturation augmentation  
        hsv_v=0.4,           # Brightness augmentation
        degrees=10,          # Rotation augmentation
        translate=0.1,       # Translation augmentation
        scale=0.2,           # Scale augmentation
        shear=5,             # Shear augmentation
        perspective=0.001,   # Perspective augmentation
        flipud=0.5,          # Vertical flip
        fliplr=0.5,          # Horizontal flip
        mosaic=0.5,          # Mosaic augmentation
        mixup=0.1,           # Mixup augmentation
        copy_paste=0.1,      # Copy-paste augmentation
        
        # Regularization
        dropout=0.2,         # Dropout rate
        weight_decay=0.0005, # L2 regularization
        
        # Learning rate scheduling
        lr0=0.001,           # Initial learning rate (lower)
        lrf=0.01,            # Final learning rate factor
        momentum=0.937,      # SGD momentum
        
        # Validation
        val=True,
        fraction=1.0,        # Use full dataset
        
        # Project settings
        project='runs/classify',
        name='coal_classification_regularized',
        exist_ok=True
    )
    
    print("\nTraining completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    
    # Validate the trained model
    print("\nValidating trained model...")
    best_model = YOLO(f"{results.save_dir}/weights/best.pt")
    val_results = best_model.val(data='data')
    
    print("\n=== Validation Results ===")
    print(f"Top-1 Accuracy: {val_results.top1:.4f}")
    print(f"Top-5 Accuracy: {val_results.top5:.4f}")
    
    return results

if __name__ == "__main__":
    train_regularized_model()
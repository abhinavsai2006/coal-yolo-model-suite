"""
YOLOv11 Coal Classification Training Script
This script trains a YOLOv11 classification model on coal image data.
"""

from ultralytics import YOLO
import os

def train_yolov11_classifier():
    """Train YOLOv11 classification model for coal types"""
    
    # Initialize YOLOv11 classification model
    model = YOLO('yolo11n-cls.pt')  # Start with nano model for faster training
    
    # Train the model
    results = model.train(
        data='data',               # path to dataset directory
        epochs=100,                # number of training epochs
        imgsz=224,                 # image size
        batch=32,                  # batch size (increased for GPU)
        lr0=0.01,                  # initial learning rate
        patience=15,               # early stopping patience
        save=True,                 # save model checkpoints
        plots=True,                # save training plots
        val=True,                  # validate during training
        project='runs/classify',   # project directory
        name='coal_classification_6class', # experiment name
        exist_ok=True,             # overwrite existing experiment
        device='cuda:0'            # use GPU for training
    )
    
    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    
    return results

def validate_model(model_path='runs/classify/coal_classification/weights/best.pt'):
    """Validate the trained model"""
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Validate the model
    metrics = model.val(data='data')
    
    print("\n=== Validation Results ===")
    print(f"Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"Top-5 Accuracy: {metrics.top5:.4f}")
    
    return metrics

if __name__ == "__main__":
    print("Starting YOLOv11 Coal Classification Training...")
    print("="*50)
    
    # Train the model
    train_results = train_yolov11_classifier()
    
    # Validate the trained model
    print("\nValidating trained model...")
    val_results = validate_model()
    
    print("\nTraining and validation completed!")
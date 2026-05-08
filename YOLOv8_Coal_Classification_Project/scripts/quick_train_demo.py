from ultralytics import YOLO
import torch
from pathlib import Path

def quick_train_demo():
    """Quick training demo with fewer epochs"""
    
    print("🔥 YOLOv8 Coal Classification - Quick Training Demo")
    print("="*60)
    
    # Check dataset
    dataset_path = "e:/Yolo/coal_classification_dataset"
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    print(f"📂 Dataset: {dataset_path}")
    print(f"🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Load model
    model = YOLO('yolov8n-cls.pt')
    print(f"🤖 Model: YOLOv8n Classification ({sum(p.numel() for p in model.model.parameters()):,} parameters)")
    
    # Training with reduced parameters for demo
    print(f"\n🚀 Starting Quick Training (20 epochs)...")
    print("="*60)
    
    results = model.train(
        data=dataset_path,
        epochs=20,           # Reduced for demo
        batch=8,             # Smaller batch size
        imgsz=224,
        project="coal_quick_demo",
        name="demo_run",
        patience=10,         # Early stopping
        save_period=5,       # Save every 5 epochs
        device='cpu',        # Force CPU for compatibility
        workers=2,           # Fewer workers
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        mixup=0.1,           # Reduced augmentation
        degrees=5,
        translate=0.05,
        scale=0.95,
        fliplr=0.3,
        verbose=True
    )
    
    print(f"\n✅ Quick training completed!")
    
    # Quick validation
    print(f"\n🔍 Validating model...")
    val_results = model.val()
    
    print(f"\n📊 Results:")
    print(f"   Top-1 Accuracy: {val_results.top1:.4f} ({val_results.top1*100:.2f}%)")
    print(f"   Top-5 Accuracy: {val_results.top5:.4f} ({val_results.top5*100:.2f}%)")
    
    # Test prediction
    train_path = Path(dataset_path) / "train"
    classes = [d.name for d in train_path.iterdir() if d.is_dir()]
    sample_image = list((train_path / classes[0]).glob("*.jpg"))[0]
    
    print(f"\n🔮 Testing prediction on: {sample_image.name}")
    pred_results = model(str(sample_image))
    probs = pred_results[0].probs
    
    predicted_class_idx = probs.top1
    confidence = probs.top1conf.item()
    
    print(f"   Predicted class: {classes[predicted_class_idx] if predicted_class_idx < len(classes) else 'Unknown'}")
    print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    model_path = "coal_quick_demo/demo_run/weights/best.pt"
    print(f"\n💾 Best model saved to: {model_path}")
    
    return model_path

if __name__ == "__main__":
    try:
        model_path = quick_train_demo()
        print(f"\n🎉 Demo completed successfully!")
        print(f"📁 Check results in: coal_quick_demo/demo_run/")
    except KeyboardInterrupt:
        print(f"\n⏸️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
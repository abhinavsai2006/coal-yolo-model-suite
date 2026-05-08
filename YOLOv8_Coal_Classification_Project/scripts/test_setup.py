from ultralytics import YOLO
import torch
from pathlib import Path

def test_yolo_setup():
    """Test YOLOv8 setup and dataset"""
    
    print("🔥 Testing YOLOv8 Coal Classification Setup")
    print("="*50)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Check dataset
    dataset_path = Path("e:/Yolo/coal_classification_dataset")
    if dataset_path.exists():
        print(f"✅ Dataset found: {dataset_path}")
        
        train_path = dataset_path / "train"
        val_path = dataset_path / "val"
        
        # Count classes and images
        classes = [d.name for d in train_path.iterdir() if d.is_dir()]
        print(f"📊 Classes: {len(classes)}")
        
        total_train = 0
        total_val = 0
        
        for class_name in classes:
            train_count = len(list((train_path / class_name).glob("*.jpg")))
            val_count = len(list((val_path / class_name).glob("*.jpg")))
            total_train += train_count
            total_val += val_count
            print(f"   {class_name}: {train_count} train, {val_count} val")
        
        print(f"📈 Total: {total_train} train, {total_val} val images")
        
    else:
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    # Test YOLOv8 model loading
    try:
        print(f"\n🤖 Testing YOLOv8 model loading...")
        model = YOLO('yolov8n-cls.pt')  # This will download if not present
        print(f"✅ YOLOv8 model loaded successfully")
        print(f"   Model type: Classification")
        print(f"   Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ Error loading YOLOv8: {e}")
        return False
    
    # Test prediction on a sample image
    try:
        sample_images = list((train_path / classes[0]).glob("*.jpg"))
        if sample_images:
            sample_image = sample_images[0]
            print(f"\n🔮 Testing prediction on: {sample_image.name}")
            results = model(str(sample_image))
            print(f"✅ Prediction successful!")
            
            # Show prediction (before training, it will be random)
            probs = results[0].probs
            top1_idx = probs.top1
            top1_conf = probs.top1conf.item()
            print(f"   Top prediction: Class {top1_idx} ({top1_conf:.4f})")
            
    except Exception as e:
        print(f"❌ Error during prediction test: {e}")
        return False
    
    print(f"\n✅ All tests passed! Ready for training!")
    return True

if __name__ == "__main__":
    success = test_yolo_setup()
    if success:
        print(f"\n🚀 You can now run: python train_coal_classifier.py")
    else:
        print(f"\n❌ Please fix the issues above before training.")
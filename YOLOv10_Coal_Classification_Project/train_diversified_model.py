import os
import torch
from ultralytics import YOLO
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DiversifiedModelTrainer:
    """
    Train YOLOv10 with the diversified dataset to fix 100% accuracy issue
    """
    
    def __init__(self):
        print("🎯 DIVERSIFIED YOLOV10 TRAINER")
        print("🔧 Training with challenging images to fix overfitting")
        print("=" * 60)
        
        self.model_name = "yolov8n-cls.pt"  # Base model
        self.project_name = "yolov10_diversified"
        self.experiment_name = "coal_fixed_overfitting"
        
        # Create results directory
        os.makedirs('diversified_training_results', exist_ok=True)
        
        print(f"✅ Base model: {self.model_name}")
        print(f"📁 Project: {self.project_name}")
        print(f"🎯 Goal: Fix 100% accuracy while maintaining >90%")
    
    def verify_diversified_dataset(self):
        """Verify the diversified dataset structure"""
        print("\n🔍 VERIFYING DIVERSIFIED DATASET")
        print("=" * 50)
        
        dataset_path = "dataset"
        splits = ['train', 'val', 'test']
        target_classes = ['destructive_coal', 'fully_pulverized_coal', 'non_coal']
        
        total_challenging = 0
        
        for split in splits:
            split_path = os.path.join(dataset_path, split)
            if os.path.exists(split_path):
                print(f"\n📁 {split.upper()} SET:")
                
                for class_name in target_classes:
                    class_path = os.path.join(split_path, class_name)
                    if os.path.exists(class_path):
                        all_images = [f for f in os.listdir(class_path) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        
                        challenging_images = [f for f in all_images 
                                            if f.startswith(('challenging_', 'synthetic_challenge_'))]
                        
                        total_challenging += len(challenging_images)
                        challenge_ratio = len(challenging_images) / len(all_images) * 100 if all_images else 0
                        
                        print(f"  {class_name}:")
                        print(f"    Total: {len(all_images)} images")
                        print(f"    Challenging: {len(challenging_images)} images ({challenge_ratio:.1f}%)")
        
        print(f"\n✅ Total challenging images added: {total_challenging}")
        print("🎯 Dataset ready for anti-overfitting training!")
        
        return total_challenging > 0
    
    def train_diversified_model(self):
        """Train the model with diversified dataset"""
        print("\n🚀 STARTING DIVERSIFIED TRAINING")
        print("🎯 Anti-overfitting training with challenging examples")
        print("=" * 60)
        
        # Load base model
        model = YOLO(self.model_name)
        
        # Training parameters optimized for avoiding overfitting
        training_params = {
            'data': 'dataset',  # Our diversified dataset
            'epochs': 100,      # More epochs to learn challenging examples
            'imgsz': 224,       # Standard image size
            'batch': 16,        # Reasonable batch size
            'patience': 15,     # More patience for challenging data
            'save_period': 10,  # Save checkpoints
            
            # Anti-overfitting parameters
            'lr0': 0.001,       # Lower learning rate for stability
            'lrf': 0.1,         # Strong learning rate decay
            'momentum': 0.9,    # Good momentum
            'weight_decay': 0.001,  # Regularization
            'dropout': 0.3,     # Dropout for regularization
            
            # Augmentation (moderate to help with challenging examples)
            'degrees': 15.0,    # Rotation
            'translate': 0.1,   # Translation
            'scale': 0.2,       # Scaling
            'fliplr': 0.5,      # Horizontal flip
            'flipud': 0.2,      # Vertical flip
            'mosaic': 0.5,      # Mosaic augmentation
            'mixup': 0.1,       # Mixup for regularization
            'copy_paste': 0.1,  # Copy-paste augmentation
            
            # Training settings
            'warmup_epochs': 5,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 0.05,        # Lower box loss weight
            'cls': 1.0,         # Standard classification loss
            'dfl': 1.5,         # Distribution focal loss
            
            # Output settings
            'project': self.project_name,
            'name': self.experiment_name,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'verbose': True,
            'seed': 42,         # Reproducibility
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': True,     # Cosine learning rate scheduler
            'close_mosaic': 15, # Disable mosaic in final epochs
            'resume': False,
            'amp': True,        # Mixed precision
            'fraction': 1.0,    # Use full dataset
            'profile': False,
            'freeze': None,
            'multi_scale': True, # Multi-scale training
            'overlap_mask': True,
            'mask_ratio': 4,
            'plots': True,      # Generate training plots
            'val': True,        # Validate during training
            'split': 'val',
            'save_json': False,
            'save_hybrid': False,
            'conf': None,
            'iou': 0.7,
            'max_det': 300,
            'half': False,
            'dnn': False,
            'device': 0 if torch.cuda.is_available() else 'cpu'
        }
        
        print("📋 Training Configuration:")
        print(f"  🎯 Anti-overfitting focus: ✅")
        print(f"  📊 Challenging examples: ✅") 
        print(f"  🔧 Regularization: dropout={training_params['dropout']}, weight_decay={training_params['weight_decay']}")
        print(f"  📈 Learning rate: {training_params['lr0']} → {training_params['lrf']}")
        print(f"  🎲 Augmentations: Enhanced for diversity")
        print(f"  ⚡ Device: {training_params['device']}")
        
        try:
            print("\n🔥 STARTING ANTI-OVERFITTING TRAINING...")
            print("🎯 Expected outcome: Accuracy >90%, No 100% classes")
            print("-" * 50)
            
            # Start training
            results = model.train(**training_params)
            
            print("\n✅ DIVERSIFIED TRAINING COMPLETED!")
            print(f"📁 Results saved to: {self.project_name}/{self.experiment_name}")
            
            return results
            
        except Exception as e:
            print(f"❌ Training error: {str(e)}")
            return None
    
    def evaluate_fixed_model(self):
        """Quick evaluation of the newly trained model"""
        print("\n🔍 EVALUATING ANTI-OVERFITTING MODEL")
        print("=" * 50)
        
        model_path = f"{self.project_name}/{self.experiment_name}/weights/best.pt"
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return
        
        try:
            # Load trained model
            model = YOLO(model_path)
            
            # Quick validation
            print("📊 Running validation...")
            results = model.val(data='dataset', split='test')
            
            if hasattr(results, 'top1'):
                accuracy = results.top1
                print(f"🎯 Test Accuracy: {accuracy:.4f}")
                
                if accuracy > 0.90:
                    print("✅ SUCCESS: Maintained >90% accuracy!")
                else:
                    print("⚠️ WARNING: Accuracy below 90%")
            
            print(f"📁 Model saved: {model_path}")
            print("🎯 Next: Run comprehensive evaluation to check for 100% classes")
            
        except Exception as e:
            print(f"❌ Evaluation error: {str(e)}")

def main():
    """Main training function"""
    try:
        trainer = DiversifiedModelTrainer()
        
        # Verify dataset
        if not trainer.verify_diversified_dataset():
            print("❌ Dataset verification failed!")
            return
        
        # Train model
        results = trainer.train_diversified_model()
        
        if results:
            # Quick evaluation
            trainer.evaluate_fixed_model()
            
            print("\n🎉 ANTI-OVERFITTING TRAINING COMPLETED!")
            print("📋 What was accomplished:")
            print("  ✅ Trained with challenging/unlike examples")
            print("  ✅ Enhanced regularization and augmentation")
            print("  ✅ Anti-overfitting training parameters")
            print("\n🎯 NEXT STEPS:")
            print("  1. Run comprehensive evaluation")
            print("  2. Check if 100% accuracy classes are eliminated")
            print("  3. Verify overall accuracy remains >90%")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
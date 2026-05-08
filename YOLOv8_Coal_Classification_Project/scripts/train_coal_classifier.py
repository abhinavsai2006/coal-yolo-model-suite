from ultralytics import YOLO
import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

# Common image extensions used across scripts
VALID_EXTS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']

def train_coal_classification_model(
    dataset_path: str = "data",
    model_size: str = "yolov8m-cls.pt",  # medium model for higher accuracy
    epochs: int = 20,
    batch_size: int = 16,
    img_size: int = 224,
    project_name: str = "coal_classification_runs",
    experiment_name: str = "coal_yolov8_exp1"
):
    """
    Train YOLOv8 classification model for coal type detection
    
    Args:
        dataset_path: Path to the organized dataset
        model_size: YOLOv8 model size (yolov8n-cls.pt, yolov8s-cls.pt, yolov8m-cls.pt, yolov8l-cls.pt, yolov8x-cls.pt)
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        project_name: Name of the project directory
        experiment_name: Name of this experiment
    """
    
    print("🔥 Starting YOLOv8 Coal Classification Training")
    print("="*60)
    
    # Resolve dataset path: allow either a folder or a dataset YAML file
    dataset_path = Path(dataset_path)

    if dataset_path.suffix.lower() in ('.yml', '.yaml'):
        # Load YAML and resolve train/val entries relative to the YAML file
        with open(dataset_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # `train`/`val` in YAML may be relative paths; resolve relative to YAML parent
        yaml_dir = dataset_path.parent
        train_entry = cfg.get('train')
        val_entry = cfg.get('val')
        if train_entry is None or val_entry is None:
            raise ValueError(f"Dataset YAML {dataset_path} must contain 'train' and 'val' entries")

        train_path = (yaml_dir / train_entry).resolve()
        val_path = (yaml_dir / val_entry).resolve()
    else:
            results_dir = Path(project_name) / experiment_name
            weights_dir = results_dir / "weights"

            # Primary expected paths
            best_model_path = weights_dir / "best.pt"
            last_model_path = weights_dir / "last.pt"

            # If training saved to a sibling experiment folder (e.g. ..._balanced2), try to locate the most recent matching folder
            if not best_model_path.exists():
                base = Path(project_name)
                candidates = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(experiment_name)]
                if candidates:
                    # pick the most recently modified candidate
                    candidates_sorted = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
                    for cand in candidates_sorted:
                        cand_weights = cand / "weights"
                        cand_best = cand_weights / "best.pt"
                        cand_last = cand_weights / "last.pt"
                        if cand_best.exists():
                            best_model_path = cand_best
                            last_model_path = cand_last if cand_last.exists() else best_model_path
                            results_dir = cand
                            weights_dir = cand_weights
                            print(f"ℹ️ Found best model in alternate results dir: {cand}")
                            break
                    else:
                        # fallback: pick newest .pt in any candidate weights folder
                        for cand in candidates_sorted:
                            cand_weights = cand / "weights"
                            if cand_weights.exists():
                                pts = sorted(list(cand_weights.glob('*.pt')), key=lambda p: p.stat().st_mtime, reverse=True)
                                if pts:
                                    best_model_path = pts[0]
                                    last_model_path = pts[0]
                                    results_dir = cand
                                    weights_dir = cand_weights
                                    print(f"ℹ️ Falling back to newest checkpoint: {best_model_path}")
                                    break
    if not val_path.exists():
        raise FileNotFoundError(f"Val path not found: {val_path}")

    classes = [d.name for d in train_path.iterdir() if d.is_dir()]
    print(f"\n📊 Classes Found: {len(classes)}")
    for i, class_name in enumerate(classes):
        # Count images with common extensions
        train_count = 0
        val_count = 0
        for ext in VALID_EXTS:
            train_count += len(list((train_path / class_name).glob(ext)))
            val_count += len(list((val_path / class_name).glob(ext)))
        print(f"  {i}: {class_name} - Train: {train_count}, Val: {val_count}")
    if 'not_coal' not in classes:
        print("⚠️  'not_coal' class missing in training data! Please add images for all 6 classes.")
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Training Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    
    # Load YOLOv8 model
    print(f"\n🤖 Loading YOLOv8 Model: {model_size}")
    model = YOLO(model_size)
    
    # Display model info
    print(f"   Model Type: Classification")
    print(f"   Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    
    # Training configuration
    # For Ultralytics classification trainer, `data` must be a directory (not a YAML file).
    # If user supplied a dataset YAML, point `data` at the resolved train directory (which contains class subfolders).
    data_arg = str(train_path) if dataset_path.suffix.lower() in ('.yml', '.yaml') else str(dataset_path)

    training_args = {
        'data': data_arg,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'project': project_name,
        'name': experiment_name,
        'patience': 20,  # Early stopping patience
        'save_period': 10,  # Save checkpoint every 10 epochs
        'device': device,
        'workers': 4,  # Number of data loading workers
        'optimizer': 'AdamW',
        'lr0': 0.001,  # Initial learning rate
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'cos_lr': True,  # Cosine learning rate scheduler
        # Stronger augmentations for underperforming classes
        'mixup': 0.4,  # Increased mixup
        'copy_paste': 0.5,  # Increased copy-paste
        'degrees': 20,  # More rotation
        'translate': 0.2,  # More translation
        'scale': 0.8,  # More aggressive scaling
        'fliplr': 0.7,  # More horizontal flip
        'mosaic': 0.7,  # More mosaic
        'hsv_h': 0.03,  # More hue augmentation
        'hsv_s': 0.8,   # More saturation
        'hsv_v': 0.5,   # More value
    }
    
    print(f"\n⚙️  Training Configuration:")
    for key, value in training_args.items():
        print(f"   {key}: {value}")
    
    print(f"\n🚀 Starting Training...")
    print("="*60)
    
    # Start training
    try:
        results = model.train(**training_args)
        
        print("\n✅ Training Completed Successfully!")
        print("="*60)
        
        # Results summary
        project_dir = Path(project_name)

        # Find the experiment directory (handle naming collisions like experiment, experiment2, ...)
        candidate_dirs = []
        if project_dir.exists():
            for d in project_dir.iterdir():
                if d.is_dir() and d.name.startswith(experiment_name):
                    candidate_dirs.append(d)

        if not candidate_dirs:
            results_dir = Path(project_name) / experiment_name
        else:
            # Pick the most recently modified matching experiment dir
            results_dir = max(candidate_dirs, key=lambda p: p.stat().st_mtime)

        weights_dir = results_dir / "weights"

        # Prefer best.pt, fall back to last.pt, else pick newest .pt
        best_model_path = weights_dir / "best.pt"
        last_model_path = weights_dir / "last.pt"

        if not best_model_path.exists():
            if last_model_path.exists():
                best_model_path = last_model_path
            else:
                # pick any .pt in weights dir
                pts = sorted(weights_dir.glob('*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
                if pts:
                    best_model_path = pts[0]
                else:
                    best_model_path = None

        print(f"📈 Training Results:")
        print(f"   Best Model (resolved): {best_model_path}")
        print(f"   Last Model (resolved): {last_model_path}")
        print(f"   Results Directory: {results_dir}")

        # Validate the best model if available
        if best_model_path is not None and best_model_path.exists():
            try:
                print(f"\n🔍 Validating Best Model... ({best_model_path})")
                best_model = YOLO(str(best_model_path))
                validation_results = best_model.val()

                print(f"✅ Validation Complete!")
                print(f"   Top-1 Accuracy: {validation_results.top1:.4f}")
                print(f"   Top-5 Accuracy: {validation_results.top5:.4f}")
            except Exception as e:
                print(f"⚠️  Validation failed: {e}")
        else:
            print(f"⚠️  No weights found in {weights_dir} to validate.")

        return best_model_path, results
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise e

def create_training_config():
    """Create a YAML configuration file for training"""
    config = {
        'dataset_info': {
            'task': 'classification',
            'classes': [
                'destructive_coal',
                'fully_pulverized_coal', 
                'non_destructive_coal',
                'pulverized_coal',
                'strongly_destructive_coal',
                'not_coal'
            ],
            'total_images': 2062,
            'description': 'Coal type classification dataset with 6 classes'
        },
        'training_params': {
            'epochs': 20,
            'batch_size': 16,
            'img_size': 224,
            'model': 'yolov8m-cls.pt'
        }
    }
    
    config_path = "e:/Yolo/training_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"📝 Training configuration saved to: {config_path}")
    return config_path

if __name__ == "__main__":
    # Create configuration file
    create_training_config()
    
    print("🎯 YOLOv8 Coal Classification Training Script")
    print("="*60)
    
    # Start training with different model sizes (you can choose one)
    models_to_try = [
        ("yolov8m-cls.pt", "medium"),   # Higher accuracy
        ("yolov8l-cls.pt", "large"),    # Even higher accuracy
    ]
    
    for model_file, model_name in models_to_try:
        try:
            print(f"\n🔥 Training {model_name.upper()} model...")
            best_model, results = train_coal_classification_model(
                model_size=model_file,
                experiment_name=f"coal_yolov8_{model_name}",
                epochs=20,
                batch_size=16
            )
            print(f"✅ {model_name.upper()} training completed!")
            
        except Exception as e:
            print(f"❌ {model_name.upper()} training failed: {e}")
            continue
    
    print("\n🎉 All training sessions completed!")
    print("Check the 'coal_classification_runs' directory for results.")
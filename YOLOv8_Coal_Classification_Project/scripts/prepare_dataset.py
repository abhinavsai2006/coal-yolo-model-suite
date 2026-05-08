import os
import shutil
import random
from pathlib import Path
from typing import Dict, List

def prepare_yolo_classification_dataset(
    source_dir: str = "e:/Yolo",
    target_dir: str = "e:/Yolo/coal_classification_dataset",
    train_ratio: float = 0.8,
    val_ratio: float = 0.2
):
    """
    Organize coal images into YOLOv8 classification format
    
    Args:
        source_dir: Directory containing the original class folders
        target_dir: Directory where organized dataset will be created
        train_ratio: Ratio of images for training
        val_ratio: Ratio of images for validation
    """
    
    # Define class mappings (clean up folder names for class labels)
    class_folders = {
        "Destructive Coal": "destructive_coal",
        "Fully pulverized coal": "fully_pulverized_coal", 
        "Non destructive Coal": "non_destructive_coal",
        "Pulverized coal": "pulverized_coal",
        "Strongly destructive coal": "strongly_destructive_coal"
    }
    
    # Create target directory structure
    dataset_path = Path(target_dir)
    train_path = dataset_path / "train"
    val_path = dataset_path / "val"
    
    # Clean up existing dataset if it exists
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    
    # Create directories
    for class_name in class_folders.values():
        (train_path / class_name).mkdir(parents=True, exist_ok=True)
        (val_path / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    dataset_stats = {}
    total_images = 0
    
    for original_folder, clean_class_name in class_folders.items():
        source_class_path = Path(source_dir) / original_folder
        
        # Find all images in the class folder (looking in subdirectories)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(source_class_path.rglob(f"*{ext}")))
            image_files.extend(list(source_class_path.rglob(f"*{ext.upper()}")))
        
        if not image_files:
            print(f"Warning: No images found in {source_class_path}")
            continue
            
        # Shuffle images for random split
        random.shuffle(image_files)
        
        # Calculate split indices
        total_class_images = len(image_files)
        train_count = int(total_class_images * train_ratio)
        
        # Split images
        train_images = image_files[:train_count]
        val_images = image_files[train_count:]
        
        # Copy images to appropriate directories
        for img_path in train_images:
            dest_path = train_path / clean_class_name / img_path.name
            shutil.copy2(img_path, dest_path)
            
        for img_path in val_images:
            dest_path = val_path / clean_class_name / img_path.name
            shutil.copy2(img_path, dest_path)
        
        # Update statistics
        dataset_stats[clean_class_name] = {
            'total': total_class_images,
            'train': len(train_images),
            'val': len(val_images)
        }
        total_images += total_class_images
        
        print(f"✓ {original_folder}:")
        print(f"  Total: {total_class_images} images")
        print(f"  Train: {len(train_images)} images")
        print(f"  Val: {len(val_images)} images")
    
    # Create class names file for reference
    classes_file = dataset_path / "classes.txt"
    with open(classes_file, 'w') as f:
        for i, class_name in enumerate(class_folders.values()):
            f.write(f"{i}: {class_name}\n")
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET ORGANIZATION COMPLETE")
    print("="*50)
    print(f"Total images processed: {total_images}")
    print(f"Dataset saved to: {target_dir}")
    print("\nClass distribution:")
    
    for class_name, stats in dataset_stats.items():
        print(f"  {class_name}: {stats['train']} train, {stats['val']} val ({stats['total']} total)")
    
    return dataset_path, dataset_stats

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Preparing YOLOv8 Classification Dataset...")
    print("="*50)
    
    dataset_path, stats = prepare_yolo_classification_dataset()
    
    print(f"\nDataset ready for YOLOv8 training!")
    print(f"Use this path for training: {dataset_path}")
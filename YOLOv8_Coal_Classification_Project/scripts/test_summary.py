#!/usr/bin/env python3
"""
Test Summary Script for YOLOv8 Coal Classification Model
Tests multiple images and provides performance statistics
"""

import os
import glob
from ultralytics import YOLO
from pathlib import Path

def test_model_performance():
    """Test the model on validation set and generate statistics"""
    
    # Load the trained model
    model_path = "coal_classification_runs/coal_yolov8_nano/weights/best.pt"
    model = YOLO(model_path)
    
    print("🔬 YOLOv8 Coal Classification Model Test Summary")
    print("=" * 60)
    
    # Test configuration
    val_dir = "coal_classification_dataset/val"
    
    # Class mapping
    class_names = {
        0: "destructive_coal",
        1: "fully_pulverized_coal", 
        2: "non_destructive_coal",
        3: "pulverized_coal",
        4: "strongly_destructive_coal"
    }
    
    # Initialize counters
    total_images = 0
    correct_predictions = 0
    class_stats = {}
    
    # Test each class
    for class_name in class_names.values():
        class_dir = os.path.join(val_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        print(f"\n📂 Testing {class_name.replace('_', ' ').title()}...")
        
        # Get all images in this class
        image_files = glob.glob(os.path.join(class_dir, "*.jpg"))
        class_total = len(image_files)
        class_correct = 0
        
        # Test first 10 images from each class for quick demo
        test_images = image_files[:10]
        
        for img_path in test_images:
            try:
                # Run prediction
                results = model(img_path, verbose=False)
                
                # Get prediction
                if results and len(results) > 0:
                    # Get the class with highest confidence
                    probs = results[0].probs
                    if probs is not None:
                        predicted_class_id = probs.top1
                        predicted_class_name = class_names[predicted_class_id]
                        confidence = probs.top1conf.item()
                        
                        # Check if prediction is correct
                        if predicted_class_name == class_name:
                            class_correct += 1
                            status = "✅"
                        else:
                            status = "❌"
                        
                        print(f"   {status} {Path(img_path).name}: {predicted_class_name.replace('_', ' ').title()} ({confidence:.1%})")
                    
                total_images += 1
                        
            except Exception as e:
                print(f"   ❌ Error processing {Path(img_path).name}: {str(e)}")
        
        # Store class statistics
        class_accuracy = (class_correct / len(test_images)) * 100 if test_images else 0
        class_stats[class_name] = {
            'total': len(test_images),
            'correct': class_correct,
            'accuracy': class_accuracy
        }
        
        correct_predictions += class_correct
        
        print(f"   📊 Class Accuracy: {class_correct}/{len(test_images)} ({class_accuracy:.1f}%)")
    
    # Overall statistics
    overall_accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
    
    print(f"\n🎯 OVERALL TEST RESULTS")
    print("=" * 60)
    print(f"📊 Total Images Tested: {total_images}")
    print(f"✅ Correct Predictions: {correct_predictions}")
    print(f"❌ Incorrect Predictions: {total_images - correct_predictions}")
    print(f"🏆 Overall Accuracy: {overall_accuracy:.2f}%")
    
    print(f"\n📈 PER-CLASS ACCURACY")
    print("=" * 60)
    for class_name, stats in class_stats.items():
        print(f"{class_name.replace('_', ' ').title():<25}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})")
    
    # Model information
    print(f"\n🔧 MODEL INFORMATION")
    print("=" * 60)
    print(f"Model: YOLOv8 Nano")
    print(f"Model Path: {model_path}")
    print(f"Classes: 5 coal types")
    print(f"Input Size: 224x224 pixels")
    
    print(f"\n✨ Test completed successfully!")

if __name__ == "__main__":
    test_model_performance()
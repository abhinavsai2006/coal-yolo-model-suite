"""
Comprehensive Model Evaluation with All Metrics
Calculates accuracy, precision, recall, F1-score, confusion matrix, and additional metrics.
"""

from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_curve,
    roc_curve, auc, cohen_kappa_score, matthews_corrcoef
)
from sklearn.preprocessing import label_binarize
import os
import pandas as pd

def comprehensive_model_evaluation(model_path='runs/classify/coal_classification_regularized/weights/best.pt'):
    """Comprehensive evaluation with all possible metrics"""
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None
    
    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION - ALL METRICS")
    print("="*80)
    
    # Load model
    model = YOLO(model_path)
    
    # Class names
    class_names = ['destructive_coal', 'fully_pulverized_coal', 'non_destructive_coal', 
                   'not_coal', 'pulverized_coal', 'strongly_destructive_coal']
    
    # Get predictions on test set
    print("🔄 Running predictions on test dataset...")
    
    # Collect all test images and true labels
    test_images = []
    true_labels = []
    
    for class_idx, class_name in enumerate(class_names):
        test_dir = f"data/test/{class_name}"
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for file in files:
                test_images.append(os.path.join(test_dir, file))
                true_labels.append(class_idx)
    
    print(f"📊 Total test images: {len(test_images)}")
    
    # Get predictions
    predicted_labels = []
    predicted_probs = []
    confidence_scores = []
    
    print("🧠 Processing predictions...")
    for i, img_path in enumerate(test_images):
        if i % 20 == 0:
            print(f"  Progress: {i+1}/{len(test_images)}")
        
        try:
            results = model(img_path, verbose=False)
            if results and len(results) > 0:
                probs = results[0].probs
                if probs is not None:
                    predicted_labels.append(int(probs.top1))
                    confidence_scores.append(float(probs.top1conf))
                    # Get all class probabilities
                    predicted_probs.append(probs.data.cpu().numpy())
                else:
                    predicted_labels.append(0)
                    confidence_scores.append(0.0)
                    predicted_probs.append(np.zeros(len(class_names)))
            else:
                predicted_labels.append(0)
                confidence_scores.append(0.0)
                predicted_probs.append(np.zeros(len(class_names)))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            predicted_labels.append(0)
            confidence_scores.append(0.0)
            predicted_probs.append(np.zeros(len(class_names)))
    
    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    predicted_probs = np.array(predicted_probs)
    confidence_scores = np.array(confidence_scores)
    
    print("✅ Predictions completed!")
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    
    # Overall Accuracy
    overall_accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    # Macro and Micro averages
    precision_macro = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    precision_micro = precision_score(true_labels, predicted_labels, average='micro', zero_division=0)
    precision_weighted = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    
    recall_macro = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall_micro = recall_score(true_labels, predicted_labels, average='micro', zero_division=0)
    recall_weighted = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    
    f1_macro = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1_micro = f1_score(true_labels, predicted_labels, average='micro', zero_division=0)
    f1_weighted = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    
    print(f"\nPRECISION:")
    print(f"  Macro Average:    {precision_macro:.4f} ({precision_macro*100:.2f}%)")
    print(f"  Micro Average:    {precision_micro:.4f} ({precision_micro*100:.2f}%)")
    print(f"  Weighted Average: {precision_weighted:.4f} ({precision_weighted*100:.2f}%)")
    
    print(f"\nRECALL:")
    print(f"  Macro Average:    {recall_macro:.4f} ({recall_macro*100:.2f}%)")
    print(f"  Micro Average:    {recall_micro:.4f} ({recall_micro*100:.2f}%)")
    print(f"  Weighted Average: {recall_weighted:.4f} ({recall_weighted*100:.2f}%)")
    
    print(f"\nF1-SCORE:")
    print(f"  Macro Average:    {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"  Micro Average:    {f1_micro:.4f} ({f1_micro*100:.2f}%)")
    print(f"  Weighted Average: {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
    
    # Additional metrics
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    mcc = matthews_corrcoef(true_labels, predicted_labels)
    
    print(f"\nADDITIONAL METRICS:")
    print(f"  Cohen's Kappa:           {kappa:.4f}")
    print(f"  Matthews Correlation:    {mcc:.4f}")
    print(f"  Average Confidence:      {np.mean(confidence_scores):.4f}")
    print(f"  Confidence Std:          {np.std(confidence_scores):.4f}")
    
    # Per-class metrics
    print("\n" + "="*80)
    print("PER-CLASS DETAILED METRICS")
    print("="*80)
    
    # Calculate per-class metrics
    precision_per_class = precision_score(true_labels, predicted_labels, average=None, zero_division=0)
    recall_per_class = recall_score(true_labels, predicted_labels, average=None, zero_division=0)
    f1_per_class = f1_score(true_labels, predicted_labels, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    print(f"{'CLASS':<25} {'PRECISION':<12} {'RECALL':<12} {'F1-SCORE':<12} {'SUPPORT':<10}")
    print("-" * 80)
    
    for i, class_name in enumerate(class_names):
        support = np.sum(true_labels == i)
        precision = precision_per_class[i] if i < len(precision_per_class) else 0
        recall = recall_per_class[i] if i < len(recall_per_class) else 0
        f1 = f1_per_class[i] if i < len(f1_per_class) else 0
        
        print(f"{class_name:<25} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10d}")
    
    # Confusion Matrix Analysis
    print("\n" + "="*80)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*80)
    
    print("\nConfusion Matrix:")
    print("Rows = True Labels, Columns = Predicted Labels")
    print(f"{'':25}", end="")
    for name in class_names:
        print(f"{name[:12]:<12}", end="")
    print()
    
    for i, true_class in enumerate(class_names):
        print(f"{true_class:<25}", end="")
        for j in range(len(class_names)):
            value = cm[i][j] if i < len(cm) and j < len(cm[i]) else 0
            print(f"{value:<12d}", end="")
        print()
    
    # Per-class accuracy from confusion matrix
    print("\nPer-Class Accuracy (from Confusion Matrix):")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        if i < len(cm):
            correct = cm[i][i]
            total = np.sum(cm[i])
            accuracy = correct / total if total > 0 else 0
            print(f"{class_name:<25}: {accuracy:.4f} ({accuracy*100:.2f}%) - {correct}/{total}")
    
    # Confidence Analysis
    print("\n" + "="*80)
    print("CONFIDENCE ANALYSIS")
    print("="*80)
    
    # Per-class confidence
    for i, class_name in enumerate(class_names):
        class_indices = np.where(true_labels == i)[0]
        if len(class_indices) > 0:
            class_confidences = confidence_scores[class_indices]
            class_predictions = predicted_labels[class_indices]
            correct_predictions = class_predictions == i
            
            if len(class_confidences) > 0:
                avg_conf = np.mean(class_confidences)
                correct_conf = np.mean(class_confidences[correct_predictions]) if np.any(correct_predictions) else 0
                incorrect_conf = np.mean(class_confidences[~correct_predictions]) if np.any(~correct_predictions) else 0
                
                print(f"{class_name:<25}:")
                print(f"  Average Confidence:     {avg_conf:.4f}")
                print(f"  Correct Predictions:    {correct_conf:.4f}")
                print(f"  Incorrect Predictions:  {incorrect_conf:.4f}")
                print(f"  Confidence Range:       {np.min(class_confidences):.4f} - {np.max(class_confidences):.4f}")
    
    # Error Analysis
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    
    # Find misclassified samples
    misclassified = true_labels != predicted_labels
    misclassified_indices = np.where(misclassified)[0]
    
    print(f"Total Misclassified: {len(misclassified_indices)}/{len(true_labels)} ({len(misclassified_indices)/len(true_labels)*100:.2f}%)")
    
    if len(misclassified_indices) > 0:
        print("\nMost Common Misclassifications:")
        print("-" * 60)
        
        # Count misclassification patterns
        error_patterns = {}
        for idx in misclassified_indices:
            true_class = class_names[true_labels[idx]]
            pred_class = class_names[predicted_labels[idx]]
            pattern = f"{true_class} → {pred_class}"
            error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
        
        # Sort by frequency
        sorted_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
        
        for pattern, count in sorted_errors[:10]:  # Top 10 error patterns
            percentage = count / len(misclassified_indices) * 100
            print(f"{pattern:<50} {count:3d} ({percentage:5.1f}%)")
    
    # Model Performance Summary
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    # Performance interpretation
    if overall_accuracy >= 0.95:
        status = "🎯 EXCELLENT"
    elif overall_accuracy >= 0.90:
        status = "✅ VERY GOOD"
    elif overall_accuracy >= 0.85:
        status = "✅ GOOD"
    elif overall_accuracy >= 0.80:
        status = "⚠️  FAIR"
    else:
        status = "❌ NEEDS IMPROVEMENT"
    
    print(f"Overall Performance: {status}")
    print(f"Overall Accuracy: {overall_accuracy*100:.2f}%")
    print(f"Balanced Accuracy: {np.mean(recall_per_class)*100:.2f}%")
    print(f"Average F1-Score: {f1_macro*100:.2f}%")
    print(f"Average Confidence: {np.mean(confidence_scores)*100:.2f}%")
    
    # Recommendations
    print(f"\nSTRENGTHS:")
    best_classes = np.argsort(f1_per_class)[-3:]  # Top 3 classes
    for idx in reversed(best_classes):
        if idx < len(class_names):
            print(f"  • {class_names[idx]}: F1={f1_per_class[idx]:.3f}")
    
    print(f"\nAREAS FOR IMPROVEMENT:")
    worst_classes = np.argsort(f1_per_class)[:3]  # Bottom 3 classes
    for idx in worst_classes:
        if idx < len(class_names) and f1_per_class[idx] < 0.95:
            print(f"  • {class_names[idx]}: F1={f1_per_class[idx]:.3f}")
    
    # Create comprehensive metrics dictionary
    metrics_summary = {
        'overall_accuracy': overall_accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'cohen_kappa': kappa,
        'matthews_correlation': mcc,
        'per_class_metrics': {
            class_names[i]: {
                'precision': precision_per_class[i] if i < len(precision_per_class) else 0,
                'recall': recall_per_class[i] if i < len(recall_per_class) else 0,
                'f1_score': f1_per_class[i] if i < len(f1_per_class) else 0,
                'support': int(np.sum(true_labels == i))
            } for i in range(len(class_names))
        },
        'confusion_matrix': cm.tolist() if cm.size > 0 else [],
        'avg_confidence': float(np.mean(confidence_scores)),
        'confidence_std': float(np.std(confidence_scores))
    }
    
    return metrics_summary

if __name__ == "__main__":
    comprehensive_model_evaluation()
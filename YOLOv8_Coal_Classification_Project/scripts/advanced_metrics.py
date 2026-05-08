#!/usr/bin/env python3
"""
Advanced YOLOv8 Coal Classification Model Metrics Calculator
Calculates precision, recall, F1-score, mAP50, mAP50-95, FPS, and composite values
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from ultralytics import YOLO
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class CoalClassificationMetrics:
    def __init__(self, model_path="coal_classification_runs/coal_yolov8_small/weights/best.pt"):
        """Initialize the metrics calculator"""
        self.model = YOLO(model_path)
        self.model_path = model_path
        # Prefer the class name mapping embedded in the trained model (if present).
        # Fallback to a sensible default mapping when model has no names attribute.
        model_names = None
        try:
            model_names = getattr(self.model, 'names', None) or getattr(getattr(self.model, 'model', None), 'names', None)
        except Exception:
            model_names = None

        if model_names is not None:
            # model_names is typically a dict like {0: 'class0', 1: 'class1', ...}
            # Ensure we have an ordered mapping from index -> name
            self.class_names = {int(k): v for k, v in sorted(model_names.items(), key=lambda x: int(x[0]))}
        else:
            # Legacy fallback (should rarely be used)
            self.class_names = {
                0: "destructive_coal",
                1: "fully_pulverized_coal", 
                2: "non_destructive_coal",
                3: "pulverized_coal",
                4: "strongly_destructive_coal",
                5: "not_coal"
            }

        # Class labels in index order for display and reports
        self.class_labels = [self.class_names[i] for i in sorted(self.class_names.keys())]
        self.val_dir = "data/val"
        
    def collect_predictions(self):
        """Collect all predictions and ground truth labels"""
        print("🔍 Collecting predictions from validation set...")
        
        y_true = []
        y_pred = []
        y_scores = []  # For mAP calculations
        inference_times = []
        
        for class_idx, class_name in self.class_names.items():
            class_dir = os.path.join(self.val_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue
                
            # Accept multiple common image extensions (jpg, jpeg, png, bmp, tif)
            exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
            image_files = []
            for pattern in exts:
                image_files.extend(glob.glob(os.path.join(class_dir, pattern)))
            print(f"Processing {len(image_files)} images from {class_name}")
            
            for img_path in image_files:
                if not os.path.exists(img_path):
                    print(f"Warning: Image {img_path} not found")
                    continue
                    
                try:
                    # Measure inference time
                    start_time = time.time()
                    results = self.model.predict(img_path, verbose=False)
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
                    
                    if len(results) > 0:
                        probs = results[0].probs
                        if probs is not None:
                            # Ground truth
                            y_true.append(class_idx)
                            
                            # Prediction
                            predicted_class_id = probs.top1
                            y_pred.append(predicted_class_id)
                            
                            # Confidence scores for all classes (for mAP)
                            all_probs = probs.data.cpu().numpy()
                            y_scores.append(all_probs)
                except Exception as e:
                    print(f"Error processing {Path(img_path).name}: {str(e)}")
                    continue
        
        return np.array(y_true), np.array(y_pred), np.array(y_scores), np.array(inference_times)
    
    def calculate_precision_recall_f1(self, y_true, y_pred):
        """Calculate precision, recall, and F1-score for each class"""
        print("📊 Calculating Precision, Recall, and F1-Score...")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(self.class_names))
        )
        
        # Overall metrics (macro and weighted averages)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        return {
            'per_class': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            },
            'macro': {
                'precision': precision_macro,
                'recall': recall_macro,
                'f1': f1_macro
            },
            'weighted': {
                'precision': precision_weighted,
                'recall': recall_weighted,
                'f1': f1_weighted
            }
        }
    
    def calculate_map_scores(self, y_true, y_scores):
        """Calculate mAP@0.5 and mAP@0.5:0.95 scores"""
        print("🎯 Calculating mAP@0.5 and mAP@0.5:0.95...")
        
        # Binarize the labels for multi-class mAP calculation
        y_true_binary = label_binarize(y_true, classes=range(len(self.class_names)))
        
        # Calculate AP for each class at different IoU thresholds
        # For classification, we use confidence thresholds instead of IoU
        
        # mAP@0.5 equivalent (confidence threshold 0.5)
        ap_scores_50 = []
        for i in range(len(self.class_names)):
            if len(np.unique(y_true_binary[:, i])) > 1:  # Check if class exists
                ap = average_precision_score(y_true_binary[:, i], y_scores[:, i])
                ap_scores_50.append(ap)
            else:
                ap_scores_50.append(0.0)
        
        map_50 = np.mean(ap_scores_50)
        
        # mAP@0.5:0.95 equivalent (multiple confidence thresholds)
        thresholds = np.linspace(0.5, 0.95, 10)
        map_scores = []
        
        for threshold in thresholds:
            # Create binary predictions at this threshold
            y_pred_thresh = (np.max(y_scores, axis=1) >= threshold).astype(int)
            
            # Calculate average precision at this threshold
            ap_scores_thresh = []
            for i in range(len(self.class_names)):
                if len(np.unique(y_true_binary[:, i])) > 1:
                    ap = average_precision_score(y_true_binary[:, i], y_scores[:, i])
                    ap_scores_thresh.append(ap)
                else:
                    ap_scores_thresh.append(0.0)
            
            map_scores.append(np.mean(ap_scores_thresh))
        
        map_50_95 = np.mean(map_scores)
        
        return {
            'mAP@0.5': map_50,
            'mAP@0.5:0.95': map_50_95,
            'per_class_ap': ap_scores_50
        }
    
    def calculate_fps(self, inference_times):
        """Calculate FPS (Frames Per Second)"""
        print("⚡ Calculating FPS...")
        
        avg_inference_time = np.mean(inference_times)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            'avg_inference_time_ms': avg_inference_time * 1000,
            'fps': fps,
            'min_inference_time_ms': np.min(inference_times) * 1000,
            'max_inference_time_ms': np.max(inference_times) * 1000,
            'std_inference_time_ms': np.std(inference_times) * 1000
        }
    
    def calculate_composite_metrics(self, metrics_dict):
        """Calculate composite performance metrics"""
        print("🔄 Calculating Composite Metrics...")
        
        # Extract key metrics
        f1_macro = metrics_dict['precision_recall_f1']['macro']['f1']
        map_50 = metrics_dict['map_scores']['mAP@0.5']
        map_50_95 = metrics_dict['map_scores']['mAP@0.5:0.95']
        fps = metrics_dict['fps_metrics']['fps']
        
        # Composite Score 1: Accuracy-Speed Balance
        # Weighted combination of F1 and FPS (normalized)
        normalized_fps = min(fps / 100.0, 1.0)  # Normalize FPS to 0-1 range
        accuracy_speed_balance = 0.7 * f1_macro + 0.3 * normalized_fps
        
        # Composite Score 2: Overall Performance Index
        # Combines F1, mAP@0.5, and mAP@0.5:0.95
        overall_performance = (f1_macro + map_50 + map_50_95) / 3.0
        
        # Composite Score 3: Production Readiness Score
        # Considers accuracy, consistency, and speed
        accuracy_component = f1_macro
        consistency_component = 1.0 - (metrics_dict['fps_metrics']['std_inference_time_ms'] / 
                                     metrics_dict['fps_metrics']['avg_inference_time_ms'])
        speed_component = normalized_fps
        
        production_readiness = (0.5 * accuracy_component + 
                              0.3 * consistency_component + 
                              0.2 * speed_component)
        
        return {
            'accuracy_speed_balance': accuracy_speed_balance,
            'overall_performance_index': overall_performance,
            'production_readiness_score': production_readiness,
            'model_efficiency_ratio': f1_macro / (metrics_dict['fps_metrics']['avg_inference_time_ms'] / 1000)
        }
    
    def create_detailed_report(self, y_true, y_pred, all_metrics):
        """Create detailed classification report"""
        print("📋 Generating Detailed Report...")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification Report
        class_report = classification_report(
            y_true, y_pred, 
            target_names=self.class_labels,
            output_dict=True
        )
        
        return {
            'confusion_matrix': cm,
            'classification_report': class_report
        }
    
    def save_confusion_matrix(self, confusion_mat, save_path="confusion_matrix_advanced.png"):
        """Save confusion matrix visualization"""
        plt.figure(figsize=(10, 8))
        
        # Create labels for better readability
        labels = [name.replace('_', ' ').title() for name in self.class_labels]
        
        # Create heatmap
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        
        plt.title('Coal Classification - Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confusion matrix saved as: {save_path}")
    
    def print_comprehensive_results(self, all_metrics):
        """Print comprehensive results in a formatted way"""
        
        print("\n" + "="*80)
        print("🔥 COMPREHENSIVE YOLOv8 COAL CLASSIFICATION METRICS")
        print("="*80)
        
        # Basic Performance Metrics
        print(f"\n📊 ACCURACY METRICS")
        print("-" * 50)
        prf_metrics = all_metrics['precision_recall_f1']
        
        print(f"Overall Accuracy: {all_metrics['accuracy']:.4f} ({all_metrics['accuracy']*100:.2f}%)")
        print(f"Macro F1-Score: {prf_metrics['macro']['f1']:.4f}")
        print(f"Weighted F1-Score: {prf_metrics['weighted']['f1']:.4f}")
        print(f"Macro Precision: {prf_metrics['macro']['precision']:.4f}")
        print(f"Macro Recall: {prf_metrics['macro']['recall']:.4f}")
        
        # Per-class metrics
        print(f"\n📈 PER-CLASS PERFORMANCE")
        print("-" * 50)
        print(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 75)
        
        for i, class_name in enumerate(self.class_labels):
            display_name = class_name.replace('_', ' ').title()
            precision = prf_metrics['per_class']['precision'][i]
            recall = prf_metrics['per_class']['recall'][i]
            f1 = prf_metrics['per_class']['f1'][i]
            support = prf_metrics['per_class']['support'][i]
            
            print(f"{display_name:<25} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")
        
        # mAP Scores
        print(f"\n🎯 mAP SCORES")
        print("-" * 50)
        map_metrics = all_metrics['map_scores']
        print(f"mAP@0.5: {map_metrics['mAP@0.5']:.4f} ({map_metrics['mAP@0.5']*100:.2f}%)")
        print(f"mAP@0.5:0.95: {map_metrics['mAP@0.5:0.95']:.4f} ({map_metrics['mAP@0.5:0.95']*100:.2f}%)")
        
        # FPS and Speed Metrics
        print(f"\n⚡ SPEED METRICS")
        print("-" * 50)
        fps_metrics = all_metrics['fps_metrics']
        print(f"FPS (Frames Per Second): {fps_metrics['fps']:.2f}")
        print(f"Average Inference Time: {fps_metrics['avg_inference_time_ms']:.2f} ms")
        print(f"Min Inference Time: {fps_metrics['min_inference_time_ms']:.2f} ms")
        print(f"Max Inference Time: {fps_metrics['max_inference_time_ms']:.2f} ms")
        print(f"Std Inference Time: {fps_metrics['std_inference_time_ms']:.2f} ms")
        
        # Composite Metrics
        print(f"\n🔄 COMPOSITE METRICS")
        print("-" * 50)
        comp_metrics = all_metrics['composite_metrics']
        print(f"Accuracy-Speed Balance: {comp_metrics['accuracy_speed_balance']:.4f}")
        print(f"Overall Performance Index: {comp_metrics['overall_performance_index']:.4f}")
        print(f"Production Readiness Score: {comp_metrics['production_readiness_score']:.4f}")
        print(f"Model Efficiency Ratio: {comp_metrics['model_efficiency_ratio']:.2f}")
        
        # Model Information
        print(f"\n🔧 MODEL INFORMATION")
        print("-" * 50)
        print(f"Model Path: {self.model_path}")
        print(f"Architecture: YOLOv8 Nano Classification")
        print(f"Classes: {len(self.class_names)} coal types")
        print(f"Input Size: 224x224 pixels")
        print(f"Total Test Images: {all_metrics['total_images']}")
    
    def save_metrics_to_csv(self, all_metrics, filename="coal_classification_metrics.csv"):
        """Save all metrics to CSV file"""
        
        # Prepare data for CSV
        csv_data = []
        
        # Add per-class metrics
        prf_metrics = all_metrics['precision_recall_f1']
        for i, class_name in enumerate(self.class_labels):
            csv_data.append({
                'Class': class_name.replace('_', ' ').title(),
                'Precision': prf_metrics['per_class']['precision'][i],
                'Recall': prf_metrics['per_class']['recall'][i],
                'F1_Score': prf_metrics['per_class']['f1'][i],
                'Support': prf_metrics['per_class']['support'][i],
                'AP_Score': all_metrics['map_scores']['per_class_ap'][i]
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False)
        
        # Save summary metrics
        summary_data = {
            'Metric': ['Overall_Accuracy', 'Macro_F1', 'Weighted_F1', 'mAP@0.5', 'mAP@0.5:0.95', 
                      'FPS', 'Avg_Inference_ms', 'Accuracy_Speed_Balance', 'Overall_Performance', 'Production_Readiness'],
            'Value': [
                all_metrics['accuracy'],
                prf_metrics['macro']['f1'],
                prf_metrics['weighted']['f1'],
                all_metrics['map_scores']['mAP@0.5'],
                all_metrics['map_scores']['mAP@0.5:0.95'],
                all_metrics['fps_metrics']['fps'],
                all_metrics['fps_metrics']['avg_inference_time_ms'],
                all_metrics['composite_metrics']['accuracy_speed_balance'],
                all_metrics['composite_metrics']['overall_performance_index'],
                all_metrics['composite_metrics']['production_readiness_score']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(filename.replace('.csv', '_summary.csv'), index=False)
        
        print(f"📁 Detailed metrics saved to: {filename}")
        print(f"📁 Summary metrics saved to: {filename.replace('.csv', '_summary.csv')}")
    
    def run_comprehensive_evaluation(self):
        """Run the complete evaluation pipeline"""
        
        print("🚀 Starting Comprehensive Model Evaluation...")
        print("="*80)
        
        # Step 1: Collect predictions
        y_true, y_pred, y_scores, inference_times = self.collect_predictions()
        
        if len(y_true) == 0:
            print("❌ No predictions collected. Check your validation dataset path.")
            return
        
        print(f"✅ Collected {len(y_true)} predictions")
        
        # Step 2: Calculate basic accuracy
        accuracy = np.mean(y_true == y_pred)
        
        # Step 3: Calculate precision, recall, F1
        prf_metrics = self.calculate_precision_recall_f1(y_true, y_pred)
        
        # Step 4: Calculate mAP scores
        map_metrics = self.calculate_map_scores(y_true, y_scores)
        
        # Step 5: Calculate FPS
        fps_metrics = self.calculate_fps(inference_times)
        
        # Step 6: Compile all metrics
        all_metrics = {
            'accuracy': accuracy,
            'total_images': len(y_true),
            'precision_recall_f1': prf_metrics,
            'map_scores': map_metrics,
            'fps_metrics': fps_metrics
        }
        
        # Step 7: Calculate composite metrics
        composite_metrics = self.calculate_composite_metrics(all_metrics)
        all_metrics['composite_metrics'] = composite_metrics
        
        # Step 8: Create detailed report
        detailed_report = self.create_detailed_report(y_true, y_pred, all_metrics)
        all_metrics['detailed_report'] = detailed_report
        
        # Step 9: Print results
        self.print_comprehensive_results(all_metrics)
        
        # Step 10: Save confusion matrix
        self.save_confusion_matrix(detailed_report['confusion_matrix'])
        
        # Step 11: Save metrics to CSV
        self.save_metrics_to_csv(all_metrics)
        
        print(f"\n✅ Comprehensive evaluation completed successfully!")
        print("="*80)
        
        return all_metrics

def main():
    """Main function to run the comprehensive metrics evaluation"""
    
    # Initialize the metrics calculator
    evaluator = CoalClassificationMetrics()
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    return results

if __name__ == "__main__":
    main()
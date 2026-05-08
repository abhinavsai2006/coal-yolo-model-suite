import os
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import json
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveModelEvaluator:
    """
    Comprehensive evaluation and testing suite for YOLOv10 Coal Classification
    """
    
    def __init__(self, model_path="yolov10_diversified/coal_fixed_overfitting/weights/best.pt"):
        """Initialize the comprehensive evaluator"""
        print("🔍 COMPREHENSIVE YOLOv10 MODEL EVALUATION & TESTING")
        print("🎯 Complete performance analysis with multiple metrics")
        print("=" * 70)
        
        # Load model
        if not os.path.exists(model_path):
            # Try alternative paths
            alt_paths = [
                "yolov10_gpu_classification/coal_classification/weights/best.pt",
                "yolov10_ultra_aggressive/coal_ultra_reg2/weights/best.pt",
                "runs/classify/train/weights/best.pt"
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Model not found at any path!")
        
        self.model = YOLO(model_path)
        self.model_path = model_path
        
        # Class definitions
        self.class_names = [
            'destructive_coal',
            'fully_pulverized_coal', 
            'non_coal',
            'non_destructive_coal',
            'pulverized_coal',
            'strongly_destructive_coal'
        ]
        
        self.test_path = "dataset/test"
        self.val_path = "dataset/val"
        
        # Create results directory
        os.makedirs('comprehensive_evaluation', exist_ok=True)
        
        print(f"✅ Model loaded: {model_path}")
        print(f"📊 Classes: {len(self.class_names)}")
        
    def evaluate_test_set(self):
        """Comprehensive evaluation on test set"""
        print("\\n🧪 TEST SET EVALUATION")
        print("=" * 50)
        
        all_predictions = []
        all_true_labels = []
        all_confidences = []
        per_class_results = {}
        inference_times = []
        
        total_images = 0
        correct_predictions = 0
        
        print("📊 Processing test classes...")
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.test_path, class_name)
            if not os.path.exists(class_path):
                print(f"⚠️ Class directory not found: {class_path}")
                continue
            
            print(f"  📁 {class_name}...")
            
            class_correct = 0
            class_total = 0
            class_confidences = []
            class_predictions = []
            class_times = []
            
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_file)
                    
                    # Skip extremely small/corrupt files only
                    try:
                        with Image.open(img_path) as test_img:
                            if test_img.size[0] < 10 or test_img.size[1] < 10:
                                continue
                    except:
                        continue
                    
                    try:
                        # Time the inference
                        start_time = time.time()
                        results = self.model(img_path, verbose=False)
                        inference_time = time.time() - start_time
                        
                        if results[0].probs is not None:
                            probs = results[0].probs.data.cpu().numpy()
                            predicted_idx = np.argmax(probs)
                            confidence = np.max(probs)
                            
                            # Store results
                            all_predictions.append(predicted_idx)
                            all_true_labels.append(class_idx)
                            all_confidences.append(confidence)
                            inference_times.append(inference_time)
                            
                            class_total += 1
                            total_images += 1
                            class_confidences.append(confidence)
                            class_predictions.append(predicted_idx)
                            class_times.append(inference_time)
                            
                            if predicted_idx == class_idx:
                                class_correct += 1
                                correct_predictions += 1
                    
                    except Exception as e:
                        print(f"    ⚠️ Error processing {img_file}: {str(e)}")
                        continue
            
            # Store per-class results
            if class_total > 0:
                accuracy = class_correct / class_total
                avg_confidence = np.mean(class_confidences)
                avg_time = np.mean(class_times)
                
                per_class_results[class_name] = {
                    'accuracy': accuracy,
                    'confidence': avg_confidence,
                    'inference_time': avg_time,
                    'correct': class_correct,
                    'total': class_total,
                    'predictions': class_predictions
                }
                
                # Status
                if accuracy >= 1.0:
                    status = "🔴 PERFECT (possible overfitting)"
                elif accuracy >= 0.95:
                    status = "🟢 EXCELLENT"
                elif accuracy >= 0.85:
                    status = "🟡 GOOD"
                else:
                    status = "🟠 NEEDS IMPROVEMENT"
                
                print(f"    {class_name:25} | Acc: {accuracy:.3f} | Conf: {avg_confidence:.3f} | Time: {avg_time:.3f}s | ({class_correct}/{class_total}) | {status}")
        
        return {
            'predictions': all_predictions,
            'true_labels': all_true_labels,
            'confidences': all_confidences,
            'inference_times': inference_times,
            'per_class_results': per_class_results,
            'total_images': total_images,
            'correct_predictions': correct_predictions
        }
    
    def evaluate_validation_set(self):
        """Evaluation on validation set"""
        print("\\n📊 VALIDATION SET EVALUATION")
        print("=" * 50)
        
        val_predictions = []
        val_true_labels = []
        val_confidences = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.val_path, class_name)
            if not os.path.exists(class_path):
                continue
            
            print(f"  📁 {class_name}...")
            
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_file)
                    
                    # Skip extremely small/corrupt files only
                    try:
                        with Image.open(img_path) as test_img:
                            if test_img.size[0] < 10 or test_img.size[1] < 10:
                                continue
                    except:
                        continue
                    
                    try:
                        results = self.model(img_path, verbose=False)
                        if results[0].probs is not None:
                            probs = results[0].probs.data.cpu().numpy()
                            predicted_idx = np.argmax(probs)
                            confidence = np.max(probs)
                            
                            val_predictions.append(predicted_idx)
                            val_true_labels.append(class_idx)
                            val_confidences.append(confidence)
                    
                    except Exception as e:
                        continue
        
        if val_predictions:
            val_accuracy = accuracy_score(val_true_labels, val_predictions)
            val_avg_confidence = np.mean(val_confidences)
            
            print(f"📊 Validation Accuracy: {val_accuracy:.4f}")
            print(f"🎯 Validation Avg Confidence: {val_avg_confidence:.4f}")
            
            return {
                'accuracy': val_accuracy,
                'avg_confidence': val_avg_confidence,
                'predictions': val_predictions,
                'true_labels': val_true_labels
            }
        
        return None
    
    def calculate_advanced_metrics(self, test_results):
        """Calculate advanced performance metrics"""
        print("\\n📈 ADVANCED METRICS CALCULATION")
        print("=" * 50)
        
        predictions = test_results['predictions']
        true_labels = test_results['true_labels']
        confidences = test_results['confidences']
        
        # Basic metrics
        overall_accuracy = accuracy_score(true_labels, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, labels=range(len(self.class_names))
        )
        
        # Macro and weighted averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)
        
        # Confidence statistics
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)
        min_confidence = np.min(confidences)
        max_confidence = np.max(confidences)
        
        # Performance timing
        avg_inference_time = np.mean(test_results['inference_times'])
        
        metrics = {
            'overall_accuracy': overall_accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std,
            'min_confidence': min_confidence,
            'max_confidence': max_confidence,
            'avg_inference_time': avg_inference_time,
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(),
            'per_class_support': support.tolist()
        }
        
        # Print metrics
        print(f"🎯 Overall Accuracy: {overall_accuracy:.4f}")
        print(f"📊 Macro F1-Score: {macro_f1:.4f}")
        print(f"📊 Weighted F1-Score: {weighted_f1:.4f}")
        print(f"🎯 Average Confidence: {avg_confidence:.4f} ± {confidence_std:.4f}")
        print(f"⚡ Average Inference Time: {avg_inference_time:.4f}s")
        
        print(f"\\n📋 Per-Class Performance:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name:25} | P: {precision[i]:.3f} | R: {recall[i]:.3f} | F1: {f1[i]:.3f} | Support: {support[i]}")
        
        return metrics
    
    def analyze_confusion_patterns(self, test_results):
        """Analyze confusion matrix and error patterns"""
        print("\\n🔍 CONFUSION MATRIX ANALYSIS")
        print("=" * 50)
        
        predictions = test_results['predictions']
        true_labels = test_results['true_labels']
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Analyze patterns
        print("📊 Most Common Confusions:")
        confusion_pairs = []
        
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i][j] > 0:
                    confusion_pairs.append({
                        'true_class': self.class_names[i],
                        'predicted_class': self.class_names[j],
                        'count': cm[i][j],
                        'percentage': cm[i][j] / np.sum(cm[i]) * 100
                    })
        
        # Sort by count
        confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        for pair in confusion_pairs[:5]:  # Top 5 confusions
            print(f"  {pair['true_class']} → {pair['predicted_class']}: {pair['count']} ({pair['percentage']:.1f}%)")
        
        return cm, confusion_pairs
    
    def create_comprehensive_visualizations(self, test_results, val_results, metrics, cm):
        """Create comprehensive visualization plots"""
        print("\\n📊 CREATING COMPREHENSIVE VISUALIZATIONS")
        print("=" * 50)
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(3, 3, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names, ax=ax1)
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # 2. Per-class accuracy
        ax2 = plt.subplot(3, 3, 2)
        class_names_short = [name.replace('_', '\\n') for name in self.class_names]
        accuracies = [test_results['per_class_results'][name]['accuracy'] for name in self.class_names]
        bars = ax2.bar(class_names_short, accuracies, color='skyblue', alpha=0.8)
        ax2.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1.1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Per-class confidence
        ax3 = plt.subplot(3, 3, 3)
        confidences = [test_results['per_class_results'][name]['confidence'] for name in self.class_names]
        bars = ax3.bar(class_names_short, confidences, color='orange', alpha=0.8)
        ax3.set_title('Per-Class Average Confidence', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Confidence')
        ax3.set_ylim(0, 1.1)
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{conf:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. F1-Scores
        ax4 = plt.subplot(3, 3, 4)
        f1_scores = metrics['per_class_f1']
        bars = ax4.bar(class_names_short, f1_scores, color='lightgreen', alpha=0.8)
        ax4.set_title('Per-Class F1-Score', fontsize=14, fontweight='bold')
        ax4.set_ylabel('F1-Score')
        ax4.set_ylim(0, 1.1)
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, f1 in zip(bars, f1_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{f1:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Confidence distribution
        ax5 = plt.subplot(3, 3, 5)
        ax5.hist(test_results['confidences'], bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax5.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Confidence')
        ax5.set_ylabel('Frequency')
        ax5.axvline(metrics['avg_confidence'], color='red', linestyle='--', 
                   label=f'Mean: {metrics["avg_confidence"]:.3f}')
        ax5.legend()
        
        # 6. Inference time distribution
        ax6 = plt.subplot(3, 3, 6)
        ax6.hist(test_results['inference_times'], bins=20, alpha=0.7, color='cyan', edgecolor='black')
        ax6.set_title('Inference Time Distribution', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Time (seconds)')
        ax6.set_ylabel('Frequency')
        ax6.axvline(metrics['avg_inference_time'], color='red', linestyle='--',
                   label=f'Mean: {metrics["avg_inference_time"]:.3f}s')
        ax6.legend()
        
        # 7. Precision vs Recall
        ax7 = plt.subplot(3, 3, 7)
        precision = metrics['per_class_precision']
        recall = metrics['per_class_recall']
        scatter = ax7.scatter(recall, precision, c=range(len(self.class_names)), 
                             cmap='viridis', s=100, alpha=0.8)
        ax7.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Recall')
        ax7.set_ylabel('Precision')
        ax7.set_xlim(0, 1.1)
        ax7.set_ylim(0, 1.1)
        ax7.grid(True, alpha=0.3)
        
        # Add class labels
        for i, (r, p) in enumerate(zip(recall, precision)):
            ax7.annotate(self.class_names[i].replace('_', '\\n'), (r, p), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 8. Model summary
        ax8 = plt.subplot(3, 3, 8)
        summary_text = f"""MODEL PERFORMANCE SUMMARY
        
Overall Accuracy: {metrics['overall_accuracy']:.3f}
Macro F1-Score: {metrics['macro_f1']:.3f}
Weighted F1-Score: {metrics['weighted_f1']:.3f}

Average Confidence: {metrics['avg_confidence']:.3f}
Confidence Std: {metrics['confidence_std']:.3f}

Avg Inference Time: {metrics['avg_inference_time']:.3f}s
Total Test Images: {test_results['total_images']}

Model Path: 
{os.path.basename(self.model_path)}"""
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax8.set_xlim(0, 1)
        ax8.set_ylim(0, 1)
        ax8.axis('off')
        
        # 9. Train vs Val comparison (if available)
        ax9 = plt.subplot(3, 3, 9)
        if val_results:
            comparison_data = {
                'Test Accuracy': metrics['overall_accuracy'],
                'Val Accuracy': val_results['accuracy'],
                'Test Confidence': metrics['avg_confidence'],
                'Val Confidence': val_results['avg_confidence']
            }
            
            categories = list(comparison_data.keys())
            values = list(comparison_data.values())
            colors = ['blue', 'lightblue', 'red', 'lightcoral']
            
            bars = ax9.bar(categories, values, color=colors, alpha=0.8)
            ax9.set_title('Test vs Validation Comparison', fontsize=14, fontweight='bold')
            ax9.set_ylabel('Score')
            ax9.set_ylim(0, 1.1)
            ax9.tick_params(axis='x', rotation=45)
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax9.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax9.text(0.5, 0.5, 'Validation data\\nnot available', 
                    ha='center', va='center', transform=ax9.transAxes, fontsize=12)
            ax9.set_xlim(0, 1)
            ax9.set_ylim(0, 1)
            ax9.axis('off')
        
        plt.tight_layout()
        plt.savefig('comprehensive_evaluation/comprehensive_evaluation.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Comprehensive visualization saved: comprehensive_evaluation/comprehensive_evaluation.png")
    
    def save_evaluation_report(self, test_results, val_results, metrics, confusion_pairs):
        """Save comprehensive evaluation report"""
        print("\\n💾 SAVING EVALUATION REPORT")
        print("=" * 50)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'test_results': {
                'total_images': test_results['total_images'],
                'correct_predictions': test_results['correct_predictions'],
                'per_class_results': test_results['per_class_results']
            },
            'validation_results': val_results,
            'metrics': metrics,
            'confusion_analysis': confusion_pairs[:10],  # Top 10 confusions
            'class_names': self.class_names
        }
        
        # Save as JSON
        report_path = 'comprehensive_evaluation/evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save as text summary
        summary_path = 'comprehensive_evaluation/evaluation_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("YOLOv10 COAL CLASSIFICATION - COMPREHENSIVE EVALUATION REPORT\\n")
            f.write("=" * 70 + "\\n\\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Model: {self.model_path}\\n\\n")
            
            f.write("OVERALL PERFORMANCE:\\n")
            f.write(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}\\n")
            f.write(f"  Macro F1-Score: {metrics['macro_f1']:.4f}\\n")
            f.write(f"  Weighted F1-Score: {metrics['weighted_f1']:.4f}\\n")
            f.write(f"  Average Confidence: {metrics['avg_confidence']:.4f}\\n")
            f.write(f"  Average Inference Time: {metrics['avg_inference_time']:.4f}s\\n\\n")
            
            f.write("PER-CLASS PERFORMANCE:\\n")
            for i, class_name in enumerate(self.class_names):
                f.write(f"  {class_name}:\\n")
                f.write(f"    Accuracy: {test_results['per_class_results'][class_name]['accuracy']:.3f}\\n")
                f.write(f"    Precision: {metrics['per_class_precision'][i]:.3f}\\n")
                f.write(f"    Recall: {metrics['per_class_recall'][i]:.3f}\\n")
                f.write(f"    F1-Score: {metrics['per_class_f1'][i]:.3f}\\n")
                f.write(f"    Support: {metrics['per_class_support'][i]}\\n\\n")
        
        print(f"📊 Evaluation report saved: {report_path}")
        print(f"📄 Summary saved: {summary_path}")
        
        return report
    
    def run_comprehensive_evaluation(self):
        """Run the complete evaluation suite"""
        print("\\n🚀 STARTING COMPREHENSIVE MODEL EVALUATION")
        print("=" * 70)
        
        # 1. Test set evaluation
        test_results = self.evaluate_test_set()
        
        # 2. Validation set evaluation
        val_results = self.evaluate_validation_set()
        
        # 3. Advanced metrics
        metrics = self.calculate_advanced_metrics(test_results)
        
        # 4. Confusion analysis
        cm, confusion_pairs = self.analyze_confusion_patterns(test_results)
        
        # 5. Visualizations
        self.create_comprehensive_visualizations(test_results, val_results, metrics, cm)
        
        # 6. Save report
        report = self.save_evaluation_report(test_results, val_results, metrics, confusion_pairs)
        
        # 7. Final summary
        print("\\n🎯 FINAL EVALUATION SUMMARY")
        print("=" * 50)
        print(f"✅ Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"📊 Macro F1-Score: {metrics['macro_f1']:.4f}")
        print(f"🎯 Average Confidence: {metrics['avg_confidence']:.4f}")
        print(f"⚡ Average Inference: {metrics['avg_inference_time']:.4f}s")
        print(f"📁 Total Test Images: {test_results['total_images']}")
        
        # Performance assessment
        if metrics['overall_accuracy'] >= 0.95:
            print("🎉 EXCELLENT PERFORMANCE!")
        elif metrics['overall_accuracy'] >= 0.90:
            print("✅ GOOD PERFORMANCE!")
        elif metrics['overall_accuracy'] >= 0.80:
            print("📊 ACCEPTABLE PERFORMANCE")
        else:
            print("⚠️ NEEDS IMPROVEMENT")
        
        return report

def main():
    """Main evaluation function"""
    try:
        evaluator = ComprehensiveModelEvaluator()
        report = evaluator.run_comprehensive_evaluation()
        
        print("\\n🎯 EVALUATION COMPLETED SUCCESSFULLY!")
        print("📁 Results saved in: comprehensive_evaluation/")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
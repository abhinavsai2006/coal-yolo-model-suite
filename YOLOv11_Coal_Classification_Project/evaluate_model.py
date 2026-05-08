"""
YOLOv11 Model Evaluation Script
Evaluates the trained YOLOv11 classification model on the test dataset.
"""

from ultralytics import YOLO
import os

def evaluate_model(model_path='runs/classify/coal_classification_6class/weights/best.pt'):
    """Evaluate the trained model on test data"""
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_model.py")
        return None
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Validate the model
    print("Running evaluation on test dataset...")
    metrics = model.val(data='data')
    
    # Print results
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Top-1 Accuracy: {metrics.top1:.4f} ({metrics.top1*100:.2f}%)")
    print(f"Top-5 Accuracy: {metrics.top5:.4f} ({metrics.top5*100:.2f}%)")
    
    # Print per-class results if available
    if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
        cm = metrics.confusion_matrix
        if hasattr(cm, 'matrix'):
            matrix = cm.matrix
            class_names = ['destructive_coal', 'fully_pulverized_coal', 
                          'non_destructive_coal', 'not_coal', 'pulverized_coal', 'strongly_destructive_coal']
            
            print("\nPer-Class Accuracy:")
            print("-" * 40)
            for i, class_name in enumerate(class_names):
                if i < len(matrix):
                    correct = matrix[i][i]
                    total = sum(matrix[i])
                    acc = correct / total if total > 0 else 0
                    print(f"{class_name:25}: {acc*100:6.2f}% ({int(correct):3d}/{int(total):3d})")
    
    print("="*50)
    return metrics

if __name__ == "__main__":
    evaluate_model()
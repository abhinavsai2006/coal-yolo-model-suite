from ultralytics import YOLO
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import yaml

class CoalClassificationEvaluator:
    def __init__(self, model_path: str = "coal_classification_runs/coal_yolov8_small/weights/best.pt", dataset_path: str = "data"):
        """
        Initialize the coal classification evaluator
        
        Args:
            model_path: Path to the trained model
            dataset_path: Path to the dataset
        """
        self.model = YOLO(model_path)
        self.dataset_path = Path(dataset_path)

        # Prefer the class name mapping embedded in the trained model (if present).
        # Fallback to a sensible default mapping when model has no names attribute.
        model_names = None
        try:
            model_names = getattr(self.model, 'names', None) or getattr(getattr(self.model, 'model', None), 'names', None)
        except Exception:
            model_names = None

        if model_names is not None:
            # model_names is typically a dict like {0: 'class0', 1: 'class1', ...}
            # Build a list indexed by class id in sorted order
            ordered = [model_names[i] for i in sorted(map(int, model_names.keys()))]
            self.class_names = ordered
        else:
            # Legacy fallback (should rarely be used)
            self.class_names = [
                'destructive_coal',
                'fully_pulverized_coal', 
                'non_destructive_coal',
                'pulverized_coal',
                'strongly_destructive_coal',
                'not_coal'
            ]
        
        print(f"🤖 Loaded model from: {model_path}")
        print(f"📂 Dataset path: {dataset_path}")
        print(f"🏷️  Classes: {self.class_names}")
    
    def evaluate_model(self):
        """Evaluate the model on validation set"""
        print("\n🔍 Evaluating Model Performance...")
        print("="*50)
        
        # Run validation
        results = self.model.val()
        
        print(f"📊 Validation Results:")
        print(f"   Top-1 Accuracy: {results.top1:.4f} ({results.top1*100:.2f}%)")
        print(f"   Top-5 Accuracy: {results.top5:.4f} ({results.top5*100:.2f}%)")
        
        return results
    
    def predict_single_image(self, image_path: str, show_result: bool = True):
        """
        Predict coal type for a single image
        
        Args:
            image_path: Path to the image
            show_result: Whether to display the result
        """
        results = self.model(image_path)
        result = results[0]
        
        # Get prediction details
        probs = result.probs
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        top5_indices = probs.top5
        top5_conf = probs.top5conf
        
        predicted_class = self.class_names[top1_idx]
        
        print(f"\n🔮 Prediction for: {Path(image_path).name}")
        print(f"   Predicted Class: {predicted_class}")
        print(f"   Confidence: {top1_conf:.4f} ({top1_conf*100:.2f}%)")
        
        print(f"\n📊 Top 5 Predictions:")
        for i, (idx, conf) in enumerate(zip(top5_indices, top5_conf)):
            class_name = self.class_names[idx]
            print(f"   {i+1}. {class_name}: {conf:.4f} ({conf*100:.2f}%)")
        
        if show_result:
            self._display_prediction(image_path, predicted_class, top1_conf, probs)
        
        return predicted_class, top1_conf, probs
    
    def _display_prediction(self, image_path: str, predicted_class: str, confidence: float, probs):
        """Display prediction results with image and probability chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax1.imshow(image_rgb)
        ax1.set_title(f'Image: {Path(image_path).name}\nPredicted: {predicted_class}\nConfidence: {confidence:.2%}', 
                     fontsize=12)
        ax1.axis('off')
        
        # Display probability chart
        confidences = probs.data.cpu().numpy()
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.class_names)))
        
        bars = ax2.bar(range(len(self.class_names)), confidences, color=colors)
        ax2.set_xlabel('Coal Types')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Prediction Confidence for All Classes')
        ax2.set_xticks(range(len(self.class_names)))
        ax2.set_xticklabels([name.replace('_', ' ').title() for name in self.class_names], 
                           rotation=45, ha='right')
        
        # Highlight the predicted class
        max_idx = np.argmax(confidences)
        bars[max_idx].set_color('red')
        
        # Add value labels on bars
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{conf:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def test_on_validation_samples(self, num_samples: int = 10):
        """Test the model on random validation samples"""
        print(f"\n🎯 Testing on {num_samples} random validation samples...")
        print("="*60)
        
        val_path = self.dataset_path / "val"
        
        # Collect all validation images
        val_images = []
        true_labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = val_path / class_name
            if class_path.exists():
                # Accept common image extensions
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
                images = []
                for ext in image_extensions:
                    images.extend(list(class_path.glob(ext)))
                val_images.extend(images)
                true_labels.extend([class_idx] * len(images))
        
        # Random sample
        import random
        sample_indices = random.sample(range(len(val_images)), min(num_samples, len(val_images)))
        
        correct_predictions = 0
        
        for i, idx in enumerate(sample_indices):
            image_path = val_images[idx]
            true_class_idx = true_labels[idx]
            true_class_name = self.class_names[true_class_idx]
            
            # Predict
            pred_class, confidence, probs = self.predict_single_image(str(image_path), show_result=False)
            
            # Check if correct
            is_correct = pred_class == true_class_name
            correct_predictions += is_correct
            
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            print(f"\n🖼️  Sample {i+1}/{len(sample_indices)}")
            print(f"   Image: {image_path.name}")
            print(f"   True: {true_class_name}")
            print(f"   Predicted: {pred_class} ({confidence:.2%})")
            print(f"   Result: {status}")
        
        accuracy = correct_predictions / len(sample_indices)
        print(f"\n📊 Sample Test Results:")
        print(f"   Correct Predictions: {correct_predictions}/{len(sample_indices)}")
        print(f"   Sample Accuracy: {accuracy:.2%}")
        
        return accuracy
    
    def create_confusion_matrix(self):
        """Create confusion matrix for validation set"""
        print("\n📊 Creating Confusion Matrix...")
        
        val_path = self.dataset_path / "val"
        true_labels = []
        pred_labels = []
        
        # Collect all validation predictions
        for class_idx, class_name in enumerate(self.class_names):
            class_path = val_path / class_name
            if class_path.exists():
                # Accept common image extensions
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
                images = []
                for ext in image_extensions:
                    images.extend(list(class_path.glob(ext)))
                for image_path in images:
                    # Get prediction
                    results = self.model(str(image_path))
                    pred_idx = results[0].probs.top1
                    
                    true_labels.append(class_idx)
                    pred_labels.append(pred_idx)
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[name.replace('_', ' ').title() for name in self.class_names],
                    yticklabels=[name.replace('_', ' ').title() for name in self.class_names])
        plt.title('Confusion Matrix - Coal Classification')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        plt.show()
        
        # Print classification report
        print("\n📋 Classification Report:")
        print("="*60)
        target_names = [name.replace('_', ' ').title() for name in self.class_names]
        report = classification_report(true_labels, pred_labels, target_names=target_names)
        print(report)
        
        return cm
    
    def predict_folder(self, folder_path: str, save_results: bool = True):
        """
        Predict coal types for all images in a folder
        
        Args:
            folder_path: Path to folder containing images
            save_results: Whether to save results to a file
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            print(f"❌ Folder not found: {folder_path}")
            return
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        for ext in image_extensions:
            images.extend(folder_path.glob(f"*{ext}"))
            images.extend(folder_path.glob(f"*{ext.upper()}"))
        
        if not images:
            print(f"❌ No images found in {folder_path}")
            return
        
        print(f"\n🔍 Predicting coal types for {len(images)} images...")
        results = []
        
        for image_path in images:
            pred_class, confidence, probs = self.predict_single_image(str(image_path), show_result=False)
            results.append({
                'image': image_path.name,
                'predicted_class': pred_class,
                'confidence': confidence
            })
            print(f"✅ {image_path.name}: {pred_class} ({confidence:.2%})")
        
        if save_results:
            results_path = folder_path / "prediction_results.txt"
            with open(results_path, 'w') as f:
                f.write("Coal Classification Results\n")
                f.write("="*40 + "\n")
                for result in results:
                    f.write(f"{result['image']}: {result['predicted_class']} ({result['confidence']:.2%})\n")
            print(f"\n💾 Results saved to: {results_path}")
        
        return results

def main():
    """Main evaluation function"""
    print("🔥 YOLOv8 Coal Classification Model Evaluation")
    print("="*60)
    
    # Define model path
    model_path = "coal_classification_runs/coal_yolov8_small/weights/best.pt"
    
    if Path(model_path).exists():
        print(f"\n🤖 Evaluating model: {model_path}")
        
        # Create evaluator
        evaluator = CoalClassificationEvaluator(model_path)
        
        # Evaluate model
        results = evaluator.evaluate_model()
        
        # Test on validation samples
        accuracy = evaluator.test_on_validation_samples(num_samples=10)
        
        # Create confusion matrix
        evaluator.create_confusion_matrix()
        
        print(f"\n✅ Evaluation completed for {model_path}")
    else:
        print(f"❌ Model not found: {model_path}")
        if Path(model_path).exists():
            print(f"\n🤖 Evaluating model: {model_path}")
            
            # Create evaluator
            evaluator = CoalClassificationEvaluator(model_path)
            
            # Evaluate model
            results = evaluator.evaluate_model()
            
            # Test on validation samples
            accuracy = evaluator.test_on_validation_samples(num_samples=10)
            
            # Create confusion matrix
            evaluator.create_confusion_matrix()
            
            print(f"\n✅ Evaluation completed for {model_path}")
        else:
            print(f"❌ Model not found: {model_path}")

if __name__ == "__main__":
    main()
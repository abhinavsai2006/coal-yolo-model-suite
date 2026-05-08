"""
Utility Functions for YOLO11 Hybrid Model
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_size(model):
    """Get model size in MB"""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def print_model_summary(model, input_size=(3, 224, 224)):
    """Print detailed model summary"""
    print("="*80)
    print("Model Summary")
    print("="*80)
    
    # Parameter count
    params = count_parameters(model)
    print(f"\nParameter Count:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Non-trainable: {params['non_trainable']:,}")
    
    # Model size
    size_mb = get_model_size(model)
    print(f"\nModel Size: {size_mb:.2f} MB")
    
    # Layer-wise breakdown
    print("\nLayer-wise Parameter Count:")
    print(f"{'Layer Name':<50} {'Parameters':>15}")
    print("-"*80)
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            num_params = sum(p.numel() for p in module.parameters())
            if num_params > 0:
                print(f"{name:<50} {num_params:>15,}")
    
    print("="*80)


def visualize_training_history(history_path, save_path=None):
    """Visualize training history"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    train_history = history['train']
    val_history = history['val']
    
    epochs = [h['epoch'] for h in train_history]
    train_loss = [h['loss'] for h in train_history]
    train_acc = [h['acc'] for h in train_history]
    val_loss = [h['loss'] for h in val_history]
    val_acc = [h['acc'] for h in val_history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_loss, label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_loss, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_acc, label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, val_acc, label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_models(results_dict):
    """
    Compare multiple model results
    
    Args:
        results_dict: Dictionary with model names as keys and result paths as values
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for model_name, result_path in results_dict.items():
        with open(result_path, 'r') as f:
            results = json.load(f)
        
        model_names.append(model_name)
        accuracies.append(results['accuracy'])
        
        # Calculate macro average
        report = results['classification_report']
        macro_avg = report['macro avg']
        precisions.append(macro_avg['precision'] * 100)
        recalls.append(macro_avg['recall'] * 100)
        f1_scores.append(macro_avg['f1-score'] * 100)
    
    x = np.arange(len(model_names))
    width = 0.2
    
    # Plot 1: Overall metrics
    ax1.bar(x - width*1.5, accuracies, width, label='Accuracy', alpha=0.8)
    ax1.bar(x - width*0.5, precisions, width, label='Precision', alpha=0.8)
    ax1.bar(x + width*0.5, recalls, width, label='Recall', alpha=0.8)
    ax1.bar(x + width*1.5, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Score (%)', fontsize=12)
    ax1.set_title('Model Comparison - Overall Metrics', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # Plot 2: Accuracy comparison
    bars = ax2.bar(model_names, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Model Comparison - Accuracy', fontsize=14)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_ylim([0, 105])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("Model comparison plot saved to model_comparison.png")
    plt.close()


def export_to_onnx(model, save_path, input_size=(1, 3, 224, 224)):
    """Export model to ONNX format"""
    dummy_input = torch.randn(input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX: {save_path}")


def calculate_flops(model, input_size=(1, 3, 224, 224)):
    """
    Estimate FLOPs (rough estimation)
    Note: This is a simplified estimation
    """
    def count_conv_flops(module, input, output):
        # For Conv2d: FLOPs = 2 * C_in * K * K * C_out * H_out * W_out
        batch_size = output.shape[0]
        output_height = output.shape[2]
        output_width = output.shape[3]
        
        kernel_height = module.kernel_size[0]
        kernel_width = module.kernel_size[1]
        in_channels = module.in_channels
        out_channels = module.out_channels
        
        flops = 2 * in_channels * kernel_height * kernel_width * \
                out_channels * output_height * output_width
        
        module.__flops__ += flops
    
    def count_linear_flops(module, input, output):
        # For Linear: FLOPs = 2 * in_features * out_features
        flops = 2 * module.in_features * module.out_features
        module.__flops__ += flops
    
    # Register hooks
    hooks = []
    total_flops = 0
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.__flops__ = 0
            hooks.append(module.register_forward_hook(count_conv_flops))
        elif isinstance(module, nn.Linear):
            module.__flops__ = 0
            hooks.append(module.register_forward_hook(count_linear_flops))
    
    # Forward pass
    dummy_input = torch.randn(input_size)
    model.eval()
    with torch.no_grad():
        model(dummy_input)
    
    # Calculate total FLOPs
    for module in model.modules():
        if hasattr(module, '__flops__'):
            total_flops += module.__flops__
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return total_flops


if __name__ == "__main__":
    print("Utility functions for YOLO11 Hybrid Model")
    print("Import this module to use the utility functions")

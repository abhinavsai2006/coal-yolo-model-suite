#!/usr/bin/env python3
"""
Visual Metrics Dashboard for YOLOv8 Coal Classification Model
Creates comprehensive visualizations of all performance metrics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def create_metrics_dashboard():
    """Create a comprehensive metrics dashboard"""
    
    # Set style for better-looking plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Define the metrics data from our evaluation
    class_names = ['Destructive Coal', 'Fully Pulverized Coal', 'Non Destructive Coal', 
                   'Pulverized Coal', 'Strongly Destructive Coal']
    
    precision_scores = [0.9783, 0.9531, 1.0000, 0.9559, 0.9595]
    recall_scores = [1.0000, 0.9385, 0.9870, 0.9420, 0.9726]
    f1_scores = [0.9890, 0.9457, 0.9935, 0.9489, 0.9660]
    support_counts = [90, 65, 77, 69, 73]
    
    # 1. Per-class Performance Bar Chart
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(class_names))
    width = 0.25
    
    bars1 = ax1.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
    bars2 = ax1.bar(x, recall_scores, width, label='Recall', alpha=0.8)
    bars3 = ax1.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax1.set_xlabel('Coal Types')
    ax1.set_ylabel('Score')
    ax1.set_title('Per-Class Performance Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.replace(' Coal', '') for name in class_names], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.9, 1.0)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Overall Metrics Gauge Chart
    ax2 = plt.subplot(2, 3, 2)
    
    # Key metrics
    overall_metrics = {
        'Overall Accuracy': 0.9706,
        'Macro F1-Score': 0.9686,
        'mAP@0.5': 0.9883,
        'mAP@0.5:0.95': 0.9883
    }
    
    # Create horizontal bar chart
    metrics_names = list(overall_metrics.keys())
    metrics_values = list(overall_metrics.values())
    
    bars = ax2.barh(metrics_names, metrics_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax2.set_xlabel('Score')
    ax2.set_title('Overall Performance Metrics')
    ax2.set_xlim(0.9, 1.0)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, metrics_values)):
        ax2.text(value + 0.002, i, f'{value:.4f} ({value*100:.2f}%)', 
                va='center', fontsize=10, weight='bold')
    
    # 3. Speed and Efficiency Metrics
    ax3 = plt.subplot(2, 3, 3)
    
    speed_metrics = {
        'FPS': 13.12,
        'Avg Inference (ms)': 76.22,
        'Min Inference (ms)': 20.66,
        'Max Inference (ms)': 252.93
    }
    
    # Create pie chart for speed distribution
    inference_times = [20.66, 76.22-20.66, 252.93-76.22]
    labels = ['Min Time', 'Avg Range', 'Max Range']
    colors = ['#90EE90', '#FFD700', '#FFB6C1']
    
    wedges, texts, autotexts = ax3.pie(inference_times, labels=labels, colors=colors, 
                                       autopct='%1.1f%%', startangle=90)
    ax3.set_title('Inference Time Distribution\n(FPS: 13.12)')
    
    # 4. Support Distribution (Class Balance)
    ax4 = plt.subplot(2, 3, 4)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    wedges, texts, autotexts = ax4.pie(support_counts, labels=[name.replace(' Coal', '') for name in class_names], 
                                       colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Dataset Class Distribution\n(Total: 374 images)')
    
    # 5. Composite Metrics Radar Chart
    ax5 = plt.subplot(2, 3, 5, projection='polar')
    
    composite_metrics = {
        'Accuracy': 0.9706,
        'Precision': 0.9693,
        'Recall': 0.9680,
        'F1-Score': 0.9686,
        'mAP@0.5': 0.9883,
        'Speed Score': 0.1312  # Normalized FPS/100
    }
    
    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(composite_metrics), endpoint=False)
    values = list(composite_metrics.values())
    
    # Close the plot
    angles = np.concatenate((angles, [angles[0]]))
    values = np.concatenate((values, [values[0]]))
    
    ax5.plot(angles, values, 'o-', linewidth=2, label='Model Performance')
    ax5.fill(angles, values, alpha=0.25)
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(composite_metrics.keys())
    ax5.set_ylim(0, 1)
    ax5.set_title('Performance Radar Chart')
    ax5.grid(True)
    
    # 6. Key Statistics Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Summary statistics text
    summary_text = f"""
    🔥 YOLO v8 Coal Classification Model
    ═══════════════════════════════════════
    
    📊 CORE METRICS
    • Overall Accuracy: 97.06%
    • Macro F1-Score: 96.86%
    • Weighted F1-Score: 97.05%
    
    🎯 mAP SCORES  
    • mAP@0.5: 98.83%
    • mAP@0.5:0.95: 98.83%
    
    ⚡ PERFORMANCE
    • FPS: 13.12
    • Avg Inference: 76.22 ms
    • Model Size: 1.44M parameters
    
    🔄 COMPOSITE SCORES
    • Overall Performance: 98.17%
    • Production Readiness: 61.96%
    • Model Efficiency: 12.71
    
    🏆 BEST PERFORMING CLASSES
    • Non Destructive Coal: 100% Precision
    • Destructive Coal: 100% Recall
    • Non Destructive Coal: 99.35% F1
    
    ✅ MODEL STATUS: PRODUCTION READY
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('coal_classification_metrics_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Comprehensive metrics dashboard saved as: coal_classification_metrics_dashboard.png")

def create_confusion_matrix_heatmap():
    """Create an enhanced confusion matrix visualization"""
    
    # Confusion matrix data from our evaluation
    confusion_matrix = np.array([
        [90,  0,  0,  0,  0],  # Destructive Coal
        [ 1, 61,  0,  3,  0],  # Fully Pulverized Coal  
        [ 0,  0, 76,  0,  1],  # Non Destructive Coal
        [ 1,  1,  0, 65,  2],  # Pulverized Coal
        [ 0,  0,  2,  0, 71]   # Strongly Destructive Coal
    ])
    
    class_names = ['Destructive\nCoal', 'Fully Pulverized\nCoal', 'Non Destructive\nCoal', 
                   'Pulverized\nCoal', 'Strongly Destructive\nCoal']
    
    plt.figure(figsize=(12, 10))
    
    # Create heatmap with custom styling
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='RdYlBu_r',
                xticklabels=class_names, yticklabels=class_names,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Coal Classification - Detailed Confusion Matrix\n(97.06% Overall Accuracy)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
    plt.ylabel('True Class', fontsize=14, fontweight='bold')
    
    # Add accuracy annotations
    for i in range(len(class_names)):
        accuracy = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
        plt.text(len(class_names) + 0.5, i + 0.5, f'{accuracy:.1%}', 
                ha='center', va='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('enhanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Enhanced confusion matrix saved as: enhanced_confusion_matrix.png")

if __name__ == "__main__":
    print("🎨 Creating comprehensive metrics visualization...")
    create_metrics_dashboard()
    create_confusion_matrix_heatmap()
    print("✅ All visualizations completed!")
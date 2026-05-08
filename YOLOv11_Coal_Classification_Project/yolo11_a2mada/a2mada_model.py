"""
A2MADA-YOLO Model Architecture
Attention Alignment Multiscale Adversarial Domain Adaptation YOLO
For Coal Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

# Add modules to path
sys.path.append(str(Path(__file__).parent))
from modules.attention_alignment import (
    MultiscaleAttentionAlignment,
    AdversarialAttention,
    CBAM
)


class ConvBlock(nn.Module):
    """Basic Convolutional Block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C3Block(nn.Module):
    """C3 Block from YOLO architecture"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=True):
        super(C3Block, self).__init__()
        hidden_channels = out_channels // 2
        
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1, 1, 0)
        self.conv2 = ConvBlock(in_channels, hidden_channels, 1, 1, 0)
        self.conv3 = ConvBlock(2 * hidden_channels, out_channels, 1, 1, 0)
        
        self.bottleneck = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, shortcut)
            for _ in range(n)
        ])
    
    def forward(self, x):
        return self.conv3(torch.cat([self.bottleneck(self.conv1(x)), self.conv2(x)], dim=1))


class Bottleneck(nn.Module):
    """Bottleneck Block"""
    def __init__(self, in_channels, out_channels, shortcut=True):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 1, 1, 0)
        self.conv2 = ConvBlock(out_channels, out_channels, 3, 1, 1)
        self.shortcut = shortcut and in_channels == out_channels
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.shortcut else self.conv2(self.conv1(x))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast"""
    def __init__(self, in_channels, out_channels, k=5):
        super(SPPF, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1, 1, 0)
        self.conv2 = ConvBlock(hidden_channels * 4, out_channels, 1, 1, 0)
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class A2MADABackbone(nn.Module):
    """
    A2MADA Backbone with Multiscale Attention Alignment
    """
    def __init__(self, channels_list=[64, 128, 256, 512], depth_multiple=0.33):
        super(A2MADABackbone, self).__init__()
        
        # Calculate depths
        n1 = max(round(3 * depth_multiple), 1)
        n2 = max(round(6 * depth_multiple), 1)
        n3 = max(round(9 * depth_multiple), 1)
        
        # Stem
        self.stem = ConvBlock(3, channels_list[0], 6, 2, 2)
        
        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBlock(channels_list[0], channels_list[1], 3, 2, 1),
            C3Block(channels_list[1], channels_list[1], n1)
        )
        self.attention1 = MultiscaleAttentionAlignment(channels_list[1])
        
        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBlock(channels_list[1], channels_list[2], 3, 2, 1),
            C3Block(channels_list[2], channels_list[2], n2)
        )
        self.attention2 = MultiscaleAttentionAlignment(channels_list[2])
        
        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBlock(channels_list[2], channels_list[3], 3, 2, 1),
            C3Block(channels_list[3], channels_list[3], n3)
        )
        self.attention3 = MultiscaleAttentionAlignment(channels_list[3])
        
        # SPPF
        self.sppf = SPPF(channels_list[3], channels_list[3])
        
        # Final attention
        self.final_attention = CBAM(channels_list[3])
    
    def forward(self, x, return_domain_outputs=False):
        domain_outputs = []
        
        # Stem
        x = self.stem(x)
        
        # Stage 1 with attention
        x = self.stage1(x)
        if return_domain_outputs:
            x, d1 = self.attention1(x, return_domain_output=True)
            domain_outputs.append(d1)
        else:
            x = self.attention1(x)
        
        # Stage 2 with attention
        x = self.stage2(x)
        if return_domain_outputs:
            x, d2 = self.attention2(x, return_domain_output=True)
            domain_outputs.append(d2)
        else:
            x = self.attention2(x)
        
        # Stage 3 with attention
        x = self.stage3(x)
        if return_domain_outputs:
            x, d3 = self.attention3(x, return_domain_output=True)
            domain_outputs.append(d3)
        else:
            x = self.attention3(x)
        
        # SPPF
        x = self.sppf(x)
        
        # Final attention
        x = self.final_attention(x)
        
        if return_domain_outputs:
            return x, domain_outputs
        return x


class A2MADAClassifier(nn.Module):
    """
    A2MADA-YOLO Classifier with Adversarial Training
    """
    def __init__(self, num_classes=6, channels_list=[64, 128, 256, 512], dropout=0.4):
        super(A2MADAClassifier, self).__init__()
        
        self.backbone = A2MADABackbone(channels_list)
        
        # Adversarial attention module
        self.adversarial_attention = AdversarialAttention(
            in_channels=channels_list[3],
            num_classes=num_classes
        )
        
        # Alternative simple classifier (for inference)
        self.simple_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels_list[3], num_classes)
        )
    
    def forward(self, x, alpha=1.0, use_adversarial=False):
        """
        Args:
            x: Input images
            alpha: Gradient reversal weight
            use_adversarial: Use adversarial training mode
        """
        # Extract features
        if use_adversarial:
            features, domain_outputs = self.backbone(x, return_domain_outputs=True)
            # Adversarial prediction
            class_pred, domain_pred = self.adversarial_attention(
                features, alpha=alpha, return_domain_output=True
            )
            return class_pred, domain_pred, domain_outputs
        else:
            features = self.backbone(x, return_domain_outputs=False)
            # Simple prediction for inference
            class_pred = self.simple_classifier(features)
            return class_pred


def create_a2mada_model(num_classes=6, channels_list=[64, 128, 256, 512], dropout=0.4):
    """
    Factory function to create A2MADA-YOLO model
    
    Args:
        num_classes: Number of classification classes
        channels_list: Channel dimensions for each stage
        dropout: Dropout rate
    
    Returns:
        A2MADAClassifier model
    """
    model = A2MADAClassifier(
        num_classes=num_classes,
        channels_list=channels_list,
        dropout=dropout
    )
    return model


def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    # Test the model
    print("Testing A2MADA-YOLO Model...")
    
    model = create_a2mada_model(num_classes=6)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / (1024**2):.2f} MB")
    
    # Test forward pass (inference mode)
    x = torch.randn(2, 3, 224, 224)
    print(f"\nInput shape: {x.shape}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        output = model(x, use_adversarial=False)
        print(f"Output shape (inference): {output.shape}")
    
    # Training mode (adversarial)
    model.train()
    output, domain_pred, domain_outputs = model(x, alpha=1.0, use_adversarial=True)
    print(f"Output shape (adversarial): {output.shape}")
    print(f"Domain prediction shape: {domain_pred.shape}")
    print(f"Number of domain outputs: {len(domain_outputs)}")
    
    print("\n✓ A2MADA-YOLO model test passed!")

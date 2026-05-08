"""
YOLO11 Hybrid Model with Pretrained Backbone
Using pretrained ResNet or EfficientNet for better feature extraction
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from modules.attention_modules import (
    MCIGLA,
    CrossLevelFeatureFusion,
    PolyKernelInception
)


class PretrainedHybridBackbone(nn.Module):
    """
    Hybrid Backbone using pretrained ResNet50 with custom attention modules
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Use ResNet blocks
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        # Add custom attention modules to enhance features
        self.attention1 = MCIGLA(256)
        self.attention2 = MCIGLA(512)
        self.attention3 = MCIGLA(1024)
        self.attention4 = MCIGLA(2048)
        
        # Add Poly-Kernel Inception for multi-scale features
        self.poly_inception = PolyKernelInception(2048, 2048)
        
        # Cross-level feature fusion
        self.cross_fusion = CrossLevelFeatureFusion(
            in_channels_list=[512, 1024, 2048],
            out_channels=2048
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Layer 1
        x1 = self.layer1(x)
        x1 = self.attention1(x1)
        
        # Layer 2
        x2 = self.layer2(x1)
        x2 = self.attention2(x2)
        
        # Layer 3
        x3 = self.layer3(x2)
        x3 = self.attention3(x3)
        
        # Layer 4
        x4 = self.layer4(x3)
        x4 = self.attention4(x4)
        
        # Apply Poly-Kernel Inception
        x4 = self.poly_inception(x4)
        
        # Cross-level fusion
        fused = self.cross_fusion([x2, x3, x4])
        
        # Global pooling
        features = self.global_pool(fused)
        features = features.flatten(1)
        
        return features


class EfficientNetHybridBackbone(nn.Module):
    """
    Hybrid Backbone using pretrained EfficientNet-B3
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load pretrained EfficientNet-B3
        if pretrained:
            efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        else:
            efficientnet = models.efficientnet_b3(weights=None)
        
        self.features = efficientnet.features
        
        # Get output channels (1536 for EfficientNet-B3)
        self.out_channels = 1536
        
        # Add custom attention
        self.attention = MCIGLA(self.out_channels)
        
        # Add Poly-Kernel Inception
        self.poly_inception = PolyKernelInception(self.out_channels, self.out_channels)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.poly_inception(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        return x


class PretrainedHybridClassifier(nn.Module):
    """
    Complete classifier with pretrained backbone
    """
    def __init__(self, num_classes=6, backbone_type='resnet50', pretrained=True, dropout=0.5):
        super().__init__()
        
        self.backbone_type = backbone_type
        
        # Select backbone
        if backbone_type == 'resnet50':
            self.backbone = PretrainedHybridBackbone(pretrained=pretrained)
            feature_dim = 2048
        elif backbone_type == 'efficientnet_b3':
            self.backbone = EfficientNetHybridBackbone(pretrained=pretrained)
            feature_dim = 1536
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
        
        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class PretrainedHybridWrapper:
    """
    Wrapper class for pretrained hybrid model
    """
    def __init__(self, model_config: Dict = None):
        self.model_config = model_config or {
            'num_classes': 6,
            'backbone_type': 'resnet50',
            'pretrained': True,
            'dropout': 0.5
        }
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def build_model(self):
        """Build the pretrained hybrid model"""
        self.model = PretrainedHybridClassifier(
            num_classes=self.model_config['num_classes'],
            backbone_type=self.model_config['backbone_type'],
            pretrained=self.model_config['pretrained'],
            dropout=self.model_config.get('dropout', 0.5)
        )
        self.model = self.model.to(self.device)
        return self.model
    
    def get_model_info(self):
        """Get model information"""
        if self.model is None:
            self.build_model()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_type': self.model_config['backbone_type'],
            'model_config': self.model_config,
            'device': str(self.device)
        }
        
        return info


def create_pretrained_hybrid_model(num_classes: int = 6, backbone_type: str = 'resnet50', 
                                   pretrained: bool = True, dropout: float = 0.5) -> PretrainedHybridWrapper:
    """
    Factory function to create Pretrained Hybrid Model
    
    Args:
        num_classes: Number of classification classes
        backbone_type: 'resnet50' or 'efficientnet_b3'
        pretrained: Whether to use ImageNet pretrained weights
        dropout: Dropout rate for classifier
    
    Returns:
        PretrainedHybridWrapper instance
    """
    config = {
        'num_classes': num_classes,
        'backbone_type': backbone_type,
        'pretrained': pretrained,
        'dropout': dropout
    }
    
    wrapper = PretrainedHybridWrapper(config)
    return wrapper


if __name__ == '__main__':
    # Test model creation
    print("Testing Pretrained Hybrid Model Creation...")
    
    # Test ResNet50 backbone
    print("\n1. ResNet50 Backbone:")
    wrapper_resnet = create_pretrained_hybrid_model(
        num_classes=6, 
        backbone_type='resnet50',
        pretrained=True
    )
    model_resnet = wrapper_resnet.build_model()
    info_resnet = wrapper_resnet.get_model_info()
    
    print(f"Total parameters: {info_resnet['total_parameters']:,}")
    print(f"Trainable parameters: {info_resnet['trainable_parameters']:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(wrapper_resnet.device)
    output = model_resnet(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Test EfficientNet-B3 backbone
    print("\n2. EfficientNet-B3 Backbone:")
    wrapper_effnet = create_pretrained_hybrid_model(
        num_classes=6,
        backbone_type='efficientnet_b3',
        pretrained=True
    )
    model_effnet = wrapper_effnet.build_model()
    info_effnet = wrapper_effnet.get_model_info()
    
    print(f"Total parameters: {info_effnet['total_parameters']:,}")
    print(f"Trainable parameters: {info_effnet['trainable_parameters']:,}")
    
    # Test forward pass
    output = model_effnet(dummy_input)
    print(f"Output shape: {output.shape}")
    
    print("\n✓ Model creation successful!")

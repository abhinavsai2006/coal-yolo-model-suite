"""
Attention Alignment Module for A2MADA-YOLO
Implements attention mechanisms for domain adaptation
Based on: https://github.com/HaoxingZhou/A2MADA-YOLO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


class AttentionAlignment(nn.Module):
    """
    Attention Alignment Module for Domain Adaptation
    Aligns features between source and target domains
    """
    def __init__(self, in_channels, hidden_channels=256):
        super(AttentionAlignment, self).__init__()
        
        # Feature transformation
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanism
        self.attention = CBAM(hidden_channels)
        
        # Domain classifier for alignment
        self.domain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )
    
    def forward(self, x, return_domain_output=False):
        # Transform features
        feat = self.transform(x)
        
        # Apply attention
        attended_feat = self.attention(feat)
        
        # Domain classification (for training)
        domain_output = None
        if return_domain_output:
            domain_output = self.domain_classifier(attended_feat)
        
        # Refine and add residual
        out = self.refine(attended_feat)
        out = out + x
        
        if return_domain_output:
            return out, domain_output
        return out


class MultiscaleAttentionAlignment(nn.Module):
    """
    Multiscale Attention Alignment
    Processes features at multiple scales with attention
    """
    def __init__(self, in_channels, scales=[1, 2, 4]):
        super(MultiscaleAttentionAlignment, self).__init__()
        self.scales = scales
        
        # Attention for each scale
        self.scale_attentions = nn.ModuleList([
            AttentionAlignment(in_channels, hidden_channels=in_channels // 2)
            for _ in scales
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * len(scales), in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, return_domain_output=False):
        B, C, H, W = x.shape
        scale_features = []
        domain_outputs = []
        
        for scale, attention_module in zip(self.scales, self.scale_attentions):
            # Downsample if needed
            if scale > 1:
                scaled_x = F.adaptive_avg_pool2d(x, (H // scale, W // scale))
            else:
                scaled_x = x
            
            # Apply attention alignment
            if return_domain_output:
                aligned_feat, domain_out = attention_module(scaled_x, return_domain_output=True)
                domain_outputs.append(domain_out)
            else:
                aligned_feat = attention_module(scaled_x)
            
            # Upsample back to original size
            if scale > 1:
                aligned_feat = F.interpolate(aligned_feat, size=(H, W), mode='bilinear', align_corners=False)
            
            scale_features.append(aligned_feat)
        
        # Fuse multiscale features
        fused = torch.cat(scale_features, dim=1)
        out = self.fusion(fused)
        
        if return_domain_output and domain_outputs:
            return out, torch.stack(domain_outputs).mean(dim=0)
        return out


class AdversarialAttention(nn.Module):
    """
    Adversarial Attention Module
    Combines attention with adversarial domain adaptation
    """
    def __init__(self, in_channels, num_classes=6):
        super(AdversarialAttention, self).__init__()
        
        # Multiscale attention alignment
        self.multiscale_alignment = MultiscaleAttentionAlignment(in_channels)
        
        # Task-specific attention
        self.task_attention = CBAM(in_channels)
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Domain discriminator
        self.domain_discriminator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        # Class predictor
        self.class_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, alpha=1.0, return_domain_output=False):
        """
        Args:
            x: Input features
            alpha: Weight for gradient reversal (for adversarial training)
            return_domain_output: Whether to return domain prediction
        """
        # Multiscale attention alignment
        if return_domain_output:
            aligned_feat, domain_out = self.multiscale_alignment(x, return_domain_output=True)
        else:
            aligned_feat = self.multiscale_alignment(x)
        
        # Task-specific attention
        task_feat = self.task_attention(aligned_feat)
        
        # Feature extraction
        feat = self.feature_extractor(task_feat)
        
        # Domain prediction (with gradient reversal during training)
        if return_domain_output:
            # Apply gradient reversal layer effect
            domain_feat = GradientReversalLayer.apply(feat, alpha)
            domain_pred = self.domain_discriminator(domain_feat)
        else:
            domain_pred = None
        
        # Class prediction
        class_pred = self.class_predictor(feat)
        
        if return_domain_output:
            return class_pred, domain_pred
        return class_pred


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for Adversarial Training
    Reverses gradients during backpropagation
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

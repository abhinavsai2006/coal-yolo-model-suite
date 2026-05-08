"""
Attention Modules for YOLO11 Hybrid Model
Implements MCI-GLA (Multiscale Channel Information with Global-Local Attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class GlobalLocalAttention(nn.Module):
    """
    Global-Local Attention (GLA) Module
    Combines global and local attention mechanisms
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(GlobalLocalAttention, self).__init__()
        
        # Global attention branch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        
        # Local attention branch
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
        self.weight = nn.Parameter(torch.ones(2))
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Global attention
        global_feat = self.global_pool(x).view(b, c)
        global_att = self.global_fc(global_feat).view(b, c, 1, 1)
        global_out = x * global_att
        
        # Local attention
        local_att = self.local_conv(x)
        local_out = x * local_att
        
        # Combine with learnable weights
        weights = F.softmax(self.weight, dim=0)
        out = weights[0] * global_out + weights[1] * local_out
        
        return out


class MultiscaleChannelInformation(nn.Module):
    """
    Multiscale Channel Information (MCI) Module
    Extracts channel information at multiple scales
    """
    def __init__(self, in_channels: int, scales: List[int] = [1, 3, 5]):
        super(MultiscaleChannelInformation, self).__init__()
        
        self.scales = scales
        self.branches = nn.ModuleList()
        
        for scale in scales:
            if scale == 1:
                branch = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1)
                )
            else:
                padding = (scale - 1) // 2
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, scale, padding=padding, groups=in_channels),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1)
                )
            self.branches.append(branch)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * len(scales), in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch_outs = []
        for branch in self.branches:
            branch_outs.append(branch(x))
        
        # Concatenate all scales
        out = torch.cat(branch_outs, dim=1)
        out = self.fusion(out)
        
        # Expand back to spatial dimensions
        out = out.expand_as(x)
        return out


class MCIGLA(nn.Module):
    """
    MCI-GLA: Multiscale Channel Information with Global-Local Attention
    A plug-in module for YOLO series models
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16, scales: List[int] = [1, 3, 5]):
        super(MCIGLA, self).__init__()
        
        self.mci = MultiscaleChannelInformation(in_channels, scales)
        self.gla = GlobalLocalAttention(in_channels, reduction_ratio)
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention()
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Extract multiscale channel information
        mci_out = self.mci(x)
        
        # Apply global-local attention
        gla_out = self.gla(x)
        
        # Combine MCI and GLA
        combined = x + mci_out * gla_out
        
        # Apply channel attention
        ca_out = self.channel_att(combined)
        combined = combined * ca_out
        
        # Apply spatial attention
        sa_out = self.spatial_att(combined)
        combined = combined * sa_out
        
        # Final fusion
        out = self.fusion(combined)
        
        return out


class CrossLevelFeatureFusion(nn.Module):
    """
    Cross-Level Feature Fusion Module
    Fuses features from different network levels
    """
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super(CrossLevelFeatureFusion, self).__init__()
        
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        # Projection layers for each level
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for in_ch in in_channels_list
        ])
        
        # Attention weights for fusion
        self.attention_weights = nn.Parameter(torch.ones(len(in_channels_list)))
        
        # Final fusion conv
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features_list: List of feature maps from different levels
        Returns:
            Fused feature map
        """
        # Get target size from the largest feature map
        target_size = features_list[0].shape[2:]
        
        # Project and resize all features to target size
        projected_features = []
        for i, (feat, proj) in enumerate(zip(features_list, self.projections)):
            proj_feat = proj(feat)
            if proj_feat.shape[2:] != target_size:
                proj_feat = F.interpolate(proj_feat, size=target_size, mode='bilinear', align_corners=False)
            projected_features.append(proj_feat)
        
        # Apply attention weights
        weights = F.softmax(self.attention_weights, dim=0)
        fused = sum(w * feat for w, feat in zip(weights, projected_features))
        
        # Final fusion
        out = self.fusion_conv(fused)
        
        return out


class PolyKernelInception(nn.Module):
    """
    Poly-Kernel Inception Module
    Uses multiple kernel sizes for feature extraction
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(PolyKernelInception, self).__init__()
        
        branch_channels = out_channels // 4
        
        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 1x1 -> 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 1x1 -> 5x5 conv (implemented as two 3x3)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: 3x3 max pooling -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention for branches
        self.branch_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # Concatenate all branches
        out = torch.cat([b1, b2, b3, b4], dim=1)
        
        # Apply attention
        att = self.branch_attention(out)
        out = out * att
        
        return out

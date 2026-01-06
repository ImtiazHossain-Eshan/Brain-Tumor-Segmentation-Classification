"""
Attention U-Net architecture for improved segmentation
Based on: https://arxiv.org/abs/1804.03999
Author: Imtiaz Hossain (ID: 23101137)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import DoubleConv, Down, OutConv, count_parameters


class AttentionGate(nn.Module):
    """
    Attention Gate module
    Highlights salient features from skip connections
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: Number of feature maps (channels) in previous layer (gating signal)
            F_l: Number of feature maps in corresponding encoder layer (skip connection)
            F_int: Number of feature maps in intermediate layer
        """
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: Gating signal from deeper layer (B, F_g, H_g, W_g)
            x: Skip connection from encoder (B, F_l, H_x, W_x)
        
        Returns:
            Attention-weighted features
        """
        # Transform gating signal
        g1 = self.W_g(g)
        
        # Transform skip connection
        x1 = self.W_x(x)
        
        # Combine and apply attention
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention weights to skip connection
        return x * psi


class AttentionUp(nn.Module):
    """Upscaling with attention gate then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(AttentionUp, self).__init__()
        
        # Upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.attention = AttentionGate(F_g=in_channels // 2, F_l=in_channels // 2, F_int=out_channels)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.attention = AttentionGate(F_g=in_channels // 2, F_l=in_channels // 2, F_int=out_channels)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Input from previous layer (gating signal)
            x2: Skip connection from encoder
        """
        # Upsample
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Apply attention gate
        x2 = self.attention(g=x1, x=x2)
        
        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net for improved segmentation
    Paper: https://arxiv.org/abs/1804.03999
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 64,
        bilinear: bool = True
    ):
        super(AttentionUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Encoder (same as U-Net)
        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(base_filters * 8, base_filters * 16 // factor)
        
        # Decoder with attention gates
        self.up1 = AttentionUp(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.up2 = AttentionUp(base_filters * 8, base_filters * 4 // factor, bilinear)
        self.up3 = AttentionUp(base_filters * 4, base_filters * 2 // factor, bilinear)
        self.up4 = AttentionUp(base_filters * 2, base_filters, bilinear)
        
        # Output layer
        self.outc = OutConv(base_filters, out_channels)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with attention gates
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits
    
    def get_encoder_features(self, x):
        """
        Get encoder features for classification head
        Returns the bottleneck features
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        return x5


class AttentionUNetWithClassifier(nn.Module):
    """
    Attention U-Net with classification head
    For joint training or separate training experiments
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_filters: int = 64,
        dropout: float = 0.5
    ):
        super(AttentionUNetWithClassifier, self).__init__()
        
        # Attention U-Net for segmentation
        self.attention_unet = AttentionUNet(
            in_channels=in_channels,
            out_channels=1,
            base_filters=base_filters
        )
        
        # Classification head
        bottleneck_channels = base_filters * 8
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(bottleneck_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, return_seg_only=False, return_cls_only=False):
        # Encoder
        x1 = self.attention_unet.inc(x)
        x2 = self.attention_unet.down1(x1)
        x3 = self.attention_unet.down2(x2)
        x4 = self.attention_unet.down3(x3)
        x5 = self.attention_unet.down4(x4)  # Bottleneck
        
        # Classification from bottleneck
        cls_logits = self.classifier(x5)
        
        if return_cls_only:
            return cls_logits
        
        # Segmentation decoder with attention
        x_up = self.attention_unet.up1(x5, x4)
        x_up = self.attention_unet.up2(x_up, x3)
        x_up = self.attention_unet.up3(x_up, x2)
        x_up = self.attention_unet.up4(x_up, x1)
        seg_logits = self.attention_unet.outc(x_up)
        
        if return_seg_only:
            return seg_logits
        
        return seg_logits, cls_logits


if __name__ == "__main__":
    # Test Attention U-Net
    model = AttentionUNet(in_channels=1, out_channels=1, base_filters=64)
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    print(f"Attention U-Net:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test Attention U-Net with classifier
    model_with_cls = AttentionUNetWithClassifier(in_channels=1, num_classes=4, base_filters=64)
    seg_out, cls_out = model_with_cls(x)
    print(f"\nAttention U-Net with Classifier:")
    print(f"Segmentation output: {seg_out.shape}")
    print(f"Classification output: {cls_out.shape}")
    print(f"Parameters: {count_parameters(model_with_cls):,}")

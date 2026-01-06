"""
U-Net architecture for brain tumor segmentation
Based on: https://arxiv.org/abs/1505.04597
Author: Imtiaz Hossain (ID: 23101137)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2D => BatchNorm => ReLU) * 2"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            # First Convolution: 3x3 kernel, padding=1 ensures output size equals input size
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), # Normalizes data for stability
            nn.ReLU(inplace=True),        # Activation function (adds non-linearity)
            # Second Convolution: 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    # Input shape: (1, 1, 256, 256) [DoubleConv(1, 64)]--- > (1, 64, 256, 256) 
    # The image is still 256x256, but now has 64 feature maps.
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

        # Pool -> (1, 64, 256, 256)
        # DoubleConv(64, 128) -> Output (1, 128, 128, 128)
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(Up, self).__init__()
        
        # Use bilinear upsampling or transposed convolution
        # Bilinear interpolation (mathematical resizing)    
        # ConvTranspose2d (learnable resizing).
        # Bilinear is chosen by default here.

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Input from previous layer
            x2: Skip connection from encoder
        """
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

        # x1 comes from below: (1, 1024, 16, 16) -> Upsampled to (1, 1024, 32, 32).
        # x2 comes from the left (skip connection): (1, 512, 32, 32).
        # torch.cat combines them: (1, 1536, 32, 32)
        # DoubleConv shrinks channels back down: Output (1, 512, 32, 32).


class OutConv(nn.Module):
    """Output convolution layer"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for binary segmentation
    Paper: https://arxiv.org/abs/1505.04597
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 64,
        bilinear: bool = True
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Encoder (downsampling path)
        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(base_filters * 8, base_filters * 16 // factor)
        
        # Decoder (upsampling path)
        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.up2 = Up(base_filters * 8, base_filters * 4 // factor, bilinear)
        self.up3 = Up(base_filters * 4, base_filters * 2 // factor, bilinear)
        self.up4 = Up(base_filters * 2, base_filters, bilinear)
        
        # Output layer
        self.outc = OutConv(base_filters, out_channels)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)    # Shape: (1, 64, 256, 256)  <- Saved for skip connection
        x2 = self.down1(x1) # Shape: (1, 128, 128, 128)
        x3 = self.down2(x2) # Shape: (1, 256, 64, 64)
        x4 = self.down3(x3) # Shape: (1, 512, 32, 32)
        x5 = self.down4(x4) # Shape: (1, 1024, 16, 16) <- The "Bottleneck"
        
        # Decoder with skip connections
        x = self.up1(x5, x4) # Upsample x5 (16->32), cat with x4. Out: (1, 512, 32, 32)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x) # 1x1 Conv to squash 64 channels to 1 channel (mask)
        return logits         # Final Shape: (1, 1, 256, 256)
    
    def get_encoder_features(self, x):
        """
        Get encoder features for classification head
        Returns the bottleneck features
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        return x5  # Bottleneck features


class UNetWithClassifier(nn.Module):
    """
    U-Net with classification head attached to encoder
    For joint training or separate training experiments
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_filters: int = 64,
        dropout: float = 0.5
    ):
        super(UNetWithClassifier, self).__init__()
        
        # U-Net for segmentation
        self.unet = UNet(in_channels=in_channels, out_channels=1, base_filters=base_filters)
        
        # Classification head
        # Bottleneck has base_filters * 8 channels = 512 channels
        bottleneck_channels = base_filters * 8
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling  # 512×16×16 → 512×1×1
            nn.Flatten(), # 512×1×1 → 512
            nn.Linear(bottleneck_channels, 512),  # Feature extraction
            nn.ReLU(inplace=True),  # regularization
            nn.Dropout(dropout),    
            nn.Linear(512, 256),  # Reduce dimensions
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes) # Final prediction (e.g., 4 classes) # 4 classes (final output)
        ) 
    
    def forward(self, x, return_seg_only=False, return_cls_only=False):
        # 1. Run the UNet Encoder manually to get the features
        # Get encoder features
        # Encoder
        x1 = self.unet.inc(x)      # (1, 1, 256, 256) → (1, 64, 256, 256)
        x2 = self.unet.down1(x1)   # (1, 64, 256, 256) → (1, 128, 128, 128)
        x3 = self.unet.down2(x2)   # (1, 128, 128, 128) → (1, 256, 64, 64)
        x4 = self.unet.down3(x3)   # (1, 256, 64, 64) → (1, 512, 32, 32)
        x5 = self.unet.down4(x4)   # Bottleneck # → (1, 512, 16, 16)
        
        # 2. Branch A: Classification
        # Classification from bottleneck
        cls_logits = self.classifier(x5) # Predict class based on bottleneck features
        
        if return_cls_only:
            return cls_logits
        
        # The "Bottleneck" (x5) contains the most compressed,
        # abstract understanding of the image.
        # It is the perfect place to attach a classifier to ask "What is in this image?"
        # while the rest of the network asks "Where is it?".


        # 3. Branch B: Segmentation
        # Segmentation decoder
        x_up = self.unet.up1(x5, x4)
        x_up = self.unet.up2(x_up, x3)
        x_up = self.unet.up3(x_up, x2)
        x_up = self.unet.up4(x_up, x1)
        seg_logits = self.unet.outc(x_up)
        
        if return_seg_only:
            return seg_logits
        
        return seg_logits, cls_logits


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test U-Net
    model = UNet(in_channels=1, out_channels=1, base_filters=64)
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test U-Net with classifier
    model_with_cls = UNetWithClassifier(in_channels=1, num_classes=4, base_filters=64)
    seg_out, cls_out = model_with_cls(x)
    print(f"\nU-Net with Classifier:")
    print(f"Segmentation output: {seg_out.shape}")
    print(f"Classification output: {cls_out.shape}")
    print(f"Parameters: {count_parameters(model_with_cls):,}")
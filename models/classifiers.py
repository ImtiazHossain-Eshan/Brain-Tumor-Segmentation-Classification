"""
Classifier architectures for brain tumor classification
Bonus Task 2: Testing multiple classifier architectures
Author: Imtiaz Hossain (ID: 23101137)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetClassifier(nn.Module):
    """
    MobileNetV2-based classifier
    Efficient architecture suitable for resource-constrained environments
    """
    
    def __init__(self, num_classes: int = 4, pretrained: bool = False, dropout: float = 0.5):
        super(MobileNetClassifier, self).__init__()
        
        # Load MobileNetV2
        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        # Modify first conv layer for grayscale input
        original_conv = self.mobilenet.features[0][0]
        self.mobilenet.features[0][0] = nn.Conv2d(
            1,  # Grayscale input
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Replace classifier
        in_features = self.mobilenet.last_channel
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.mobilenet(x)


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 based classifier
    State-of-the-art accuracy with good efficiency
    """
    
    def __init__(self, num_classes: int = 4, pretrained: bool = False, dropout: float = 0.5):
        super(EfficientNetClassifier, self).__init__()
        
        # Load EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        
        # Modify first conv layer for grayscale input
        original_conv = self.efficientnet.features[0][0]
        self.efficientnet.features[0][0] = nn.Conv2d(
            1,  # Grayscale input
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Replace classifier
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.efficientnet(x)


class DenseNetClassifier(nn.Module):
    """
    DenseNet121-based classifier
    Excellent feature reuse with dense connections
    """
    
    def __init__(self, num_classes: int = 4, pretrained: bool = False, dropout: float = 0.5):
        super(DenseNetClassifier, self).__init__()
        
        # Load DenseNet121
        self.densenet = models.densenet121(pretrained=pretrained)
        
        # Modify first conv layer for grayscale input
        original_conv = self.densenet.features.conv0
        self.densenet.features.conv0 = nn.Conv2d(
            1,  # Grayscale input
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Replace classifier
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.densenet(x)


class SimpleClassifier(nn.Module):
    """
    Simple CNN-based classifier for baseline comparison
    """
    
    def __init__(self, num_classes: int = 4, dropout: float = 0.5):
        super(SimpleClassifier, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_classifier(name: str, num_classes: int = 4, pretrained: bool = False, dropout: float = 0.5):
    """
    Factory function to get classifier by name
    
    Args:
        name: One of ['mobilenet', 'efficientnet', 'densenet', 'simple']
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
    
    Returns:
        Classifier model
    """
    name = name.lower()
    
    if name == 'mobilenet':
        return MobileNetClassifier(num_classes, pretrained, dropout)
    elif name == 'efficientnet':
        return EfficientNetClassifier(num_classes, pretrained, dropout)
    elif name == 'densenet':
        return DenseNetClassifier(num_classes, pretrained, dropout)
    elif name == 'simple':
        return SimpleClassifier(num_classes, dropout)
    else:
        raise ValueError(f"Unknown classifier: {name}. Choose from: mobilenet, efficientnet, densenet, simple")


if __name__ == "__main__":
    from .unet import count_parameters
    
    # Test all classifiers
    x = torch.randn(2, 1, 256, 256)
    
    for name in ['mobilenet', 'efficientnet', 'densenet', 'simple']:
        model = get_classifier(name, num_classes=4, pretrained=False)
        out = model(x)
        print(f"\n{name.upper()}:")
        print(f"Output shape: {out.shape}")
        print(f"Parameters: {count_parameters(model):,}")
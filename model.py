"""
CNN model for classifying object proposals.
Uses a pretrained backbone with a custom classifier head.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ProposalClassifier(nn.Module):
    """
    CNN classifier for object proposals.
    Takes cropped proposal regions and classifies them as pothole or background.

    Architecture: Pretrained backbone (ResNet) + custom classifier head
    """

    def __init__(self, num_classes=2, backbone='resnet18', pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes: Number of classes (2 for pothole + background)
            backbone: Backbone architecture ('resnet18', 'resnet34', 'resnet50')
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone weights during training
        """
        super(ProposalClassifier, self).__init__()

        self.num_classes = num_classes

        # Load pretrained backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            logits: Output tensor of shape (B, num_classes)
        """
        return self.backbone(x)

    def get_trainable_params(self):
        """Get parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_backbone_layers(self, num_layers_to_freeze=6):
        """
        Freeze early layers of the backbone.

        Args:
            num_layers_to_freeze: Number of layers to freeze (from the beginning)
        """
        # Get all children modules
        children = list(self.backbone.children())

        # Freeze first num_layers_to_freeze layers
        for i, child in enumerate(children[:num_layers_to_freeze]):
            for param in child.parameters():
                param.requires_grad = False


class SimpleCNN(nn.Module):
    """
    Simple CNN classifier (without pretrained weights).
    Useful for comparison or when pretrained models aren't available.
    """

    def __init__(self, num_classes=2, input_size=224):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(model_type='resnet18', num_classes=2, pretrained=True, freeze_backbone=False):
    """
    Factory function to create models.

    Args:
        model_type: 'resnet18', 'resnet34', 'resnet50', or 'simple'
        num_classes: Number of output classes
        pretrained: Use pretrained weights (for ResNet models)
        freeze_backbone: Freeze backbone weights

    Returns:
        model: PyTorch model
    """
    if model_type == 'simple':
        return SimpleCNN(num_classes=num_classes)
    else:
        return ProposalClassifier(
            num_classes=num_classes,
            backbone=model_type,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")

    # Test ProposalClassifier
    model = ProposalClassifier(num_classes=2, backbone='resnet18', pretrained=True)
    print(f"ProposalClassifier created")

    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test SimpleCNN
    print("\nTesting SimpleCNN...")
    simple_model = SimpleCNN(num_classes=2)
    output = simple_model(x)
    print(f"SimpleCNN output shape: {output.shape}")

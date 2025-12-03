import torch
import torch.nn as nn
from torchvision.models import resnet18

def get_model(num_classes=43):
    """
    Returns a ResNet18 model adapted for GTSRB (32x32 images).
    """
    # Load ResNet18 from scratch
    model = resnet18(weights=None)
    
    # Modify the first convolutional layer to accept 32x32 input without downsampling too much
    # Original: kernel_size=7, stride=2, padding=3
    # New: kernel_size=3, stride=1, padding=1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove the maxpool layer to preserve spatial dimensions
    model.maxpool = nn.Identity()
    
    # Modify the fully connected layer for the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

if __name__ == "__main__":
    # Test the model
    model = get_model()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(f"Output shape: {y.shape}")

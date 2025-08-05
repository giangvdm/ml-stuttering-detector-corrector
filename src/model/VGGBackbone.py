import torch
import torch.nn as nn

class VGGBackbone(nn.Module):
    def __init__(self, input_channels=1):
        """
        VGG-16 style CNN backbone for feature extraction from spectrograms.
        
        Args:
            input_channels: Number of input channels (1 for single spectrograms)
        """
        super(VGGBackbone, self).__init__()
        
        # VGG-16 style architecture with 5 blocks
        self.features = nn.Sequential(
            # Block 1: 128 -> 64 (spatial dimension)
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x301 -> 64x150
            
            # Block 2: 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x150 -> 32x75
            
            # Block 3: 32 -> 16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x75 -> 16x37
            
            # Block 4: 16 -> 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x37 -> 8x18
            
            # Block 5: 8 -> 4
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x18 -> 4x9
        )
        
        # Calculate output size for reshaping
        # After all pooling: 128x301 -> 4x9 with 512 channels
        self.feature_size = 4 * 9 * 512  # 18,432
        
        print(f"VGGBackbone initialized:")
        print(f"  Input channels: {input_channels}")
        print(f"  Expected output feature size: {self.feature_size}")
    
    def forward(self, x):
        """
        Forward pass through VGG backbone.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            features: Flattened features [batch_size, feature_size]
        """
        x = self.features(x)
        
        # Flatten for further processing
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        return x
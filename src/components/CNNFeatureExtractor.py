import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """
    CNN Feature Extraction following the specification:
    
    4 Convolutional Blocks:
    - Block 1: Conv2D(32 filters, 3×3 kernel) → BatchNorm → ReLU → MaxPool2D(2×2)
    - Block 2: Conv2D(64 filters, 3×3 kernel) → BatchNorm → ReLU → MaxPool2D(2×2)  
    - Block 3: Conv2D(128 filters, 3×3 kernel) → BatchNorm → ReLU → MaxPool2D(2×2)
    - Block 4: Conv2D(256 filters, 3×3 kernel) → BatchNorm → ReLU → GlobalAvgPool2D
    
    Input: Log Mel spectrograms (128 mel bands)
    Output: Feature tensor for BiLSTM temporal modeling
    """
    
    def __init__(self):
        super().__init__()
        
        # Block 1: 32 filters
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 2: 64 filters  
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 3: 128 filters
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 4: 256 filters with GlobalAvgPool2D
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # GlobalAvgPool2D
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN blocks.
        
        Args:
            x: Input tensor of shape (batch_size, 1, mel_bands, time_steps)
            
        Returns:
            Feature tensor of shape (batch_size, 256) for classification
        """
        x = self.block1(x)  # (B, 32, H/2, W/2)
        x = self.block2(x)  # (B, 64, H/4, W/4)  
        x = self.block3(x)  # (B, 128, H/8, W/8)
        x = self.block4(x)  # (B, 256, 1, 1)
        
        # Flatten for classification head
        x = x.view(x.size(0), -1)  # (B, 256)
        
        return x
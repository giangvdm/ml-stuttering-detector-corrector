import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Basic residual block with skip connection.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels  
            stride: Stride for convolution (1 for same size, 2 for downsampling)
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection - adjust dimensions if needed
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection
        residual = self.skip_connection(residual)
        out += residual
        out = self.relu(out)
        
        return out

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_dim=1024):
        """
        ResNet-style feature extractor to process VGG backbone output.
        
        Args:
            input_size: Size of flattened VGG features
            hidden_dim: Hidden dimension for processing
        """
        super(ResNetFeatureExtractor, self).__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        
        # Project to 1D sequence for ResNet blocks
        # Reshape flattened features to [batch, channels, sequence_length]
        self.input_projection = nn.Linear(input_size, hidden_dim * 2)
        
        # ResNet blocks for feature refinement
        self.resnet_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim)
        )
        
        # Final projection to get 2x1024 output as specified in architecture
        self.output_projection = nn.Linear(hidden_dim * 2, 2 * 1024)
        
        print(f"ResNetFeatureExtractor initialized:")
        print(f"  Input size: {input_size}")
        print(f"  Hidden dimension: {hidden_dim}")
        print(f"  Output size: {2 * 1024}")
    
    def forward(self, x):
        """
        Forward pass through ResNet feature extractor.
        
        Args:
            x: Flattened VGG features [batch_size, input_size]
            
        Returns:
            features: Processed features [batch_size, 2, 1024] for BiLSTM input
        """
        batch_size = x.size(0)
        
        # Project to higher dimension
        x = self.input_projection(x)  # [batch_size, hidden_dim * 2]
        
        # Reshape for 1D ResNet processing
        x = x.view(batch_size, self.hidden_dim, 2)  # [batch_size, hidden_dim, 2]
        
        # Apply ResNet blocks
        x = self.resnet_blocks(x)  # [batch_size, hidden_dim, 2]
        
        # Reshape back and project to final size
        x = x.view(batch_size, -1)  # [batch_size, hidden_dim * 2]
        x = self.output_projection(x)  # [batch_size, 2048]
        
        # Reshape to [batch_size, 2, 1024] for BiLSTM
        x = x.view(batch_size, 2, 1024)
        
        return x
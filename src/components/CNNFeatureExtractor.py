import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """
    Enhanced CNN Feature Extraction following the same pattern as the original,
    but with improvements based on the study:
    "Stuttering Detection Using LSTM and LSTM-Attention Based Convolutional Neural Network"
    
    Key improvements:
    - 5 Convolutional Blocks (inspired by study's VGG-16 approach) 
    - Enhanced filter progression: 64→128→256→512→512
    - Additional conv layers in deeper blocks for better feature extraction
    - Maintains exact same interface as original
    
    Input: Log Mel spectrograms (128 mel bands)
    Output: Feature tensor (512-dim) for BiLSTM temporal modeling
    """
    
    def __init__(self):
        super().__init__()
        
        # Block 1: 64 filters (enhanced from original 32)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 2: 128 filters (enhanced from original 64)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 3: 256 filters with additional conv layer
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 4: 512 filters with additional conv layer (enhanced from original 256)
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 5: Additional block (new, inspired by study's 5-block architecture)
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # GlobalAvgPool2D
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN blocks.
        
        Args:
            x: Input tensor of shape (batch_size, 1, mel_bands, time_steps)
            
        Returns:
            Feature tensor of shape (batch_size, 512) for temporal modeling
        """
        x = self.block1(x)  # (B, 64, H/2, W/2)
        x = self.block2(x)  # (B, 128, H/4, W/4)  
        x = self.block3(x)  # (B, 256, H/8, W/8)
        x = self.block4(x)  # (B, 512, H/16, W/16)
        x = self.block5(x)  # (B, 512, 1, 1)
        
        # Flatten for classification head
        x = x.view(x.size(0), -1)  # (B, 512)
        
        return x


class EnhancedBiLSTMAttentionModule(nn.Module):
    """
    Enhanced BiLSTM + Attention module based on the study's findings.
    
    Key improvements:
    1. 2-layer BiLSTM (study showed this works better than single layer)
    2. Hybrid attention combining multi-head attention with study's classic approach
    3. Better temporal modeling with additional processing
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Enhanced BiLSTM: 2 layers as recommended by the study
        self.bilstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,      # Study used 2 LSTM layers
            bidirectional=True,
            batch_first=True,
            dropout=dropout if 2 > 1 else 0
        )
        
        # Multi-head attention (keeping your current approach)
        self.multi_head_attention = nn.MultiheadAttention(
            hidden_dim * 2,  # Bidirectional
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Study's classic attention approach (Linear + Tanh + Softmax)
        self.classic_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),  # Study used 256 units
            nn.Tanh(),                       # Study used Tanh activation
            nn.Linear(256, 1)
        )
        
        # Fusion layer to combine both attention mechanisms
        self.attention_fusion = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        
        self.output_dim = hidden_dim * 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced temporal processing with hybrid attention.
        
        Args:
            x: Input features (batch_size, sequence_length, input_dim)
            
        Returns:
            Enhanced temporal features (batch_size, hidden_dim * 2)
        """
        # Enhanced BiLSTM processing (2 layers)
        lstm_out, _ = self.bilstm(x)  # (B, seq_len, hidden_dim*2)
        
        # Multi-head attention approach
        multi_head_out, _ = self.multi_head_attention(lstm_out, lstm_out, lstm_out)
        
        # Study's classic attention approach
        attention_weights = torch.softmax(self.classic_attention(lstm_out), dim=1)
        classic_attended = torch.sum(lstm_out * attention_weights, dim=1, keepdim=True)
        classic_attended = classic_attended.expand_as(multi_head_out)
        
        # Fuse both attention mechanisms
        fused_attention = self.attention_fusion(
            torch.cat([multi_head_out, classic_attended], dim=-1)
        )
        
        # Residual connection and normalization
        enhanced_features = self.layer_norm(lstm_out + self.dropout(fused_attention))
        
        # Global average pooling over sequence dimension
        output = torch.mean(enhanced_features, dim=1)  # (B, hidden_dim*2)
        
        return output


class EnhancedTemporalProcessor(nn.Module):
    """
    Advanced temporal processing module that can be used as an alternative
    to the BiLSTM approach, incorporating study's insights with additional improvements.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # 2-layer BiLSTM (as recommended by study)
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        
        # Temporal convolutions for additional sequence modeling
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True)
        )
        
        # Enhanced attention mechanism
        self.attention = EnhancedAttentionMechanism(
            hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.output_dim = hidden_dim * 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through enhanced temporal modeling.
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)
            
        Returns:
            Processed features (batch_size, hidden_dim * 2)
        """
        # BiLSTM processing
        lstm_out, _ = self.lstm(x)  # (B, seq_len, hidden_dim*2)
        
        # Temporal convolution for additional sequence modeling
        lstm_out_conv = lstm_out.transpose(1, 2)  # (B, features, seq_len)
        conv_out = self.temporal_conv(lstm_out_conv)
        conv_out = conv_out.transpose(1, 2)  # Back to (B, seq_len, features)
        
        # Combine LSTM and conv outputs
        combined = lstm_out + conv_out
        
        # Enhanced attention
        attended = self.attention(combined)
        
        # Global pooling
        output = torch.mean(attended, dim=1)  # (B, hidden_dim*2)
        
        return output


class EnhancedAttentionMechanism(nn.Module):
    """
    Enhanced attention mechanism combining multi-head attention with
    the study's classic attention approach.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Study's attention approach (Linear + Tanh + Softmax)
        self.classic_attention = nn.Sequential(
            nn.Linear(hidden_dim, 256),  # Study used 256 units
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )
        
        # Fusion of both attention mechanisms
        self.attention_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply enhanced attention mechanism.
        
        Args:
            x: Input tensor (batch_size, sequence_length, hidden_dim)
            
        Returns:
            Attention-enhanced features (batch_size, sequence_length, hidden_dim)
        """
        # Multi-head self-attention
        attended, _ = self.self_attention(x, x, x)
        
        # Classic attention (as in the study)
        attention_weights = self.classic_attention(x)
        classic_attended = torch.sum(x * attention_weights, dim=1, keepdim=True)
        classic_attended = classic_attended.expand_as(attended)
        
        # Fuse both attention mechanisms
        fused = self.attention_fusion(torch.cat([attended, classic_attended], dim=-1))
        
        # Residual connection and normalization
        output = self.layer_norm(x + self.dropout(fused))
        
        return output
import torch
import torch.nn as nn
from typing import Dict
from src.components.CNNFeatureExtractor import CNNFeatureExtractor
from src.components.BiLSTMAttentionModule import BiLSTMAttentionModule
from src.components.DysfluencyClassificationHead import DisfluencyClassificationHead


class CNNBiLSTMAttentionModel(nn.Module):
    """
    CNN-BiLSTM-Attention Architecture for Stuttering Detection following the specification:
    
    Core Architecture:
    - Input: Log Mel spectrograms (128 mel bands, 3-second segments)
    - CNN Feature Extraction: 4 convolutional blocks (32→64→128→256 filters)
    - Temporal Modeling: Single BiLSTM layer with 256 hidden units
    - Attention Mechanism: 8-head self-attention for salient segment focus
    - Classification: Dense(128, ReLU) → Dropout(0.3) → Dense(num_classes, Sigmoid)
    
    Expected Performance: 87-90% accuracy, 1.5M parameters
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        lstm_hidden_dim: int = 256,
        attention_heads: int = 8,
        dropout_rate: float = 0.3,
        classification_hidden_dim: int = 128
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # 1. CNN Feature Extraction (4 blocks)
        self.cnn_extractor = CNNFeatureExtractor()
        cnn_output_dim = 256  # From CNNFeatureExtractor
        
        # 2. Reshape CNN output for sequence processing
        # We need to create a sequence from CNN features for LSTM
        # This is a design choice - we'll treat CNN output as single timestep
        self.feature_projection = nn.Linear(cnn_output_dim, lstm_hidden_dim)
        
        # 3. BiLSTM + Attention Temporal Modeling
        self.temporal_model = BiLSTMAttentionModule(
            input_dim=lstm_hidden_dim,
            hidden_dim=lstm_hidden_dim,
            num_heads=attention_heads,
            dropout=dropout_rate
        )
        
        # 4. Classification Head
        # BiLSTM outputs 2*hidden_dim due to bidirectional
        temporal_output_dim = self.temporal_model.output_dim  # 512
        
        self.classification_head = DisfluencyClassificationHead(
            input_dim=temporal_output_dim,
            hidden_dim=classification_hidden_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        # Store architecture info
        self.architecture_info = {
            'cnn_output_dim': cnn_output_dim,
            'lstm_hidden_dim': lstm_hidden_dim,
            'temporal_output_dim': temporal_output_dim,
            'num_classes': num_classes,
            'total_params': self._count_parameters()
        }
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full architecture.
        
        Args:
            x: Input tensor of shape (batch_size, mel_bands, time_steps)
               Expected: Log Mel spectrograms (80 mel bands, ~301 time steps)
            
        Returns:
            Dictionary containing logits and intermediate outputs (compatible with Whisper model format)
        """
        batch_size = x.size(0)
        
        # Add channel dimension for CNN: (B, H, W) -> (B, 1, H, W)
        x = x.unsqueeze(1)  # (B, 1, mel_bands, time_steps)
        
        # 1. CNN Feature Extraction
        cnn_features = self.cnn_extractor(x)  # (B, 256)
        
        # 2. Project and create sequence for LSTM
        # Add sequence dimension for temporal modeling
        projected_features = self.feature_projection(cnn_features)  # (B, lstm_hidden_dim)
        sequence_features = projected_features.unsqueeze(1)  # (B, 1, lstm_hidden_dim)
        
        # 3. Temporal Modeling with BiLSTM + Attention
        temporal_features = self.temporal_model(sequence_features)  # (B, 512)
        
        # 4. Classification
        logits = self.classification_head(temporal_features)  # (B, num_classes)
        
        # Return dictionary format compatible with Whisper model
        return {
            'logits': logits,
            'pooled_features': temporal_features,
            'cnn_features': cnn_features,
            'projected_features': projected_features
        }
    
    def get_model_info(self) -> Dict:
        """Return model architecture information."""
        return self.architecture_info
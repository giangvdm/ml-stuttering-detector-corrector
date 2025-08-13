import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, WhisperConfig
from typing import Optional, Dict, Any
import numpy as np

class StutteringClassifier(nn.Module):
    """
    Whisper-based stuttering classification model following the 'Whisper in Focus' architecture.
    
    Architecture:
    - Pre-trained Whisper encoder (6 layers)
    - Selective layer freezing strategies
    - Projection layer for dimensionality reduction
    - Multi-class or multi-label classification head
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        num_classes: int = 6,
        projection_dim: int = 256,
        dropout_rate: float = 0.1,
        freeze_strategy: str = "base",  # "base", "1", "2"
        multi_label: bool = False  # Multi-label vs single-label classification
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.projection_dim = projection_dim
        self.freeze_strategy = freeze_strategy
        self.multi_label = multi_label
        
        # Load pre-trained Whisper model
        self.whisper = WhisperModel.from_pretrained(model_name)
        
        # Only use the encoder component
        self.encoder = self.whisper.encoder
        
        # Get encoder hidden dimension
        self.encoder_dim = self.encoder.config.d_model  # 512 for base model
        
        # Apply freezing strategy
        self._apply_freezing_strategy(freeze_strategy)
        
        # Projection layer for dimensionality reduction
        self.projector = nn.Linear(self.encoder_dim, projection_dim)
        
        # Classification head - different for multi-label vs single-label
        if multi_label:
            # Multi-label: independent binary classifiers for each disfluency type
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(projection_dim, num_classes)
                # No final activation - will use sigmoid during training
            )
        else:
            # Single-label: standard multi-class classification
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(projection_dim, num_classes)
                # No final activation - will use softmax during inference
            )
        
        # Initialize projection and classification layers
        self._initialize_weights()
    
    def _apply_freezing_strategy(self, strategy: str):
        """
        Apply encoder layer freezing strategies as described in the paper.
        
        Args:
            strategy: "base" (no freezing), "1" (freeze first 3), 
                     "2" (freeze last 3)
        """
        if strategy == "base":
            # No freezing - all layers trainable
            return
        
        elif strategy == "1":
            # Freeze first 3 encoder layers, keep last 3 trainable
            for i, layer in enumerate(self.encoder.layers):
                if i < 3:
                    for param in layer.parameters():
                        param.requires_grad = False
                        
        elif strategy == "2":
            # Freeze last 3 encoder layers, keep first 3 trainable
            for i, layer in enumerate(self.encoder.layers):
                if i >= 3:
                    for param in layer.parameters():
                        param.requires_grad = False
        else:
            raise ValueError(f"Unknown freezing strategy: {strategy}")
    
    def _initialize_weights(self):
        """Initialize projection and classification layers."""
        # Xavier initialization for projection layer
        nn.init.xavier_uniform_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)
        
        # Initialize classification layer
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_features: Log-mel spectrogram features [batch_size, mel_bins, time_steps]
                           Can handle variable time steps (not just 3000)
            
        Returns:
            Dictionary containing logits and hidden states
        """
        # Handle variable input lengths by interpolating to expected size if needed
        batch_size, n_mels, time_steps = input_features.shape
        
        # Whisper encoder expects specific input format
        # For 3-second clips, we need to adapt the input to work with Whisper
        if time_steps != 3000:
            # Interpolate to match Whisper's expected input size
            # This maintains the temporal structure while fitting Whisper's architecture
            target_time_steps = 3000
            
            # Use interpolation to resize time dimension
            input_features = torch.nn.functional.interpolate(
                input_features.unsqueeze(1),  # Add channel dimension
                size=(n_mels, target_time_steps),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # Remove channel dimension
        
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_features, 
            return_dict=True,
            output_hidden_states=True
        )
        
        # Get the last hidden state
        hidden_states = encoder_outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        
        # Global average pooling across time dimension
        pooled_output = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_dim]
        # Aggressive Dropout strategy
        pooled_output = F.dropout(pooled_output, p=0.5, training=self.training)
        
        # Project to lower dimension
        projected = self.projector(pooled_output)  # [batch_size, projection_dim]
        # projected = F.gelu(projected)  # Apply GELU as in the architecture
        projected = F.dropout(projected, p=0.3, training=self.training) # Aggressive Dropout strategy
        
        # Classification
        logits = self.classifier(projected)  # [batch_size, num_classes]
        
        return {
            'logits': logits,
            # 'hidden_states': hidden_states,
            'hidden_states': pooled_output, # Aggressive Dropout strategy
            'projected_features': projected
        }
    
    def get_trainable_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_frozen_parameters(self) -> int:
        """Return the number of frozen parameters."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
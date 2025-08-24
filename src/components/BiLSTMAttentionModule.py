import torch
import torch.nn as nn


class BiLSTMAttentionModule(nn.Module):
    """
    Bidirectional LSTM with Multi-Head Self-Attention following the specification:
    
    Temporal Modeling:
    - Bidirectional LSTM: Single layer with 256 hidden units
    - Return sequences: True (for attention mechanism)
    - Dropout: 0.3 to prevent overfitting
    
    Attention Mechanism:
    - Multi-head self-attention: 8 attention heads
    - Purpose: Focus on salient disfluent segments
    - Implementation: Scaled dot-product attention
    
    Input: Feature sequences from CNN
    Output: Attended features for classification
    """
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Bidirectional LSTM with 256 hidden units
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # BiLSTM outputs 2*hidden_dim due to bidirectional
        lstm_output_dim = hidden_dim * 2
        
        # Multi-head self-attention with 8 heads
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Global average pooling for final representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.output_dim = lstm_output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through BiLSTM and attention mechanism.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Attended feature tensor of shape (batch_size, lstm_output_dim)
        """
        # BiLSTM forward pass
        lstm_out, _ = self.bilstm(x)  # (B, seq_len, hidden_dim*2)
        
        # Self-attention mechanism
        # MultiheadAttention expects (batch, seq_len, embed_dim)
        attended_out, attention_weights = self.attention(
            query=lstm_out,
            key=lstm_out, 
            value=lstm_out
        )  # (B, seq_len, hidden_dim*2)
        
        # Global average pooling across sequence dimension
        # Transpose for pooling: (B, seq_len, hidden_dim*2) -> (B, hidden_dim*2, seq_len)
        pooled = self.global_pool(attended_out.transpose(1, 2))  # (B, hidden_dim*2, 1)
        pooled = pooled.squeeze(-1)  # (B, hidden_dim*2)
        
        return pooled
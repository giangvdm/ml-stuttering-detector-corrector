import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMechanism(nn.Module):
    def __init__(self, input_size=512, attention_size=256):
        """
        Self-attention mechanism for focusing on relevant temporal features.
        
        Args:
            input_size: Size of BiLSTM output (512 for bidirectional with 256 units each)
            attention_size: Size of attention hidden layer (256 as per architecture)
        """
        super(AttentionMechanism, self).__init__()
        
        self.input_size = input_size
        self.attention_size = attention_size
        
        # Attention layers as specified in architecture
        self.attention_linear = nn.Linear(input_size, attention_size)
        self.tanh = nn.Tanh()
        self.context_vector = nn.Linear(attention_size, 1, bias=False)
        
        print(f"AttentionMechanism initialized:")
        print(f"  Input size: {input_size}")
        print(f"  Attention hidden size: {attention_size}")
        print(f"  Output size: {input_size} (same as input)")
    
    def forward(self, lstm_output):
        """
        Apply attention mechanism to BiLSTM output.
        
        Args:
            lstm_output: BiLSTM output [batch_size, sequence_length, input_size]
            
        Returns:
            attended_output: Attention-weighted output [batch_size, input_size]
            attention_weights: Attention weights [batch_size, sequence_length]
        """
        batch_size, seq_len, input_size = lstm_output.size()
        
        # Apply linear transformation with Tanh activation
        # [batch_size, seq_len, input_size] -> [batch_size, seq_len, attention_size]
        attention_hidden = self.tanh(self.attention_linear(lstm_output))
        
        # Compute attention scores
        # [batch_size, seq_len, attention_size] -> [batch_size, seq_len, 1]
        attention_scores = self.context_vector(attention_hidden)
        
        # Remove last dimension and apply softmax
        # [batch_size, seq_len, 1] -> [batch_size, seq_len]
        attention_scores = attention_scores.squeeze(2)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention weights to original input
        # Expand attention weights: [batch_size, seq_len] -> [batch_size, seq_len, 1]
        attention_weights_expanded = attention_weights.unsqueeze(2)
        
        # Element-wise multiplication and sum across sequence dimension
        # [batch_size, seq_len, input_size] * [batch_size, seq_len, 1] -> [batch_size, input_size]
        attended_output = torch.sum(lstm_output * attention_weights_expanded, dim=1)
        
        return attended_output, attention_weights
    
    def get_attention_output(self, lstm_output):
        """
        Get only the attended output without attention weights.
        
        Args:
            lstm_output: BiLSTM output [batch_size, sequence_length, input_size]
            
        Returns:
            attended_output: Attention-weighted output [batch_size, input_size]
        """
        attended_output, _ = self.forward(lstm_output)
        return attended_output
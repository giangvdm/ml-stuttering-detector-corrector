import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, num_layers=2, dropout=0.1):
        """
        Bidirectional LSTM encoder for temporal feature modeling.
        
        Args:
            input_size: Size of input features (1024 from ResNet)
            hidden_size: Hidden size for each LSTM layer (256 units each direction)
            num_layers: Number of LSTM layers (2 as per architecture)
            dropout: Dropout rate between LSTM layers
        """
        super(BiLSTMEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output size is 2 * hidden_size due to bidirectionality
        self.output_size = 2 * hidden_size
        
        print(f"BiLSTMEncoder initialized:")
        print(f"  Input size: {input_size}")
        print(f"  Hidden size per direction: {hidden_size}")
        print(f"  Number of layers: {num_layers}")
        print(f"  Bidirectional output size: {self.output_size}")
        print(f"  Dropout: {dropout}")
    
    def forward(self, x):
        """
        Forward pass through bidirectional LSTM.
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_size]
            
        Returns:
            output: LSTM outputs [batch_size, sequence_length, 2*hidden_size]
            hidden: Final hidden state tuple (h_n, c_n)
        """
        # Initialize hidden states
        batch_size = x.size(0)
        device = x.device
        
        # Hidden state shape: [num_layers * num_directions, batch_size, hidden_size]
        h_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, 
                         device=device, dtype=x.dtype)
        c_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, 
                         device=device, dtype=x.dtype)
        
        # Forward pass through BiLSTM
        output, (h_n, c_n) = self.bilstm(x, (h_0, c_0))
        
        return output, (h_n, c_n)
    
    def get_final_output(self, x):
        """
        Get only the final output without hidden states.
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_size]
            
        Returns:
            output: LSTM outputs [batch_size, sequence_length, 2*hidden_size]
        """
        output, _ = self.forward(x)
        return output
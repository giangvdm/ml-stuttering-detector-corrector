import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, input_size=512, hidden_dim=4096, dropout=0.5):
        """
        Classification head with 3 fully connected layers.
        
        Args:
            input_size: Size of attention output (512 from BiLSTM)
            hidden_dim: Hidden dimension for FC layers (4096 as per architecture)
            dropout: Dropout rate for regularization
        """
        super(ClassificationHead, self).__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        
        # 3 fully connected layers as specified: 4096 -> 4096 -> 1
        self.classifier = nn.Sequential(
            # First FC layer: 512 -> 4096
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Second FC layer: 4096 -> 4096
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Final FC layer: 4096 -> 1 (binary classification)
            nn.Linear(hidden_dim, 1)
            # Note: No sigmoid here - we'll use BCEWithLogitsLoss
        )
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
        
        print(f"ClassificationHead initialized:")
        print(f"  Input size: {input_size}")
        print(f"  Hidden dimension: {hidden_dim}")
        print(f"  Dropout rate: {dropout}")
        print(f"  Output size: 1 (binary classification)")
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through classification head.
        
        Args:
            x: Attention output [batch_size, input_size]
            
        Returns:
            logits: Raw logits [batch_size, 1] (before sigmoid)
        """
        logits = self.classifier(x)
        return logits
    
    def predict_proba(self, x):
        """
        Get probability predictions (with sigmoid).
        
        Args:
            x: Attention output [batch_size, input_size]
            
        Returns:
            probabilities: Sigmoid probabilities [batch_size, 1]
        """
        logits = self.forward(x)
        probabilities = torch.sigmoid(logits)
        return probabilities
    
    def predict(self, x, threshold=0.5):
        """
        Get binary predictions.
        
        Args:
            x: Attention output [batch_size, input_size]
            threshold: Decision threshold (default 0.5)
            
        Returns:
            predictions: Binary predictions [batch_size, 1]
        """
        probabilities = self.predict_proba(x)
        predictions = (probabilities > threshold).float()
        return predictions
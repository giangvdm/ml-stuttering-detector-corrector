import torch.nn as nn

class BiLSTMAttentionClassifier(nn.Module):
    """BiLSTM for dysfluency classification."""
    def __init__(self, input_dim=1024, hidden_dim=256, num_layers=2, num_classes=6):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.3 if num_layers > 1 else 0
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        logits = self.classifier(lstm_out)
        return logits
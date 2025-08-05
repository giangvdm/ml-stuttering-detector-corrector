import torch
import torch.nn as nn
from src.model.VGGBackbone import VGGBackbone
from src.model.ResNetFeatureExtractor import ResNetFeatureExtractor
from src.model.BiLSTMEncoder import BiLSTMEncoder
from src.model.AttentionMechanism import AttentionMechanism
from src.model.ClassificationHead import ClassificationHead

class DysfluencyDetector(nn.Module):
    def __init__(self, input_channels=1, vgg_feature_size=None):
        """
        Complete dysfluency detection model integrating all components.
        
        Args:
            input_channels: Number of input channels for spectrograms (1)
            vgg_feature_size: Size of VGG output features (auto-calculated if None)
        """
        super(DysfluencyDetector, self).__init__()
        
        # Component 1: VGG-style CNN backbone
        self.vgg_backbone = VGGBackbone(input_channels=input_channels)
        
        # Get VGG output size for ResNet input
        if vgg_feature_size is None:
            vgg_feature_size = self.vgg_backbone.feature_size
        
        # Component 2: ResNet feature extractor
        self.resnet_extractor = ResNetFeatureExtractor(
            input_size=vgg_feature_size, 
            hidden_dim=1024
        )
        
        # Component 3: Bidirectional LSTM encoder
        self.bilstm_encoder = BiLSTMEncoder(
            input_size=1024,
            hidden_size=256,
            num_layers=2,
            dropout=0.1
        )
        
        # Component 4: Attention mechanism
        self.attention = AttentionMechanism(
            input_size=512,  # BiLSTM output size (2 * 256)
            attention_size=256
        )
        
        # Component 5: Classification head
        self.classifier = ClassificationHead(
            input_size=512,
            hidden_dim=4096,
            dropout=0.5
        )
        
        print("DisfluencyDetector model initialized successfully!")
        print(f"  Total parameters: {self.count_parameters():,}")
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        """
        Complete forward pass through the model.
        
        Args:
            x: Input spectrograms [batch_size, channels, height, width]
            
        Returns:
            logits: Raw classification logits [batch_size, 1]
        """
        # Step 1: VGG feature extraction
        vgg_features = self.vgg_backbone(x)
        
        # Step 2: ResNet feature processing
        resnet_features = self.resnet_extractor(vgg_features)
        
        # Step 3: BiLSTM temporal modeling
        lstm_output, _ = self.bilstm_encoder(resnet_features)
        
        # Step 4: Attention mechanism
        attended_features, _ = self.attention(lstm_output)
        
        # Step 5: Classification
        logits = self.classifier(attended_features)
        
        return logits
    
    def predict_proba(self, x):
        """Get probability predictions."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def predict(self, x, threshold=0.5):
        """Get binary predictions."""
        probabilities = self.predict_proba(x)
        return (probabilities > threshold).float()
    
    def get_intermediate_outputs(self, x):
        """
        Get intermediate outputs for debugging/analysis.
        
        Returns:
            dict: Dictionary with outputs from each component
        """
        outputs = {}
        
        # VGG features
        vgg_features = self.vgg_backbone(x)
        outputs['vgg_features'] = vgg_features
        
        # ResNet features
        resnet_features = self.resnet_extractor(vgg_features)
        outputs['resnet_features'] = resnet_features
        
        # BiLSTM output
        lstm_output, lstm_hidden = self.bilstm_encoder(resnet_features)
        outputs['lstm_output'] = lstm_output
        outputs['lstm_hidden'] = lstm_hidden
        
        # Attention output and weights
        attended_features, attention_weights = self.attention(lstm_output)
        outputs['attended_features'] = attended_features
        outputs['attention_weights'] = attention_weights
        
        # Final classification
        logits = self.classifier(attended_features)
        outputs['logits'] = logits
        outputs['probabilities'] = torch.sigmoid(logits)
        
        return outputs
    
def create_dysfluency_detector(model_config=None):
    """
    Factory function to create DisfluencyDetector with standard configuration.
    
    Args:
        model_config: Optional dict with custom configuration
        
    Returns:
        model: Configured DisfluencyDetector instance
    """
    default_config = {
        'input_channels': 1,
        'vgg_feature_size': None,  # Auto-calculated
    }
    
    if model_config:
        default_config.update(model_config)
    
    model = DysfluencyDetector(**default_config)
    return model
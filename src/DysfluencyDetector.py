import torch
import torch.nn as nn
from src.model.Wav2Vec2Encoder import *
from src.model.BertLinguisticEncoder import *
from src.model.FeatureFusion import *
from src.model.BiLSTMAttentionClassifier import *

class DysfluencyDetector(nn.Module):
    """Dysfluency Detector System"""
    def __init__(self, num_classes=6, device='cpu'):
        super().__init__()
        self.device = device
        self.acoustic_encoder = Wav2Vec2Encoder().to(device)
        self.linguistic_encoder = BertLinguisticEncoder().to(device)
        self.feature_fusion = FeatureFusion().to(device)
        self.classifier = BiLSTMAttentionClassifier(num_classes=num_classes).to(device)
        
    def forward(self, waveforms, transcripts):
        """Forward pass taking waveforms and pre-computed transcripts."""
        acoustic_features = self.acoustic_encoder(waveforms)
        linguistic_features = self.linguistic_encoder(transcripts, self.device)
        fused_features = self.feature_fusion(acoustic_features, linguistic_features)
        frame_logits = self.classifier(fused_features)
        clip_logits = torch.mean(frame_logits, dim=1)
        return clip_logits
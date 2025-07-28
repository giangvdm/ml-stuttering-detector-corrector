import torch.nn as nn
import torchaudio

class Wav2Vec2Encoder(nn.Module):
    """Acoustic stream using Wav2Vec2.0 for extracting acoustic embeddings."""
    def __init__(self):
        super().__init__()
        self.wav2vec2 = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            
    def forward(self, audio_input):
        features, _ = self.wav2vec2.extract_features(audio_input)
        return features[-1]
import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    """Feature fusion module for concatenating acoustic and linguistic embeddings."""
    def __init__(self, acoustic_dim=768, linguistic_dim=768, fusion_dim=1024):
        super().__init__()
        self.acoustic_proj = nn.Linear(acoustic_dim, fusion_dim // 2)
        self.linguistic_proj = nn.Linear(linguistic_dim, fusion_dim // 2)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(fusion_dim)
        
    def forward(self, acoustic_features, linguistic_features):
        acoustic_proj = self.acoustic_proj(acoustic_features)
        target_len = acoustic_proj.size(1)
        linguistic_upsampled = nn.functional.interpolate(
            linguistic_features.transpose(1, 2), size=target_len, mode='linear'
        ).transpose(1, 2)
        linguistic_proj = self.linguistic_proj(linguistic_upsampled)
        fused_features = torch.cat([acoustic_proj, linguistic_proj], dim=-1)
        fused_features = self.layer_norm(fused_features)
        fused_features = self.dropout(fused_features)
        return fused_features
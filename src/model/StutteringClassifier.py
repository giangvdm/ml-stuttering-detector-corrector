import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from src.components.WhisperLoRAEncoder import LoRAWhisperEncoder, apply_lora_to_whisper


class StutteringClassifier(nn.Module):
    """
    Dysfluency Detection Architecture following the specification:
    
    Frozen Whisper Encoder + LoRA-Adapted Classification Head
    - Raw Audio Input: 3-second clips  
    - Data Augmentation Pipeline: Applied during training
    - FROZEN Whisper Encoder (whisper-small or whisper-base)
    - Temporal Mean Pooling: T × D_hidden → D_hidden
    - LoRA ADAPTED Classification Head
    - Multi-label Dysfluency Classification
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        num_classes: int = 6,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        classification_dropout: float = 0.3,
        num_lora_layers: int = 4
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # 1. FROZEN Whisper Encoder + LoRA Adaptation
        self.encoder = apply_lora_to_whisper(
            model_name=model_name,
            rank=lora_rank,
            alpha=lora_alpha,
            num_layers_from_end=num_lora_layers
        )
        
        # Get hidden dimension from encoder
        self.hidden_dim = self.encoder.hidden_dim  # 512 for base, 768 for small
        
        # 2. LoRA ADAPTED Classification Head
        # Following spec: Linear(D_hidden, 256) + Dropout(0.3) + Linear(256, num_classes)
        self.classification_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.Dropout(classification_dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classification head weights
        self._initialize_classification_head()
        
        # Store architecture info
        self.architecture_info = {
            'encoder_model': model_name,
            'hidden_dim': self.hidden_dim,
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
            'num_classes': num_classes,
            'trainable_params': self.get_trainable_parameters()
        }
    
    def _initialize_classification_head(self):
        """Initialize classification head with Xavier initialization."""
        for module in self.classification_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the architecture.
        
        Args:
            input_features: Log-mel spectrogram features [batch_size, n_mels, time_steps]
            
        Returns:
            Dictionary containing logits and intermediate outputs
        """
        batch_size, n_mels, time_steps = input_features.shape

        if time_steps != 3000:
            # Interpolate to match Whisper's expected input size
            target_time_steps = 3000
            
            # Use interpolation to resize time dimension
            input_features = torch.nn.functional.interpolate(
                input_features.unsqueeze(1),  # Add channel dimension
                size=(n_mels, target_time_steps),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        # 1. Frozen Whisper Encoder with LoRA adaptation
        encoder_outputs = self.encoder(input_features)
        
        # Extract hidden states: [batch_size, seq_len, hidden_dim]
        hidden_states = encoder_outputs['last_hidden_state']
        
        # 2. Temporal Mean Pooling: T × D_hidden → D_hidden
        # This pools across the time dimension to get fixed-size representation
        pooled_features = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_dim]
        
        # 3. LoRA ADAPTED Classification Head
        logits = self.classification_head(pooled_features)  # [batch_size, num_classes]
        
        return {
            'logits': logits,
            'pooled_features': pooled_features,
            'hidden_states': hidden_states,
            'encoder_outputs': encoder_outputs
        }
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get information about trainable parameters."""
        return self.encoder.get_trainable_parameters()
    
    def get_lora_parameters(self) -> Dict[str, torch.Tensor]:
        """Get LoRA parameters for saving/loading."""
        return self.encoder.get_lora_parameters()
    
    def save_lora_weights(self, path: str):
        """Save only LoRA weights (efficient storage)."""
        lora_state = {}
        
        # Save LoRA parameters from encoder
        encoder_lora_state = self.encoder.get_lora_state_dict()
        lora_state.update(encoder_lora_state)
        
        # Save classification head parameters
        for name, param in self.classification_head.named_parameters():
            lora_state[f'classification_head.{name}'] = param.data.clone()
        
        torch.save(lora_state, path)
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights."""
        state_dict = torch.load(path, map_location='cpu')
        
        # Separate encoder and classification head parameters
        encoder_state = {}
        head_state = {}
        
        for name, param in state_dict.items():
            if name.startswith('classification_head.'):
                head_name = name.replace('classification_head.', '')
                head_state[head_name] = param
            else:
                encoder_state[name] = param
        
        # Load encoder LoRA weights
        if encoder_state:
            self.encoder.load_lora_state_dict(encoder_state)
        
        # Load classification head weights
        if head_state:
            # Use strict=True to ensure all parameters are loaded correctly
            missing_keys, unexpected_keys = self.classification_head.load_state_dict(head_state, strict=False)
            if missing_keys:
                print(f"Warning: Missing keys in classification head: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in classification head: {unexpected_keys}")


def create_model(
    model_name: str = "openai/whisper-base",
    num_classes: int = 6,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    classification_dropout: float = 0.3,
    num_lora_layers: int = 4
) -> StutteringClassifier:
    """
    Factory function to create the dysfluency detection model.
    
    Args:
        model_name: Whisper model to use ('openai/whisper-base' or 'openai/whisper-small')
        num_classes: Number of dysfluency classes
        lora_rank: LoRA rank (r=16 as per spec)
        lora_alpha: LoRA alpha scaling (α=32 as per spec)
        classification_dropout: Dropout rate for classification head (0.3 as per spec)
        num_lora_layers: Number of layers from end to apply LoRA (4-6 as per spec)
        
    Returns:
        Configured dysfluency classifier
    """
    model = StutteringClassifier(
        model_name=model_name,
        num_classes=num_classes,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        classification_dropout=classification_dropout,
        num_lora_layers=num_lora_layers
    )
    
    print("Dysfluency Detection Model Created!")
    print("=" * 50)
    print(f"Architecture Summary:")
    print(f"   Encoder: {model_name} (FROZEN)")
    print(f"   LoRA Rank: {lora_rank}, Alpha: {lora_alpha}")
    print(f"   Hidden Dim: {model.hidden_dim}")
    print(f"   Classification: {model.hidden_dim} → 256 → {num_classes}")
    print(f"   Dropout: {classification_dropout}")
    
    # Print parameter efficiency
    param_stats = model.get_trainable_parameters()
    print(f"Parameter Efficiency:")
    print(f"   Total: {param_stats['total_params']:,}")
    print(f"   Trainable: {param_stats['trainable_params']:,}")
    print(f"   Percentage: {param_stats['trainable_percentage']:.2f}%")
    print("=" * 50)
    
    return model


class TrainingConfig:
    """Training configuration matching the architecture specifications."""
    
    def __init__(
        self,
        learning_rate: float = 1e-4,  # As per spec
        batch_size: int = 32,         # As per spec
        max_epochs: int = 50,
        early_stopping_patience: int = 10,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        gradient_clip_norm: float = 1.0,
        augmentation_prob: float = 0.4  # Applied stochastically (p=0.3-0.5)
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.augmentation_prob = augmentation_prob
    
    def get_optimizer(self, model: StutteringClassifier) -> torch.optim.Optimizer:
        """Get optimizer configured for the architecture."""
        # Only optimize trainable parameters (LoRA + classification head)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    def get_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps: int):
        """Get learning rate scheduler with warmup."""
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=self.warmup_steps
        )
        
        # Cosine annealing after warmup
        cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=num_training_steps - self.warmup_steps,
            eta_min=self.learning_rate * 0.1
        )
        
        # Combine warmup + cosine annealing
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps]
        )
        
        return scheduler
    
    def __str__(self) -> str:
        return f"""TrainingConfig:
    Learning Rate: {self.learning_rate}
    Batch Size: {self.batch_size}
    Max Epochs: {self.max_epochs}
    Early Stopping: {self.early_stopping_patience} epochs
    Weight Decay: {self.weight_decay}
    Warmup Steps: {self.warmup_steps}
    Gradient Clipping: {self.gradient_clip_norm}
    Augmentation Prob: {self.augmentation_prob}"""
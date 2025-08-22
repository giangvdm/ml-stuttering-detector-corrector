import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer implementation.
    
    This creates the low-rank matrices A and B that adapt a frozen linear layer.
    The adaptation is: output = frozen_layer(x) + (dropout(x) @ A @ B) * scaling
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float
    ):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA decomposition: adaptation = B @ A where A is in_features × rank, B is rank × out_features
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout for LoRA path
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Initialize LoRA weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA parameters following the paper's recommendations."""
        # Initialize A with random Gaussian, B with zeros (so initial adaptation is zero)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LoRA adaptation: (dropout(x) @ A.T @ B.T) * scaling
        
        Args:
            x: Input tensor [..., in_features]
            
        Returns:
            LoRA adaptation tensor [..., out_features]
        """
        # Apply dropout to input
        x_dropped = self.dropout(x)
        
        # LoRA forward: x @ A.T @ B.T
        # A is [rank, in_features], B is [out_features, rank]
        result = x_dropped @ self.lora_A.T  # [..., rank]
        result = result @ self.lora_B.T     # [..., out_features]
        
        return result * self.scaling


class LoRALinear(nn.Module):
    """
    A linear layer with LoRA adaptation applied to a frozen base layer.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.lora = LoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base_output + lora_adaptation"""
        base_output = self.base_layer(x)
        lora_output = self.lora(x)
        return base_output + lora_output


def replace_linear_with_lora(
    module: nn.Module,
    target_modules: list,
    rank: int,
    alpha: float,
    dropout: float
) -> Dict[str, LoRALinear]:
    """
    Replace specified linear layers in a module with LoRA-adapted versions.
    
    Args:
        module: The module to modify
        target_modules: List of attribute names to replace (e.g., ['q_proj', 'k_proj'])
        rank: LoRA rank
        alpha: LoRA alpha scaling
        dropout: LoRA dropout rate
        
    Returns:
        Dictionary mapping module names to LoRA layers
    """
    lora_layers = {}
    
    for target_name in target_modules:
        if hasattr(module, target_name):
            original_layer = getattr(module, target_name)
            
            if isinstance(original_layer, nn.Linear):
                # Create LoRA adaptation
                lora_layer = LoRALinear(
                    base_layer=original_layer,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )
                
                # Replace the original layer
                setattr(module, target_name, lora_layer)
                lora_layers[target_name] = lora_layer
    
    return lora_layers


class LoRAConfig:
    """Configuration class for LoRA adaptation."""
    
    def __init__(
        self,
        rank: int,
        alpha: float,
        dropout: float,
        target_modules: list = None,
        apply_to_layers: list = None
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        # Default target modules for Whisper attention layers
        if target_modules is None:
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj']
        self.target_modules = target_modules
        
        self.apply_to_layers = apply_to_layers


def get_lora_target_layers(model_name: str, num_layers_from_end: int = 4) -> list:
    """
    Determine which layers to apply LoRA to based on model size.
    
    Args:
        model_name: Whisper model name (e.g., 'whisper-small', 'whisper-base')
        num_layers_from_end: Number of layers from the end to apply LoRA to
        
    Returns:
        List of layer indices to apply LoRA to
    """
    # Whisper model layer counts
    layer_counts = {
        'tiny': 4,
        'base': 6,
        'small': 12,
        'medium': 24,
        'large': 32
    }
    
    # Extract model size from name
    for size, num_layers in layer_counts.items():
        if size in model_name:
            # Return last num_layers_from_end layers
            start_layer = max(0, num_layers - num_layers_from_end)
            return list(range(start_layer, num_layers))
    
    # Default fallback for base model
    return [2, 3, 4, 5]
import torch
import torch.nn as nn
from transformers import WhisperModel
from typing import List, Optional, Dict, Any
from src.components.LoRA import LoRAConfig, get_lora_target_layers, replace_linear_with_lora


class LoRAWhisperEncoder(nn.Module):
    """
    Whisper encoder with LoRA adaptation applied to specified layers.
    
    This module freezes the entire Whisper encoder and only applies LoRA
    adaptation to the last 4-6 layers as specified in the architecture.
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        lora_config: Optional[LoRAConfig] = None
    ):
        super().__init__()
        
        self.model_name = model_name
        
        # Load pre-trained Whisper model and extract ONLY the encoder
        whisper_full = WhisperModel.from_pretrained(model_name)
        self.encoder = whisper_full.encoder
        
        # Remove reference to full model to avoid keeping decoder
        del whisper_full
        
        # Set up LoRA configuration
        if lora_config is None:
            # Default configuration
            target_layers = get_lora_target_layers(model_name, num_layers_from_end=4)
            lora_config = LoRAConfig(
                rank=16,
                alpha=32.0,
                dropout=0.0,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'],
                apply_to_layers=target_layers
            )
        
        self.lora_config = lora_config
        self.lora_layers = {}
        
        # First freeze all parameters
        self._freeze_all_parameters()
        
        # Then apply LoRA adaptation
        self._apply_lora_adaptation()
        
        # Store model dimensions
        self.hidden_dim = self.encoder.config.d_model
        
    def _freeze_all_parameters(self):
        """Freeze all parameters in the Whisper encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def _apply_lora_adaptation(self):
        """Apply LoRA adaptation to specified encoder layers."""
        if self.lora_config.apply_to_layers is None:
            print("Warning: No layers specified for LoRA adaptation")
            return
            
        print(f"Applying LoRA to layers: {self.lora_config.apply_to_layers}")
        print(f"Total encoder layers: {len(self.encoder.layers)}")
        
        for layer_idx in self.lora_config.apply_to_layers:
            if layer_idx < len(self.encoder.layers):
                encoder_layer = self.encoder.layers[layer_idx]
                
                # Apply LoRA to self-attention
                if hasattr(encoder_layer, 'self_attn'):
                    print(f"Adding LoRA to layer {layer_idx} self-attention")
                    
                    # Replace linear layers in attention with LoRA versions
                    lora_modules = replace_linear_with_lora(
                        module=encoder_layer.self_attn,
                        target_modules=self.lora_config.target_modules,
                        rank=self.lora_config.rank,
                        alpha=self.lora_config.alpha,
                        dropout=self.lora_config.dropout
                    )
                    
                    # Store reference to LoRA modules
                    for module_name, lora_layer in lora_modules.items():
                        key = f'layer_{layer_idx}_self_attn_{module_name}'
                        self.lora_layers[key] = lora_layer
                
                else:
                    print(f"Warning: Layer {layer_idx} does not have self_attn attribute")
            else:
                print(f"Warning: Layer index {layer_idx} exceeds available layers")
        
        print(f"Created {len(self.lora_layers)} LoRA adaptations")
    
    def forward(
        self, 
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through LoRA-adapted Whisper encoder.
        
        Args:
            input_features: Log-mel spectrogram features [batch_size, n_mels, time_steps]
            attention_mask: Optional attention mask
            head_mask: Optional head mask
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict
            
        Returns:
            Dictionary containing encoder outputs
        """
        # Forward pass through encoder (LoRA layers are already integrated)
        encoder_outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict if return_dict is not None else True
        )
        
        return encoder_outputs
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """Return the number of trainable parameters (should be ~1-2% of total)."""
        trainable_params = 0
        total_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        return {
            'trainable_params': trainable_params,
            'total_params': total_params,
            'trainable_percentage': (trainable_params / total_params) * 100 if total_params > 0 else 0
        }
    
    def get_lora_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all LoRA parameters."""
        lora_params = {}
        
        for name, param in self.named_parameters():
            if param.requires_grad and ('lora_A' in name or 'lora_B' in name):
                lora_params[name] = param
                
        return lora_params
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict containing only LoRA parameters."""
        lora_state_dict = {}
        
        for name, param in self.named_parameters():
            if param.requires_grad and ('lora_A' in name or 'lora_B' in name):
                lora_state_dict[name] = param.data.clone()
        
        return lora_state_dict
    
    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load LoRA parameters from state dict."""
        for name, param_data in state_dict.items():
            if hasattr(self, name.replace('.', '_')):
                # Navigate to the parameter using the name path
                module = self
                attrs = name.split('.')
                
                for attr in attrs[:-1]:
                    module = getattr(module, attr)
                
                param = getattr(module, attrs[-1])
                param.data.copy_(param_data)


def apply_lora_to_whisper(
    model_name: str = "openai/whisper-base",
    rank: int = 16,
    alpha: float = 32.0,
    num_layers_from_end: int = 4
) -> LoRAWhisperEncoder:
    """
    Convenience function to create a LoRA-adapted Whisper encoder.
    
    Args:
        model_name: Whisper model name
        rank: LoRA rank (r parameter)
        alpha: LoRA alpha scaling parameter  
        num_layers_from_end: Number of layers from end to apply LoRA to
        
    Returns:
        LoRA-adapted Whisper encoder
    """
    target_layers = get_lora_target_layers(model_name, num_layers_from_end)
    
    lora_config = LoRAConfig(
        rank=rank,
        alpha=alpha,
        dropout=0.0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'],
        apply_to_layers=target_layers
    )
    
    return LoRAWhisperEncoder(model_name=model_name, lora_config=lora_config)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor


class MultilabelFocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification tasks.
    
    Based on Lin et al. (2017): "Focal Loss for Dense Object Detection"
    Adapted for multi-label stuttering classification with class balancing.
    
    This implementation addresses:
    1. Class imbalance through alpha weighting
    2. Hard sample mining through gamma focusing
    3. Multi-label prediction confidence calibration
    
    Args:
        alpha: Class balancing factor. Can be:
               - float: same alpha for all classes
               - Tensor: per-class alpha weights [num_classes]
               - None: no class balancing
        gamma: Focusing parameter. Higher values down-weight easy examples.
               Recommended: 2.0 for most cases, 1.0-3.0 range
        reduction: Loss reduction method ('mean', 'sum', 'none')
        pos_weight: Positive class weights (like BCEWithLogitsLoss)
    """
    
    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        pos_weight: Optional[Tensor] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
        
        # Validate parameters
        if gamma < 0:
            raise ValueError(f"Gamma must be non-negative, got {gamma}")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction must be 'mean', 'sum', or 'none', got {reduction}")
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute focal loss for multi-label classification.
        
        Args:
            inputs: Raw logits [batch_size, num_classes]
            targets: Binary labels [batch_size, num_classes] (0 or 1)
            
        Returns:
            Focal loss tensor
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(inputs)
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, 
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        # Compute focal weight: (1 - p_t)^gamma
        # For positive samples: p_t = p, for negative: p_t = 1 - p
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply class balancing (alpha)
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                # Same alpha for all classes
                alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                # Per-class alpha
                alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            alpha_weight = 1.0
        
        # Final focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss that automatically adjusts gamma based on training progress.
    
    Starts with higher gamma (focuses more on hard examples) and gradually
    reduces it as training progresses. Useful for stuttering detection where
    the model needs to learn subtle patterns first.
    
    Args:
        initial_gamma: Starting gamma value (default: 3.0)
        final_gamma: Final gamma value (default: 1.0) 
        decay_steps: Steps to decay from initial to final gamma
        alpha: Class balancing factor
        pos_weight: Positive class weights
    """
    
    def __init__(
        self,
        initial_gamma: float = 3.0,
        final_gamma: float = 1.0,
        decay_steps: int = 1000,
        alpha: Optional[float] = None,
        pos_weight: Optional[Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.initial_gamma = initial_gamma
        self.final_gamma = final_gamma
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.step_count = 0
        
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # Compute current gamma with exponential decay
        progress = min(self.step_count / self.decay_steps, 1.0)
        current_gamma = self.initial_gamma * (1 - progress) + self.final_gamma * progress
        
        # Create focal loss with current gamma
        focal_loss = MultilabelFocalLoss(
            alpha=self.alpha,
            gamma=current_gamma,
            reduction=self.reduction,
            pos_weight=self.pos_weight
        )
        
        self.step_count += 1
        return focal_loss(inputs, targets)


def create_focal_loss_from_bce(
    pos_weight: Optional[Tensor] = None,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25
) -> MultilabelFocalLoss:
    """
    Factory function to create focal loss compatible with existing BCEWithLogitsLoss setup.
    
    This maintains the same pos_weight behavior as your current implementation
    while adding focal loss benefits.
    
    Args:
        pos_weight: Same as BCEWithLogitsLoss pos_weight parameter
        gamma: Focusing parameter (2.0 recommended for stuttering detection)
        alpha: Class balancing (0.25 is standard, None to disable)
        
    Returns:
        Configured MultilabelFocalLoss instance
    """
    return MultilabelFocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction='mean',
        pos_weight=pos_weight
    )
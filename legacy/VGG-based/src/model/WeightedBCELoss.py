import torch
import torch.nn as nn

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=None, reduction='mean'):
        """
        Weighted Binary Cross Entropy Loss for imbalanced datasets.
        
        Args:
            pos_weight: Weight for positive class (calculated from dataset if None)
            reduction: 'mean', 'sum', or 'none'
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
        print(f"WeightedBCELoss initialized:")
        print(f"  Positive weight: {pos_weight}")
        print(f"  Reduction: {reduction}")
    
    def forward(self, logits, targets):
        """
        Compute weighted BCE loss.
        
        Args:
            logits: Model predictions [batch_size, 1] (raw logits)
            targets: Ground truth labels [batch_size, 1] or [batch_size]
            
        Returns:
            loss: Weighted BCE loss
        """
        # Ensure targets have same shape as logits
        if targets.dim() == 1:
            targets = targets.unsqueze(1)
        
        # Use PyTorch's built-in BCEWithLogitsLoss with pos_weight
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )
        
        loss = criterion(logits, targets)
        return loss
    
    def update_pos_weight(self, pos_weight):
        """Update positive weight (useful when changing datasets)."""
        self.pos_weight = pos_weight
        print(f"Updated positive weight to: {pos_weight}")

def calculate_pos_weight_from_dataset(dataset):
    """
    Calculate positive weight from a dataset.
    
    Args:
        dataset: SEP28kDataset instance with target_disfluency set
        
    Returns:
        pos_weight: Tensor with positive weight
    """
    if not dataset.target_disfluency:
        raise ValueError("Dataset must have target_disfluency specified")
    
    # Get class counts
    pos_count = (dataset.df[dataset.target_disfluency] == 1).sum()
    neg_count = (dataset.df[dataset.target_disfluency] == 0).sum()
    
    if pos_count == 0:
        raise ValueError(f"No positive samples found for {dataset.target_disfluency}")
    
    # Calculate weight: neg_count / pos_count
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)
    
    print(f"Calculated pos_weight from dataset:")
    print(f"  Positive samples: {pos_count}")
    print(f"  Negative samples: {neg_count}")
    print(f"  Positive weight: {pos_weight.item():.2f}")
    
    return pos_weight
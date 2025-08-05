import torch
from src.model.WeightedBCELoss import *
from src.model.Dataset import SEP28kDataset

# Load a dataset to calculate weights
train_dataset = SEP28kDataset('./splits/SoundRep/SoundRep_train.csv', target_disfluency='SoundRep')

# Calculate positive weight
pos_weight = calculate_pos_weight_from_dataset(train_dataset)

# Create loss function
criterion = WeightedBCELoss(pos_weight=pos_weight)

# Test with dummy predictions and targets
dummy_logits = torch.randn(4, 1)  # Batch of 4 samples
dummy_targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])  # Mixed labels

loss = criterion(dummy_logits, dummy_targets)
print(f"Dummy loss: {loss.item():.4f}")

from src.model.Dataset import SEP28kDataset
from src.DysfluencyDetector import create_dysfluency_detector
from src.model.DysfluencyTrainer import DysfluencyTrainer

# Load datasets
train_dataset = SEP28kDataset('./splits/SoundRep/SoundRep_train.csv', target_disfluency='SoundRep')
val_dataset = SEP28kDataset('./splits/SoundRep/SoundRep_val.csv', target_disfluency='SoundRep')

# Create model
model = create_dysfluency_detector()

# Create trainer (use 'cpu' if no GPU available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = DysfluencyTrainer(model, train_dataset, val_dataset, device=device)

# Setup optimizer
trainer.setup_optimizer(learning_rate=1e-5)

print("Trainer initialized successfully!")

# Quick test with 2 epochs
print("Starting test training...")
train_losses, val_losses, val_accuracies = trainer.train(num_epochs=2, batch_size=8)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from src.model.WeightedBCELoss import WeightedBCELoss, calculate_pos_weight_from_dataset
from src.model.SpectrogramProcessor import SpectrogramProcessor

class DysfluencyTrainer:
    def __init__(self, model, train_dataset, val_dataset, device='cuda'):
        """
        Trainer class for dysfluency detection models.
        
        Args:
            model: DisfluencyDetector instance
            train_dataset: Training SEP28kDataset
            val_dataset: Validation SEP28kDataset  
            device: 'cuda' or 'cpu'
        """
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        
        # Initialize spectrogram processor
        self.spectrogram_processor = SpectrogramProcessor()
        
        # Calculate and set up weighted loss
        pos_weight = calculate_pos_weight_from_dataset(train_dataset)
        self.criterion = WeightedBCELoss(pos_weight=pos_weight.to(device))
        
        # Training configuration (will be set in train method)
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        print(f"DisfluencyTrainer initialized:")
        print(f"  Device: {device}")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Target disfluency: {train_dataset.target_disfluency}")
    
    def setup_optimizer(self, learning_rate=1e-5, weight_decay=1e-4):
        """Setup optimizer and learning rate scheduler."""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # StepLR scheduler - reduce LR by 0.1 every 20 epochs
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=20, 
            gamma=0.1
        )
        
        print(f"Optimizer configured:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
    
    def process_batch(self, batch):
        """
        Process a batch of audio waveforms to spectrograms.
        
        Args:
            batch: Batch from DataLoader
            
        Returns:
            spectrograms: Processed spectrograms [batch_size, 1, n_mels, time_frames]
            labels: Labels [batch_size, 1]
        """
        waveforms = torch.stack([item['waveform'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        
        # Process to spectrograms
        spectrograms = self.spectrogram_processor.process_batch(waveforms)
        
        # Add channel dimension for CNN: [batch_size, n_mels, time_frames] -> [batch_size, 1, n_mels, time_frames]
        spectrograms = spectrograms.unsqueeze(1)
        
        # Ensure labels have correct shape
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        
        return spectrograms.to(self.device), labels.to(self.device)
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Process batch
            spectrograms, labels = self.process_batch(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(spectrograms)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {avg_loss:.4f}")
        
        return total_loss / num_batches

    def validate_epoch(self, dataloader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Process batch
                spectrograms, labels = self.process_batch(batch)
                
                # Forward pass
                logits = self.model(spectrograms)
                loss = self.criterion(logits, labels)
                
                # Calculate accuracy
                predictions = (torch.sigmoid(logits) > 0.5).float()
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.numel()
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy

    def train(self, num_epochs=30, batch_size=32, patience=10):
        """
        Main training loop with early stopping.
        
        Args:
            num_epochs: Maximum number of epochs
            batch_size: Batch size for training
            patience: Early stopping patience
        """
        if self.optimizer is None:
            self.setup_optimizer()
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,
            collate_fn=lambda x: x  # Return list of dicts
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=lambda x: x
        )
        
        print(f"Starting training:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Early stopping patience: {patience}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_accuracy = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            epoch_time = time.time() - start_time
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            print(f"  LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_checkpoint(f'best_model_{self.train_dataset.target_disfluency}.pth')
                print(f"  New best model saved!")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        print("\nTraining completed!")
        return self.train_losses, self.val_losses, self.val_accuracies

    def save_checkpoint(self, filepath):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'target_disfluency': self.train_dataset.target_disfluency
        }, filepath)
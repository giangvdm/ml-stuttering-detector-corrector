import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_score, recall_score
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import json
from pathlib import Path
from src.components.Dataset import Sep28kDataset

class StutteringDetectorTrainer:
    """
    Training pipeline for Whisper-based stuttering classification.
    
    Implements the training configuration from 'Whisper in Focus' paper:
    - Batch size: 32
    - Learning rate: 2.5e-5
    - Cross-entropy loss
    - Early stopping
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        # learning_rate: float = 2.5e-5, # Recommended by Whisper
        learning_rate: float = 1e-5,
        num_epochs: int = 30,
        patience: int = 3,
        save_dir: str = './models',
        log_dir: str = './logs',
        multi_label: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.multi_label = multi_label
        
        # Create directories
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.05, # Higher weight decay
            eps=1e-8,
            betas=(0.9, 0.98) # Different beta values for stability
        )

        # Setup logging
        self._setup_logging()
        
        # Calculate class weights BEFORE creating loss function
        self.logger.info("Calculating class weights for imbalanced dataset...")
        weights = self._calculate_class_weights(train_loader, multi_label)
        weights = weights.to(device)
        
        # WEIGHTED Loss function depends on single vs multi-label
        if multi_label:
            # For multi-label: use pos_weight parameter
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
        else:
            # For single-label: use weight parameter  
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.3,  # Reduce LR more aggressively
            patience=2,  # Reduce LR faster
            min_lr=1e-6  # Lower minimum LR
        )
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.best_f1 = 0.0
        self.early_stop_counter = 0

    
    def _setup_logging(self):
        """Setup training logger."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.log_dir}/training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _calculate_class_weights(self, train_loader: DataLoader, multi_label: bool = False):
        """
        Calculate class weights for handling imbalanced datasets.
        
        Args:
            train_loader: Training data loader
            multi_label: Whether this is multi-label classification
            
        Returns:
            torch.Tensor: Class weights for loss function
        """
        if multi_label:
            # For multi-label: calculate positive weights for each class
            class_counts = torch.zeros(6)  # 6 classes
            total_samples = 0
            
            for batch in train_loader:
                labels = batch['labels']  # Shape: [batch_size, num_classes]
                class_counts += labels.sum(dim=0)
                total_samples += labels.shape[0]
            
            # Calculate positive class weights: total_samples / (2 * positive_count)
            pos_weights = total_samples / (2 * class_counts)
            
            # Handle edge case where class has no positive samples
            pos_weights = torch.where(class_counts > 0, pos_weights, torch.tensor(1.0))
            
            self.logger.info("Multi-label positive class weights:")
            class_names = ["NoStutter", "WordRep", "SoundRep", "Prolongation", "Interjection", "Block"]
            for i, (name, weight) in enumerate(zip(class_names, pos_weights)):
                self.logger.info(f"  {name}: {weight:.4f} (positive samples: {class_counts[i]:.0f})")
            
            return pos_weights
        
        else:
            # For single-label: calculate class weights
            class_counts = torch.zeros(6)  # 6 classes: 0-5
            
            for batch in train_loader:
                labels = batch['labels'].squeeze()
                for class_idx in range(6):
                    class_counts[class_idx] += (labels == class_idx).sum()
            
            # Calculate inverse frequency weights
            total_samples = class_counts.sum()
            class_weights = total_samples / (6 * class_counts)  # 6 classes
            
            # Handle edge case where class has no samples
            class_weights = torch.where(class_counts > 0, class_weights, torch.tensor(1.0))
            
            self.logger.info("Single-label class weights:")
            class_names = ["NoStutter", "WordRep", "SoundRep", "Prolongation", "Interjection", "Block"]
            for i, (name, weight) in enumerate(zip(class_names, class_weights)):
                self.logger.info(f"  Class {i} ({name}): {weight:.4f} (samples: {class_counts[i]:.0f})")
            
            return class_weights
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            input_features = batch['input_features'].to(self.device)
            
            if self.multi_label:
                labels = batch['labels'].to(self.device)  # [batch_size, num_classes]
            else:
                labels = batch['labels'].squeeze().to(self.device)  # [batch_size]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_features)
            
            loss = self.criterion(outputs['logits'], labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> Tuple[float, float, Dict]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_features = batch['input_features'].to(self.device)
                
                if self.multi_label:
                    labels = batch['labels'].to(self.device)
                else:
                    labels = batch['labels'].squeeze().to(self.device)
                
                outputs = self.model(input_features)
                loss = self.criterion(outputs['logits'], labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Get predictions
                if self.multi_label:
                    # Multi-label: apply sigmoid and threshold at 0.5
                    predictions = torch.sigmoid(outputs['logits']) > 0.5
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                else:
                    # Single-label: argmax
                    predictions = torch.argmax(outputs['logits'], dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Calculate metrics
        if self.multi_label:
            # Multi-label F1 calculation
            f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
            f1_per_class = f1_score(all_labels, all_predictions, average=None)
            
            # Multi-label classification report
            class_names = [
                "NoStutter", "WordRep", "SoundRep", 
                "Prolongation", "Interjection", "Block"
            ]
            
            report = {}
            for i, class_name in enumerate(class_names):
                y_true_class = [row[i] for row in all_labels]
                y_pred_class = [row[i] for row in all_predictions]
                
                if sum(y_true_class) > 0:  # Only calculate if class has positive samples
                    report[class_name] = {
                        'precision': precision_score(y_true_class, y_pred_class, zero_division=0),
                        'recall': recall_score(y_true_class, y_pred_class, zero_division=0),
                        'f1-score': f1_score(y_true_class, y_pred_class, zero_division=0),
                        'support': sum(y_true_class)
                    }
        else:
            # Single-label F1 calculation  
            f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
            f1_per_class = f1_score(all_labels, all_predictions, average=None)
            
            # Single-label classification report
            class_names = [
                "NoStutter", "WordRep", "SoundRep", 
                "Prolongation", "Interjection", "Block"
            ]
            
            report = classification_report(
                all_labels, all_predictions, 
                target_names=class_names, 
                output_dict=True,
                zero_division=0
            )
        
        return avg_loss, f1_weighted, report
    
    def train(self) -> Dict:
        """Complete training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Model has {self.model.get_trainable_parameters():,} trainable parameters")
        self.logger.info(f"Model has {self.model.get_frozen_parameters():,} frozen parameters")
        
        for epoch in range(self.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_f1, report = self.validate()
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)
            
            # Scheduler step
            self.scheduler.step(val_f1)
            
            # Log metrics
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Val Loss: {val_loss:.4f}")
            self.logger.info(f"Val F1 (weighted): {val_f1:.4f}")
            
            # Per-class F1 scores
            for i, class_name in enumerate(["No Stutter", "Word Rep", "Sound Rep", 
                                          "Prolongation", "Interjection", "Block"]):
                if str(i) in report:
                    f1_class = report[str(i)]['f1-score']
                    self.logger.info(f"{class_name} F1: {f1_class:.3f}")
            
            # Early stopping and model saving
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.early_stop_counter = 0
                self._save_model(epoch, val_f1, report)
                self.logger.info(f"New best model saved! F1: {val_f1:.4f}")
            else:
                self.early_stop_counter += 1
                
            if self.early_stop_counter >= self.patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        return {
            'best_f1': self.best_f1,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores
        }
    
    def _save_model(self, epoch: int, f1_score: float, report: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'f1_score': f1_score,
            'classification_report': report,
            'freeze_strategy': self.model.freeze_strategy,
            'num_classes': self.model.num_classes
        }
        
        save_path = f"{self.save_dir}/best_model_f1_{f1_score:.4f}.pth"
        torch.save(checkpoint, save_path)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores
        }
        
        with open(f"{self.log_dir}/training_history.json", 'w') as f:
            json.dump(history, f, indent=2)


def create_data_loaders(
    train_spectrograms: List[np.ndarray],
    train_labels: List,  # Can be List[int] or List[List[int]] for multi-label
    train_ids: List[str],
    val_spectrograms: List[np.ndarray],
    val_labels: List,
    val_ids: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    multi_label: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    
    train_dataset = Sep28kDataset(
        train_spectrograms, train_labels, train_ids, 
        multi_label=multi_label,
        is_training=True
    )
    
    val_dataset = Sep28kDataset(
        val_spectrograms, val_labels, val_ids, 
        multi_label=multi_label
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
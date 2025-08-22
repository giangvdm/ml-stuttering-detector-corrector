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
from src.utils.training_plotter import plot_training_metrics

DYSFLUENT_CLASSES = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']

class StutteringDetectorTrainer:
    """
    Training pipeline for Whisper-based stuttering classification.
    
    Default training configurations:
    - Batch size: 32
    - Learning rate: 1e-4 
    - Cross-entropy loss
    - Early stopping
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        num_epochs: int = 30,
        patience: int = 3,
        save_dir: str = './models',
        log_dir: str = './logs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        
        # Create directories
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Initialize optimizer - only for trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=0.05,
            eps=1e-8,
            betas=(0.9, 0.98)
        )

        # Setup logging
        self._setup_logging()
        
        # Calculate class weights for multi-label classification
        self.logger.info("Calculating class weights for imbalanced dataset...")
        weights = self._calculate_class_weights(train_loader)
        weights = weights.to(device)
        
        # Multi-label loss function only
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.3,
            patience=2,
            min_lr=1e-6
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

    def _calculate_class_weights(self, train_loader) -> torch.Tensor:
        """Calculate class weights for imbalanced multi-label dataset."""
        class_counts = torch.zeros(self.model.num_classes)
        total_samples = 0
        
        for batch in train_loader:
            labels = batch['labels']  # [batch_size, num_classes]
            
            # Count positive samples for each class
            class_counts += labels.sum(dim=0)
            total_samples += labels.shape[0]
        
        # Calculate positive weight for BCEWithLogitsLoss
        # pos_weight = negative_samples / positive_samples
        negative_counts = total_samples - class_counts
        pos_weights = negative_counts / (class_counts + 1e-7)  # Add small epsilon to avoid division by zero
        
        self.logger.info(f"Class distribution (positive samples): {class_counts.tolist()}")
        self.logger.info(f"Calculated pos_weights: {pos_weights.tolist()}")
        
        return pos_weights
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            input_features = batch['input_features'].to(self.device)
            labels = batch['labels'].to(self.device)  # [batch_size, num_classes]
            
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
    
    def validate(self) -> Tuple[float, float, Dict, Dict, any]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_features = batch['input_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_features)
                loss = self.criterion(outputs['logits'], labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Multi-label predictions: apply sigmoid and threshold at 0.5
                predictions = torch.sigmoid(outputs['logits']) > 0.5
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Multi-label F1 calculation
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        f1_per_class = f1_score(all_labels, all_predictions, average=None)
        
        # Multi-label classification report
        report = {}
        for i, class_name in enumerate(DYSFLUENT_CLASSES):
            y_true_class = [row[i] for row in all_labels]
            y_pred_class = [row[i] for row in all_predictions]
            
            if sum(y_true_class) > 0:  # Only calculate if class has positive samples
                report[class_name] = {
                    'precision': precision_score(y_true_class, y_pred_class, zero_division=0),
                    'recall': recall_score(y_true_class, y_pred_class, zero_division=0),
                    'f1-score': f1_score(y_true_class, y_pred_class, zero_division=0),
                    'support': sum(y_true_class)
                }

        # Calculate per-class accuracy and confusion matrix
        per_class_acc = self._calculate_per_class_accuracy(all_labels, all_predictions)
        cm_data = self._calculate_confusion_matrix(all_labels, all_predictions)

        # Add per-class accuracy to the report
        for class_name, accuracy in per_class_acc.items():
            if class_name in report:
                report[class_name]['accuracy'] = accuracy

        return avg_loss, f1_weighted, report, per_class_acc, cm_data
    
    def _calculate_per_class_accuracy(self, all_labels, all_predictions):
        """Calculate per-class accuracy."""
        per_class_acc = {}
        
        for i, class_name in enumerate(DYSFLUENT_CLASSES):
            y_true = np.array([row[i] for row in all_labels])
            y_pred = np.array([row[i] for row in all_predictions])
            
            if len(y_true) > 0:
                accuracy = np.mean(y_true == y_pred)
                per_class_acc[class_name] = accuracy
            else:
                per_class_acc[class_name] = 0.0
        
        return per_class_acc
    
    def _calculate_confusion_matrix(self, all_labels, all_predictions):
        """Calculate confusion matrix for each class."""
        cm_data = {}
        
        for i, class_name in enumerate(DYSFLUENT_CLASSES):
            y_true = np.array([row[i] for row in all_labels])
            y_pred = np.array([row[i] for row in all_predictions])
            
            if len(y_true) > 0 and len(np.unique(y_true)) > 1:
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                cm_data[class_name] = cm
            else:
                cm_data[class_name] = np.array([])
        
        return cm_data
    
    def train(self) -> Dict:
        """Complete training loop."""
        self.logger.info("Starting training...")
        
        # Handle different model types
        if hasattr(self.model, 'get_trainable_parameters'):
            param_info = self.model.get_trainable_parameters()
            if isinstance(param_info, dict):
                trainable_params = param_info.get('trainable_params', 0)
                total_params = param_info.get('total_params', 0)
                frozen_params = total_params - trainable_params
                self.logger.info(f"Model has {trainable_params:,} trainable parameters")
                self.logger.info(f"Model has {frozen_params:,} frozen parameters")
            else:
                # Legacy format - integer
                self.logger.info(f"Model has {param_info:,} trainable parameters")
                if hasattr(self.model, 'get_frozen_parameters'):
                    self.logger.info(f"Model has {self.model.get_frozen_parameters():,} frozen parameters")
        
        for epoch in range(self.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_f1, report, per_class_acc, cm_data = self.validate()
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)
            
            # Scheduler step
            self.scheduler.step(val_f1)
            
            # Log metrics
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Val Loss: {val_loss:.4f}")
            self.logger.info(f"Val F1 (weighted): {val_f1:.4f}")
            
            # Per-class accuracy
            self.logger.info("Per-class accuracy:")
            for class_name in DYSFLUENT_CLASSES:
                accuracy = per_class_acc.get(class_name, 0.0)
                self.logger.info(f"  {class_name}: {accuracy:.3f}")

            # Confusion matrix
            self._log_confusion_matrix(cm_data)

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

        plot_path = Path(self.log_dir) / "training_metrics.png"
        saved_path = plot_training_metrics(self.train_losses, self.val_losses, self.val_f1_scores, str(plot_path))
        self.logger.info(f"Training plot saved: {saved_path}")
        
        return {
            'best_f1': self.best_f1,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores
        }
    
    def _log_confusion_matrix(self, cm_data):
        """Log confusion matrix data."""
        # Multi-label: show binary confusion matrix for each class
        self.logger.info("Confusion matrices (per class):")
        for class_name, cm in cm_data.items():
            if cm.size > 0:
                self.logger.info(f"  {class_name}:")
                self.logger.info(f"    [[TN={cm[0,0]}, FP={cm[0,1]}],")
                self.logger.info(f"     [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    def _save_model(self, epoch: int, f1_score: float, report: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'f1_score': f1_score,
            'classification_report': report,
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
    train_labels: List[List[int]],  # Always multi-label format
    train_ids: List[str],
    val_spectrograms: List[np.ndarray],
    val_labels: List[List[int]],  # Always multi-label format
    val_ids: List[str],
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders for multi-label classification."""
    
    train_dataset = Sep28kDataset(
        train_spectrograms, train_labels, train_ids, 
        is_training=True
    )
    
    val_dataset = Sep28kDataset(
        val_spectrograms, val_labels, val_ids
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
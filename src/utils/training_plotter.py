"""
Simple Training Metrics Plotter for Stuttering Detection
< 100 lines of code for essential training visualization
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List

def plot_training_metrics(
    train_losses: List[float],
    val_losses: List[float], 
    val_f1_scores: List[float],
    save_path: str = "training_metrics.png"
) -> str:
    """
    Create a simple 2x2 subplot showing essential training metrics.
    
    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch
        val_f1_scores: Validation F1 scores per epoch
        save_path: Path to save the plot
        
    Returns:
        Path to saved plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)  # Force integer ticks
    
    # 2. F1 Score progression
    ax2.plot(epochs, val_f1_scores, 'g-', linewidth=2, marker='o')
    ax2.set_title('Validation F1 Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)  # Force integer ticks
    
    # Mark best F1
    best_idx = np.argmax(val_f1_scores)
    best_f1 = val_f1_scores[best_idx]
    ax2.scatter(best_idx + 1, best_f1, color='red', s=100, zorder=5)
    ax2.text(best_idx + 1, best_f1 + 0.05, f'Best: {best_f1:.3f}', 
             ha='center', fontweight='bold')
    
    # 3. Overfitting check (Val - Train loss)
    loss_diff = np.array(val_losses) - np.array(train_losses)
    ax3.plot(epochs, loss_diff, 'purple', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Overfitting Check (Val - Train Loss)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Difference')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(epochs)  # Force integer ticks
    
    # 4. Training summary
    ax4.axis('off')
    summary = f"""Training Summary:
    
Total Epochs: {len(train_losses)}
Final Train Loss: {train_losses[-1]:.4f}
Final Val Loss: {val_losses[-1]:.4f}
Best F1 Score: {max(val_f1_scores):.4f}
Loss Reduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%

Status: {'Converged' if loss_diff[-1] < 0.1 else 'Training'}
"""
    ax4.text(0.1, 0.9, summary, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

# Simple integration for your Trainer class
def add_to_trainer_class():
    """
    Add this method to your StutteringDetectorTrainer class:
    """
    code = '''
def plot_metrics(self):
    """Generate simple training plot."""
    if len(self.train_losses) == 0:
        return None
        
    plot_path = Path(self.log_dir) / "training_metrics.png"
    
    try:
        from src.components.simple_plotter import plot_training_metrics
        saved_path = plot_training_metrics(
            self.train_losses,
            self.val_losses, 
            self.val_f1_scores,
            str(plot_path)
        )
        self.logger.info(f"Training plot saved: {saved_path}")
        return saved_path
    except Exception as e:
        self.logger.error(f"Plotting failed: {e}")
        return None
    '''
    return code

# Call this at the end of your training
def plot_from_trainer(trainer):
    """Simple function to plot from trainer object."""
    return plot_training_metrics(
        trainer.train_losses,
        trainer.val_losses,
        trainer.val_f1_scores,
        "./training_metrics.png"
    )
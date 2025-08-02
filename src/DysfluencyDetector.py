import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from dotenv import load_dotenv
from src.DysfluencyDetector import *
from src.model.Dataset import SEP28kDataset
from src.model.Trainer import *
from collections import defaultdict

def collate_fn(batch):
    """Custom collate function to handle variable length audio and text."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    waveforms = [item['waveform'] for item in batch]
    transcripts = [item['transcript'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    padded_waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    
    return {'waveforms': padded_waveforms, 'transcripts': transcripts, 'labels': labels}

def run_training_loop(trainer, train_loader, val_loader, num_epochs):
    best_val_f1 = 0.0
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        trainer.model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            if batch is None:
                continue
            loss = trainer.train_step(batch)
            train_losses.append(loss)
        
        avg_train_loss = np.mean(train_losses)
        print(f"\n  Average Training Loss: {avg_train_loss:.4f}")
        
        print("  Validating...")
        val_loss, macro_f1, per_class_f1 = trainer.evaluate(val_loader)
        
        print(f"  Validation Loss: {val_loss:.4f}, Validation Macro F1-Score: {macro_f1:.4f}")
        print("  Per-Class F1 Scores:")
        for label, score in per_class_f1.items():
            print(f"    {label}: {score:.4f}")

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            torch.save(trainer.model.state_dict(), 'best_dysfluency_model.pth')
            print(f"  New best model saved! (Validation Macro F1: {macro_f1:.4f})")

def stratified_split_multilabel(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Stratified split for multi-label classification ensuring balanced representation
    of each class across train/val/test splits.
    """
    labels = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']
    
    # Get all labels from dataset
    all_labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        if item is not None:
            all_labels.append(item['label'].numpy())
    
    all_labels = np.array(all_labels)
    
    # For multi-label, we'll use the dominant class (highest value) for stratification
    # or create a composite key from all active labels
    stratification_keys = []
    for label_vec in all_labels:
        # Option 1: Use dominant class
        dominant_class = np.argmax(label_vec)
        stratification_keys.append(dominant_class)
        
        # Option 2: Create composite key from active labels (uncomment if preferred)
        # active_labels = tuple(np.where(label_vec > 0.5)[0])
        # stratification_keys.append(active_labels)
    
    # Group indices by stratification key
    class_indices = defaultdict(list)
    for idx, key in enumerate(stratification_keys):
        class_indices[key].append(idx)
    
    train_indices, val_indices, test_indices = [], [], []
    
    # Split each class proportionally
    for class_key, indices in class_indices.items():
        np.random.shuffle(indices)
        n_samples = len(indices)
        
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        n_test = n_samples - n_train - n_val
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])
    
    # Shuffle the final splits
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset

def print_class_distribution(dataset, dataset_name="Dataset"):
    """Print the distribution of classes in a dataset."""
    labels = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']
    class_counts = defaultdict(int)
    total_samples = 0
    
    for i in range(len(dataset)):
        item = dataset[i]
        if item is not None:
            label_vec = item['label'].numpy()
            dominant_class = np.argmax(label_vec)
            class_counts[labels[dominant_class]] += 1
            total_samples += 1
    
    print(f"\n{dataset_name} class distribution:")
    for label in labels:
        count = class_counts[label]
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  {label}: {count} samples ({percentage:.1f}%)")
    print(f"  Total: {total_samples} samples")

if __name__ == "__main__":
    load_dotenv()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = DysfluencyDetector(num_classes=6, device=device)

    csv_file = os.getenv("TRANSCRIBED_LABEL_FILE_NAME")
    
    if not os.path.exists(csv_file):
        print(f"ERROR: Transcript file not found at '{csv_file}'. Please run the transcription script first.")
    else:
        dataset = SEP28kDataset(csv_file=csv_file)

        # Use stratified split instead of random split
        train_dataset, val_dataset, test_dataset = stratified_split_multilabel(
            dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )
        
        print(f"\nDataset split:")
        print(f"  Training set size: {len(train_dataset)}")
        print(f"  Validation set size: {len(val_dataset)}")
        print(f"  Test set size: {len(test_dataset)}")
        
        # Print class distributions for each split
        print_class_distribution(train_dataset, "Training")
        print_class_distribution(val_dataset, "Validation")
        print_class_distribution(test_dataset, "Test")
        
        # dataset_size = len(dataset)
        # train_size = int(0.8 * dataset_size)
        # val_size = int(0.1 * dataset_size)
        # test_size = dataset_size - train_size - val_size
        
        # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        #     dataset, [train_size, val_size, test_size]
        # )
        
        # print(f"\nDataset split:")
        # print(f"  Training set size: {len(train_dataset)}")
        # print(f"  Validation set size: {len(val_dataset)}")
        # print(f"  Test set size: {len(test_dataset)}")
        
        batch_size = 128
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

        trainer = SEP28kTrainer(model, device=device)

        num_training_epochs = 50
        run_training_loop(trainer, train_dataloader, val_dataloader, num_epochs=num_training_epochs)
        
        print("\nTraining complete.")

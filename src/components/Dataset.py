import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
import logging


class Sep28kDataset(Dataset):
    """
    Dataset class for stuttering classification with support for both single and multi-label formats.
    
    Single-label classes (converted from binary):
    0: No Stuttered Words
    1: Word Repetition  
    2: Sound Repetition
    3: Prolongation
    4: Interjection
    5: Block
    
    Multi-label: Binary vector for each disfluency type
    """
    
    def __init__(
        self,
        spectrograms: List[np.ndarray],
        labels: List,  # Can be List[int] for single-label or List[List[int]] for multi-label
        file_ids: List[str],
        multi_label: bool = False,
        augmentation_prob: float = 0.3
    ):
        self.spectrograms = spectrograms
        self.labels = labels
        self.file_ids = file_ids
        self.multi_label = multi_label
        self.augmentation_prob = augmentation_prob
        
        assert len(spectrograms) == len(labels) == len(file_ids)
        
        # Class mapping
        self.class_names = [
            "No Stuttered Words",
            "Word Repetition", 
            "Sound Repetition",
            "Prolongation",
            "Interjection",
            "Block"
        ]
    
    def __len__(self) -> int:
        return len(self.spectrograms)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spectrogram = self.spectrograms[idx].copy()
        label = self.labels[idx]
        file_id = self.file_ids[idx]
        
        if self.multi_label:
            # Multi-label: return binary vector
            return {
                'input_features': torch.FloatTensor(spectrogram),
                'labels': torch.FloatTensor(label),  # Binary vector for BCEWithLogitsLoss
                'file_id': file_id
            }
        else:
            # Single-label: return class index
            return {
                'input_features': torch.FloatTensor(spectrogram),
                'labels': torch.LongTensor([label]),  # Class index for CrossEntropyLoss
                'file_id': file_id
            }
        
@staticmethod
def stratified_split(
    spectrograms: List[np.ndarray],
    labels: List,
    file_ids: List[str],
    test_size: float = 0.2,
    multi_label: bool = False,
    random_state: int = 42
) -> Tuple[List[np.ndarray], List[np.ndarray], List, List, List[str], List[str]]:
    """
    Perform stratified split for both single-label and multi-label cases.
    
    Args:
        spectrograms: List of spectrogram arrays
        labels: List of labels (int for single-label, List[int] for multi-label)
        file_ids: List of file identifiers
        test_size: Proportion of data for validation set
        multi_label: Whether to use multi-label stratification
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_spectrograms, val_spectrograms, train_labels, val_labels, train_ids, val_ids)
    """
    if multi_label:
        # For multi-label, use iterative stratification
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
            msss = MultilabelStratifiedShuffleSplit(
                n_splits=1, 
                test_size=test_size, 
                random_state=random_state
            )
            
            # Convert labels to numpy array for sklearn compatibility
            labels_array = np.array(labels)
            train_idx, val_idx = next(msss.split(spectrograms, labels_array))
            
        except ImportError:
            logging.warning("iterative-stratification not available. Using random split for multi-label.")
            # Fallback to random split if iterative-stratification is not installed
            from sklearn.model_selection import train_test_split
            train_idx, val_idx = train_test_split(
                range(len(spectrograms)),
                test_size=test_size,
                random_state=random_state
            )
    else:
        # For single-label, use standard stratified split
        train_idx, val_idx = train_test_split(
            range(len(spectrograms)),
            test_size=test_size,
            stratify=labels,
            random_state=random_state
        )
    
    # Split the data
    train_spectrograms = [spectrograms[i] for i in train_idx]
    val_spectrograms = [spectrograms[i] for i in val_idx]
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    train_ids = [file_ids[i] for i in train_idx]
    val_ids = [file_ids[i] for i in val_idx]
    
    # Log split statistics
    if multi_label:
        logging.info("Multi-label stratified split completed")
        # Calculate class distribution for multi-label
        train_counts = np.array(train_labels).sum(axis=0)
        val_counts = np.array(val_labels).sum(axis=0)
        class_names = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection", "NoStutter"]
        
        for i, name in enumerate(class_names):
            total = train_counts[i] + val_counts[i]
            if total > 0:
                logging.info(f"{name}: Train={train_counts[i]} ({train_counts[i]/total*100:.1f}%), "
                           f"Val={val_counts[i]} ({val_counts[i]/total*100:.1f}%)")
    else:
        # Calculate class distribution for single-label
        train_counter = Counter(train_labels)
        val_counter = Counter(val_labels)
        class_names = ["NoStutter", "WordRep", "SoundRep", "Prolongation", "Interjection", "Block"]
        
        logging.info("Single-label stratified split completed:")
        for cls in range(len(class_names)):
            train_count = train_counter.get(cls, 0)
            val_count = val_counter.get(cls, 0)
            total = train_count + val_count
            if total > 0:
                logging.info(f"Class {cls} ({class_names[cls]}): "
                           f"Train={train_count} ({train_count/total*100:.1f}%), "
                           f"Val={val_count} ({val_count/total*100:.1f}%)")
    
    return train_spectrograms, val_spectrograms, train_labels, val_labels, train_ids, val_ids
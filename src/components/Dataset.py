import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional


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
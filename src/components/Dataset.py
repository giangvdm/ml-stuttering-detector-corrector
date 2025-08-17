import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from collections import Counter
import logging
from src.components.DataAugmentation import AudioAugmentation

DYSFLUENT_CLASSES = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']

class Sep28kDataset(Dataset):
    """
    Dataset class for multi-label stuttering classification with augmentation.
    
    Multi-label format: Binary vector for each disfluency type
    """
    
    def __init__(
        self,
        spectrograms: List[np.ndarray],
        labels: List[List[int]],  # Always List[List[int]] for multi-label
        file_ids: List[str],
        augmentation_prob: float = 0.4,  # Recommended (0.3-0.5)
        is_training: bool = False,
        sample_rate: int = 16000
    ):
        self.spectrograms = spectrograms
        self.labels = labels
        self.file_ids = file_ids
        self.augmentation_prob = augmentation_prob
        self.is_training = is_training
        self.augmentator = None
        
        assert len(spectrograms) == len(labels) == len(file_ids)
        
        # Validate that all labels are multi-label format
        for i, label in enumerate(labels[:5]):  # Check first 5 for validation
            if not isinstance(label, (list, np.ndarray)) or len(label) != 6:
                raise ValueError(f"All labels must be binary vectors of length 6. "
                               f"Found label at index {i}: {label}")
        
        # Class mapping
        self.class_names = DYSFLUENT_CLASSES
        
        # Initialize augmentation pipeline during training
        if self.is_training:
            self.augmentator = AudioAugmentation(
                sample_rate=sample_rate,
                apply_prob=augmentation_prob,
                individual_aug_prob=0.5
            )
    
    def __len__(self) -> int:
        return len(self.spectrograms)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spectrogram = self.spectrograms[idx].copy()
        label = self.labels[idx]
        file_id = self.file_ids[idx]

        # Apply augmentation during training only
        if self.is_training:
            if self.augmentator is not None:
                # Use augmentation pipeline
                spectrogram = self._apply_augmentation(spectrogram)
        
        # Multi-label: return binary vector for BCEWithLogitsLoss
        return {
            'input_features': torch.FloatTensor(spectrogram),
            'labels': torch.FloatTensor(label),  # Binary vector for BCEWithLogitsLoss
            'file_id': file_id
        }
    
    def _apply_augmentation(self, spectrogram: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline to spectrogram."""
        return self.augmentator.apply_augmentation(spectrogram=spectrogram)
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in the dataset."""
        class_counts = {class_name: 0 for class_name in self.class_names}
        
        for label in self.labels:
            for i, class_name in enumerate(self.class_names):
                if label[i] == 1:  # Binary label
                    class_counts[class_name] += 1
        
        return class_counts
    
    def get_augmentation_stats(self) -> Dict[str, any]:
        """Get statistics about augmentation configuration."""
        return {
            'augmentation_prob': self.augmentation_prob,
            'is_training': self.is_training,
            'augmentation_available': self.augmentator is not None
        }


    @staticmethod
    def stratified_split(
        spectrograms: List[np.ndarray],
        labels: List[List[int]], # Always multi-label format
        file_ids: List[str],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[int]], List[List[int]], List[str], List[str]]:
        """
        Create stratified split for multi-label classification.
        
        Args:
            spectrograms: List of spectrogram arrays
            labels: List of multi-label binary vectors
            file_ids: List of file identifiers
            test_size: Proportion of data for validation set
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (train_spectrograms, val_spectrograms, train_labels, val_labels, train_ids, val_ids)
        """
        # Use iterative stratification for multi-label
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
            
            logging.info("Using iterative stratification for multi-label split")
            
        except ImportError:
            logging.warning("iterative-stratification not available. Using random split for multi-label.")
            # Fallback to random split if iterative-stratification is not installed
            train_idx, val_idx = train_test_split(
                range(len(spectrograms)),
                test_size=test_size,
                random_state=random_state
            )
        
        # Split the data
        train_spectrograms = [spectrograms[i] for i in train_idx]
        val_spectrograms = [spectrograms[i] for i in val_idx]
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]
        train_ids = [file_ids[i] for i in train_idx]
        val_ids = [file_ids[i] for i in val_idx]
        
        # Log split statistics for multi-label
        logging.info("Multi-label stratified split completed")
        train_counts = np.array(train_labels).sum(axis=0)
        val_counts = np.array(val_labels).sum(axis=0)
        class_names = DYSFLUENT_CLASSES
        
        for i, name in enumerate(class_names):
            total = train_counts[i] + val_counts[i]
            if total > 0:
                logging.info(f"{name}: Train={train_counts[i]} ({train_counts[i]/total*100:.1f}%), "
                        f"Val={val_counts[i]} ({val_counts[i]/total*100:.1f}%)")
        
        return train_spectrograms, val_spectrograms, train_labels, val_labels, train_ids, val_ids
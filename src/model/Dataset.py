import torch
import torchaudio
import pandas as pd
import os

FLUENT_LABEL = 'NoStutteredWords'

class SEP28kDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, target_disfluency=None):
        """
        Args:
            csv_file: Path to CSV
            target_disfluency: Specific disfluency type to train on ('Block', 'Prolongation', etc.)
                            If None, returns all labels (for multi-task approach)
        """
        self.df = pd.read_csv(csv_file)
        self.disfluency_labels = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection']
        self.target_disfluency = target_disfluency
        
        # Validate target_disfluency if specified
        if target_disfluency and target_disfluency not in self.disfluency_labels:
            raise ValueError(f"target_disfluency must be one of {self.disfluency_labels}")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row['filepath']

        # Load and resample audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # Prepare labels based on target_disfluency
        if self.target_disfluency:
            # Binary classification for specific disfluency type
            label = torch.tensor(row[self.target_disfluency], dtype=torch.float32)
        else:
            # Multi-label classification (all disfluency types)
            label = torch.tensor([row[label_name] for label_name in self.disfluency_labels], 
                            dtype=torch.float32)
        
        return {
            'waveform': waveform.squeeze(0),
            'label': label,
            'filepath': audio_path
        }
    
    @staticmethod
    def create_stratified_splits(csv_file, target_disfluency, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """
        Create stratified train/val/test splits maintaining class distribution.
        
        Args:
            csv_file: Path to preprocessed CSV
            target_disfluency: Disfluency type to stratify on
            output_dir: Directory to save split CSV files
            train_ratio, val_ratio, test_ratio: Split ratios (must sum to 1.0)
        
        Returns:
            dict: Paths to train, val, test CSV files
        """
        from sklearn.model_selection import train_test_split
        import os
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Load data
        df = pd.read_csv(csv_file)
        
        print(f"Creating stratified splits for {target_disfluency}")
        print(f"Original dataset size: {len(df)}")
        print(f"Class distribution: {df[target_disfluency].value_counts().to_dict()}")
        
        # First split: train vs (val+test)
        temp_ratio = val_ratio + test_ratio
        X_train, X_temp, y_train, y_temp = train_test_split(
            df, df[target_disfluency], 
            test_size=temp_ratio, 
            stratify=df[target_disfluency], 
            random_state=42
        )
        
        # Second split: val vs test
        test_ratio_adjusted = test_ratio / temp_ratio
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=test_ratio_adjusted,
            stratify=y_temp, 
            random_state=42
        )
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save splits
        splits = {
            'train': X_train,
            'val': X_val, 
            'test': X_test
        }
        
        file_paths = {}
        for split_name, split_df in splits.items():
            filename = f"{target_disfluency}_{split_name}.csv"
            filepath = os.path.join(output_dir, filename)
            split_df.to_csv(filepath, index=False)
            file_paths[split_name] = filepath
            
            # Print split statistics
            pos_count = (split_df[target_disfluency] == 1).sum()
            neg_count = (split_df[target_disfluency] == 0).sum()
            print(f"{split_name.upper()} - Size: {len(split_df)}, Positive: {pos_count}, Negative: {neg_count}")
        
        return file_paths
    
    def apply_upsampling(self):
        """
        Apply upsampling to balance classes.
        Recommended for training set only.
        """
        if not self.target_disfluency:
            raise ValueError("Upsampling only supported for single target disfluency")
        
        # Get class counts
        positive_samples = self.df[self.df[self.target_disfluency] == 1]
        negative_samples = self.df[self.df[self.target_disfluency] == 0]
        
        print(f"Before upsampling - Positive: {len(positive_samples)}, Negative: {len(negative_samples)}")
        
        # Upsample minority class to match majority
        if len(positive_samples) < len(negative_samples):
            upsampled_positive = positive_samples.sample(
                n=len(negative_samples), 
                replace=True, 
                random_state=42  # Fixed seed for reproducibility
            )
            self.df = pd.concat([negative_samples, upsampled_positive], ignore_index=True)
        
        # Shuffle the dataset
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"After upsampling - Total: {len(self.df)}, Distribution: {self.df[self.target_disfluency].value_counts().to_dict()}")

    def get_class_weights(self):
        """
        Calculate class weights for weighted loss function.
        Returns weight for positive class.
        """
        if not self.target_disfluency:
            raise ValueError("Class weights only supported for single target disfluency")
        
        pos_count = (self.df[self.target_disfluency] == 1).sum()
        neg_count = (self.df[self.target_disfluency] == 0).sum()
        
        if pos_count == 0:
            raise ValueError(f"No positive samples found for {self.target_disfluency}")
        
        pos_weight = neg_count / pos_count
        print(f"Class weights for {self.target_disfluency}: pos_weight = {pos_weight:.2f}")
        
        return pos_weight

    def get_dataset_info(self):
        """
        Print dataset statistics for debugging/monitoring.
        """
        print(f"Dataset Info:")
        print(f"  Total samples: {len(self.df)}")
        print(f"  Target disfluency: {self.target_disfluency}")
        
        if self.target_disfluency:
            pos_count = (self.df[self.target_disfluency] == 1).sum()
            neg_count = (self.df[self.target_disfluency] == 0).sum()
            print(f"  Positive samples: {pos_count} ({pos_count/len(self.df)*100:.1f}%)")
            print(f"  Negative samples: {neg_count} ({neg_count/len(self.df)*100:.1f}%)")
        else:
            for label in self.disfluency_labels:
                count = (self.df[label] == 1).sum()
                print(f"  {label}: {count} ({count/len(self.df)*100:.1f}%)")
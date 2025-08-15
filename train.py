#!/usr/bin/env python3
"""
Complete implementation example for Whisper-based stuttering classification.

This script demonstrates how to:
1. Load and preprocess audio data from CSV with binary labels
2. Create the model with different freezing strategies
3. Train and evaluate the model

Usage:
    python train.py --csv_file all_labels.csv --strategy 1
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
import logging
from typing import List, Tuple, Dict
import pandas as pd
from sklearn.metrics import precision_score, recall_score

# Import our modules
from src.model.StutteringClassifier import StutteringClassifier
from src.components.AudioPreprocessor import AudioPreprocessor
from src.components.Trainer import StutteringDetectorTrainer, create_data_loaders
from src.components.Dataset import Sep28kDataset

DYSFLUENT_CLASSES = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']


def load_csv_data(csv_file: str, multi_label: bool = False) -> Tuple[List[str], List, List[str]]:
    """
    Load data from CSV file with binary disfluency labels.
    
    Expected CSV format:
    filepath,Block,Prolongation,SoundRep,WordRep,Interjection,NoStutteredWords
    /path/file.wav,0,1,0,0,0,0
    
    Args:
        csv_file: Path to CSV file with binary labels
        multi_label: If True, treats as multi-label problem; if False, converts to single-label
        
    Returns:
        Tuple of (audio_paths, labels, file_ids)
    """
    if not Path(csv_file).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    # Load CSV
    df = pd.read_csv(csv_file)
    
    # Disfluency columns in order matching the paper
    disfluency_cols = DYSFLUENT_CLASSES
    
    audio_paths = []
    labels = []
    file_ids = []
    
    for _, row in df.iterrows():
        audio_path = row['filepath']
        if Path(audio_path).exists():
            audio_paths.append(audio_path)
            
            if multi_label:
                # Multi-label: return binary vector for each disfluency type
                label_vector = [int(row[col]) for col in disfluency_cols]
                labels.append(label_vector)
            else:
                # Single-label: convert binary to single class index
                label_vector = [int(row[col]) for col in disfluency_cols]
                
                # Find the first positive class (1-hot encoding to class index)
                # Map to class indices: Block=0, Prolongation=1, SoundRep=2, WordRep=3, Interjection=4, NoStutteredWords=5
                single_label = 5  # Default to NoStutteredWords
                for i, val in enumerate(label_vector[:-1]):  # Exclude NoStutteredWords from loop
                    if val == 1:
                        single_label = i
                        break
                
                labels.append(single_label)
            
            # Create file ID from filename
            file_id = Path(audio_path).stem
            file_ids.append(file_id)
        else:
            print(f"Warning: Audio file not found: {audio_path}")
    
    print(f"Loaded {len(audio_paths)} valid audio files")
    print(f"Label format: {'Multi-label' if multi_label else 'Single-label'}")
    
    if multi_label:
        # Print multi-label statistics
        labels_array = np.array(labels)
        for i, class_name in enumerate(DYSFLUENT_CLASSES):
            positive_count = np.sum(labels_array[:, i])
            print(f"  {class_name}: {positive_count} positive samples")
    else:
        # Print single-label statistics
        from collections import Counter
        label_counts = Counter(labels)
        for i, class_name in enumerate(DYSFLUENT_CLASSES):
            count = label_counts.get(i, 0)
            print(f"  Class {i} ({class_name}): {count} samples")
    
    return audio_paths, labels, file_ids


def evaluate_model(model, test_loader, device, multi_label=False):
    """Evaluate model with proper multi-label support."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_features = batch['input_features'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_features)
            
            if multi_label:
                # Multi-label: use sigmoid for probabilities
                probabilities = torch.sigmoid(outputs['logits'])
                predictions = probabilities > 0.5  # Threshold at 0.5
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
            else:
                # Single-label: use softmax + argmax
                probabilities = torch.softmax(outputs['logits'], dim=1)
                predictions = torch.argmax(outputs['logits'], dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    f1_per_class = f1_score(all_labels, all_predictions, average=None)
    
    class_names = DYSFLUENT_CLASSES
    
    if multi_label:
        # Multi-label classification report
        report = {}
        for i, class_name in enumerate(class_names):
            y_true_class = [row[i] for row in all_labels]
            y_pred_class = [row[i] for row in all_predictions]
            
            report[class_name] = {
                'precision': precision_score(y_true_class, y_pred_class, zero_division=0),
                'recall': recall_score(y_true_class, y_pred_class, zero_division=0),
                'f1-score': f1_score(y_true_class, y_pred_class, zero_division=0),
                'support': sum(y_true_class)
            }
        
        # For multi-label, confusion matrix is per-class
        cm = {}
        for i, class_name in enumerate(class_names):
            y_true_class = [row[i] for row in all_labels]
            y_pred_class = [row[i] for row in all_predictions]
            cm[class_name] = confusion_matrix(y_true_class, y_pred_class).tolist()
    else:
        # Single-label classification report
        report = classification_report(
            all_labels, all_predictions, 
            target_names=class_names, 
            output_dict=True
        )
        cm = confusion_matrix(all_labels, all_predictions).tolist()
    
    return {
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }


def main():
    parser = argparse.ArgumentParser(description='Whisper Stuttering Classification')
    parser.add_argument('--csv_file', type=str, help='Path to CSV file with binary labels')
    parser.add_argument('--strategy', type=str, default='base', 
                       choices=['base', '1', '2'],
                       help='Freezing strategy')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2.5e-5, help='Learning rate')
    parser.add_argument('--multi_label', action='store_true',
                       help='Treat as multi-label classification problem')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Multi-label mode: {args.multi_label}")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{args.output_dir}/data_loading.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load data
    if not args.csv_file:
        raise ValueError("--csv_file required")
    
    print(f"Loading SEP-28k data from {args.csv_file}...")
    audio_paths, labels, file_ids = load_csv_data(args.csv_file, multi_label=args.multi_label)
    
    # Process audio files
    preprocessor = AudioPreprocessor()
    print("Processing audio files...")
    
    logging.info("Extracting features...")
    spectrograms = []
    processed_labels = []
    processed_ids = []
    
    for audio_path, label, file_id in zip(audio_paths, labels, file_ids):
        try:
            mel_specs = preprocessor.process_audio_file(audio_path, chunk_duration=3.0)
            spectrograms.extend(mel_specs)
            processed_labels.extend([label] * len(mel_specs))
            processed_ids.extend([f"{file_id}_chunk_{i}" for i in range(len(mel_specs))])
        except Exception as e:
            print(f"Failed to process {audio_path}: {e}")
    
    # Stratified data split
    logging.info("Performing stratified train/validation split...")
    
    # Use stratified split instead of simple percentage split
    train_spectrograms, val_spectrograms, train_labels, val_labels, train_ids, val_ids = \
        Sep28kDataset.stratified_split(
            spectrograms=spectrograms,
            labels=processed_labels,
            file_ids=processed_ids,
            test_size=0.2,  # 20% for validation
            multi_label=args.multi_label,
            random_state=42
        )
    
    print(f"Training samples: {len(train_spectrograms)}")
    print(f"Validation samples: {len(val_spectrograms)}")
    
    # Train single model
    print(f"\nTraining with strategy: {args.strategy}")
    print(f"Multi-label mode: {args.multi_label}")
    
    model = StutteringClassifier(
        freeze_strategy=args.strategy,
        num_classes=6,
        multi_label=args.multi_label
    )
    
    train_loader, val_loader = create_data_loaders(
        train_spectrograms, train_labels, train_ids,
        val_spectrograms, val_labels, val_ids,
        batch_size=args.batch_size,
        multi_label=args.multi_label
    )
    
    trainer = StutteringDetectorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        save_dir=f"{args.output_dir}/models",
        log_dir=f"{args.output_dir}/logs",
        multi_label=args.multi_label
    )
    
    training_results = trainer.train()
    eval_results = evaluate_model(model, val_loader, device, multi_label=args.multi_label)
    
    # Save results   
    with open(f"{args.output_dir}/training_results.json", 'w') as f:
        json.dump({
            'strategy': args.strategy,
            'best_f1': float(training_results['best_f1']),
            'final_f1': float(eval_results['f1_weighted']),
            'trainable_params': model.get_trainable_parameters(),
            'frozen_params': model.get_frozen_parameters()
        }, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation F1: {training_results['best_f1']:.4f}")
    print(f"Final test F1: {eval_results['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
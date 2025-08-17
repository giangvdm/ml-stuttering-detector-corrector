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
from src.model.StutteringClassifier import create_improved_model
from src.components.AudioPreprocessor import AudioPreprocessor
from src.components.Trainer import StutteringDetectorTrainer, create_data_loaders
from src.components.Dataset import Sep28kDataset

DYSFLUENT_CLASSES = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']


def load_csv_data(csv_file: str) -> Tuple[List[str], List[List[int]], List[str]]:
    """
    Load data from CSV file with binary disfluency labels.
    Always returns multi-label format.
    
    Returns:
        Tuple of (audio_paths, labels, file_ids) where labels are List[List[int]]
    """
    if not Path(csv_file).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    
    # Extract binary labels for each disfluency type
    audio_paths = []
    labels = []
    file_ids = []
    
    for _, row in df.iterrows():
        audio_path = row['filepath']
        if Path(audio_path).exists():
            audio_paths.append(audio_path)

            # Create binary label vector for multi-label classification
            label_vector = [int(row[col]) for col in DYSFLUENT_CLASSES]
            labels.append(label_vector)

            # Create file ID from filename
            file_id = Path(audio_path).stem
            file_ids.append(file_id)
        else:
            print(f"Warning: Audio file not found: {audio_path}")

    print(f"Loaded {len(audio_paths)} valid audio files")

    labels_array = np.array(labels)
    for i, class_name in enumerate(DYSFLUENT_CLASSES):
        positive_count = np.sum(labels_array[:, i])
        print(f"  {class_name}: {positive_count} positive samples")
    
    return audio_paths, labels, file_ids


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set - multi-label classification only."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_features = batch['input_features'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_features)
            predictions = torch.sigmoid(outputs['logits']) > 0.5
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1_weighted = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1_weighted
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Train Whisper-based stuttering classifier')
    parser.add_argument('--csv_file', type=str, required=True,
                       help='Path to CSV file with labeled data')
    parser.add_argument('--strategy', type=str, default='base', choices=['base', '1', '2'],
                       help='Freezing strategy: base (no freezing), 1 (freeze first 3), 2 (freeze last 3)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for models and logs')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
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
    audio_paths, labels, file_ids = load_csv_data(args.csv_file)
    
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
            random_state=42
        )
    
    print(f"Training samples: {len(train_spectrograms)}")
    print(f"Validation samples: {len(val_spectrograms)}")
    
    # Train model
    print(f"\nTraining with strategy: {args.strategy}")
    print("Multi-label classification")
    
    model = create_improved_model(
        model_name="openai/whisper-base",
        num_classes=6
    )
    
    train_loader, val_loader = create_data_loaders(
        train_spectrograms, train_labels, train_ids,
        val_spectrograms, val_labels, val_ids,
        batch_size=args.batch_size
    )
    
    trainer = StutteringDetectorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        save_dir=f"{args.output_dir}/models",
        log_dir=f"{args.output_dir}/logs"
    )
    
    training_results = trainer.train()
    eval_results = evaluate_model(model, val_loader, device)
    
    # Save results   
    with open(f"{args.output_dir}/training_results.json", 'w') as f:
        json.dump({
            'strategy': args.strategy,
            'best_f1': float(training_results['best_f1']),
            'final_f1': float(eval_results['f1_weighted']),
            'trainable_params': model.get_trainable_parameters()['trainable_params'],
            'frozen_params': model.get_trainable_parameters()['total_params'] - model.get_trainable_parameters()['trainable_params']
        }, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation F1: {training_results['best_f1']:.4f}")
    print(f"Final test F1: {eval_results['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
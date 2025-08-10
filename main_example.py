#!/usr/bin/env python3
"""
Complete implementation example for Whisper-based stuttering classification.

This script demonstrates how to:
1. Load and preprocess audio data from CSV with binary labels
2. Create the model with different freezing strategies
3. Train and evaluate the model
4. Compare different strategies as in the paper

Usage:
    python main_example.py --csv_file all_labels.csv --strategy strategy1
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
import logging
from typing import List, Tuple, Dict
import pandas as pd

# Import our modules
from whisper_stutter_core import WhisperStutterClassifier
from audio_preprocessing import AudioPreprocessor
from training_pipeline import StutteringTrainer, create_data_loaders


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
    disfluency_cols = ['NoStutteredWords', 'WordRep', 'SoundRep', 'Prolongation', 'Interjection', 'Block']
    
    audio_paths = []
    labels = []
    file_ids = []
    
    for _, row in df.iterrows():
        audio_path = row['filepath']
        if Path(audio_path).exists():
            audio_paths.append(audio_path)
            
            if multi_label:
                # Multi-label: return binary vector for each disfluency type
                label_vector = [row[col] for col in disfluency_cols]
                labels.append(label_vector)
            else:
                # Single-label: find the dominant class (prioritize stuttering over no-stutter)
                stuttering_labels = [row['Block'], row['Prolongation'], row['SoundRep'], 
                                   row['WordRep'], row['Interjection']]
                
                if sum(stuttering_labels) > 0:
                    # Has stuttering - find the first stuttering type
                    if row['WordRep'] == 1:
                        labels.append(1)  # Word Repetition
                    elif row['SoundRep'] == 1:
                        labels.append(2)  # Sound Repetition  
                    elif row['Prolongation'] == 1:
                        labels.append(3)  # Prolongation
                    elif row['Interjection'] == 1:
                        labels.append(4)  # Interjection
                    elif row['Block'] == 1:
                        labels.append(5)  # Block
                    else:
                        labels.append(0)  # Fallback to no stutter
                else:
                    labels.append(0)  # No Stuttered Words
            
            # Extract filename from full path
            filename = Path(audio_path).name
            file_ids.append(filename)
        else:
            logging.warning(f"Audio file not found: {audio_path}")
    
    logging.info(f"Loaded {len(audio_paths)} valid audio files")
    
    if not multi_label:
        # Log class distribution for single-label
        unique, counts = np.unique(labels, return_counts=True)
        class_names = ['NoStutter', 'WordRep', 'SoundRep', 'Prolongation', 'Interjection', 'Block']
        for cls, count in zip(unique, counts):
            logging.info(f"Class {cls} ({class_names[cls]}): {count} samples ({count/len(labels)*100:.1f}%)")
    
    return audio_paths, labels, file_ids


def create_synthetic_data() -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Create synthetic data for demonstration purposes.
    This simulates the structure of processed SEP-28k data.
    """
    np.random.seed(42)
    
    # Simulate mel spectrograms (80 mel bins, ~300 time steps for 3-second clips)
    spectrograms = []
    labels = []
    file_ids = []
    
    # Create balanced synthetic dataset
    samples_per_class = 50
    n_mels, n_frames = 80, 300
    
    for class_idx in range(6):  # 6 disfluency types
        for sample_idx in range(samples_per_class):
            # Generate synthetic mel spectrogram
            spec = np.random.randn(n_mels, n_frames) * 0.5
            
            # Add class-specific patterns (simulating different disfluency signatures)
            if class_idx == 1:  # Word repetition - periodic patterns
                spec[:, ::50] += 2.0
            elif class_idx == 2:  # Sound repetition - higher frequency emphasis
                spec[:20, :] += 1.0
            elif class_idx == 3:  # Prolongation - sustained energy
                spec[20:40, 100:200] += 1.5
            elif class_idx == 4:  # Interjection - brief bursts
                spec[30:50, 50:80] += 2.0
                spec[30:50, 150:180] += 2.0
            elif class_idx == 5:  # Block - silence patterns
                spec[10:70, 50:150] *= 0.1
            
            spectrograms.append(spec)
            labels.append(class_idx)
            file_ids.append(f"synthetic_class{class_idx}_sample{sample_idx}.wav")
    
    return spectrograms, labels, file_ids


def evaluate_model(model: torch.nn.Module, test_loader, device: str) -> Dict:
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_features = batch['input_features'].to(device)
            labels = batch['labels'].squeeze().to(device)
            
            outputs = model(input_features)
            probabilities = torch.softmax(outputs['logits'], dim=1)
            predictions = torch.argmax(outputs['logits'], dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    f1_per_class = f1_score(all_labels, all_predictions, average=None)
    
    class_names = [
        "No Stutter", "Word Rep", "Sound Rep", 
        "Prolongation", "Interjection", "Block"
    ]
    
    report = classification_report(
        all_labels, all_predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    return {
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }


def compare_freezing_strategies(
    train_spectrograms: List[np.ndarray],
    train_labels: List[int],
    train_ids: List[str],
    val_spectrograms: List[np.ndarray],
    val_labels: List[int],
    val_ids: List[str],
    device: str,
    multi_label: bool = False
) -> Dict:
    """
    Compare different freezing strategies as done in the paper.
    
    Returns comparison results for all three strategies.
    """
    strategies = ["base", "strategy1", "strategy2"]
    results = {}
    
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Training with {strategy}")
        print(f"{'='*50}")
        
        # Create model
        model = WhisperStutterClassifier(
            freeze_strategy=strategy,
            num_classes=6,
            multi_label=multi_label
        )
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_spectrograms, train_labels, train_ids,
            val_spectrograms, val_labels, val_ids,
            batch_size=32,
            multi_label=multi_label
        )
        
        # Create trainer
        trainer = StutteringTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=10,  # Reduced for demo
            patience=5,
            multi_label=multi_label
        )
        
        # Train
        training_results = trainer.train()
        
        # Evaluate
        eval_results = evaluate_model(model, val_loader, device)
        
        results[strategy] = {
            'training': training_results,
            'evaluation': eval_results,
            'trainable_params': model.get_trainable_parameters(),
            'frozen_params': model.get_frozen_parameters()
        }
        
        print(f"Strategy {strategy} - Best F1: {training_results['best_f1']:.4f}")
        print(f"Trainable parameters: {model.get_trainable_parameters():,}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Whisper Stuttering Classification')
    parser.add_argument('--csv_file', type=str, help='Path to CSV file with binary labels')
    parser.add_argument('--strategy', type=str, default='base', 
                       choices=['base', 'strategy1', 'strategy2'],
                       help='Freezing strategy')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2.5e-5, help='Learning rate')
    parser.add_argument('--use_synthetic', action='store_true', 
                       help='Use synthetic data for demonstration')
    parser.add_argument('--multi_label', action='store_true',
                       help='Treat as multi-label classification problem')
    parser.add_argument('--compare_strategies', action='store_true',
                       help='Compare all freezing strategies')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--chunk_duration', type=float, default=3.0,
                       help='Audio chunk duration in seconds (3.0 as per Whisper in Focus paper)')
    parser.add_argument('--max_files', type=int, default=10,
                       help='Maximum number of files to process (for testing)')
    
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
    if args.use_synthetic:
        print("Using synthetic data for demonstration...")
        spectrograms, labels, file_ids = create_synthetic_data()
        
        # Split into train/val
        split_idx = int(0.8 * len(spectrograms))
        train_spectrograms = spectrograms[:split_idx]
        train_labels = labels[:split_idx]
        train_ids = file_ids[:split_idx]
        val_spectrograms = spectrograms[split_idx:]
        val_labels = labels[split_idx:]
        val_ids = file_ids[split_idx:]
        
    else:
        if not args.csv_file:
            raise ValueError("--csv_file required when not using synthetic data")
        
        print(f"Loading SEP-28k data from {args.csv_file}...")
        audio_paths, labels, file_ids = load_csv_data(args.csv_file)
        
        # Process audio files
        preprocessor = AudioPreprocessor()
        print("Processing audio files...")
        
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
        
        # Split data (using speaker-exclusive splits as in paper would be better)
        split_idx = int(0.8 * len(spectrograms))
        train_spectrograms = spectrograms[:split_idx]
        train_labels = processed_labels[:split_idx]
        train_ids = processed_ids[:split_idx]
        val_spectrograms = spectrograms[split_idx:]
        val_labels = processed_labels[split_idx:]
        val_ids = processed_ids[split_idx:]
    
    print(f"Training samples: {len(train_spectrograms)}")
    print(f"Validation samples: {len(val_spectrograms)}")
    
    # Compare strategies or train single model
    if args.compare_strategies:
        results = compare_freezing_strategies(
            train_spectrograms, train_labels, train_ids,
            val_spectrograms, val_labels, val_ids,
            device,
            multi_label=args.multi_label
        )
        
        # Save comparison results
        with open(f"{args.output_dir}/strategy_comparison.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for strategy, result in results.items():
                json_results[strategy] = {
                    'best_f1': float(result['training']['best_f1']),
                    'final_f1': float(result['evaluation']['f1_weighted']),
                    'trainable_params': int(result['trainable_params']),
                    'frozen_params': int(result['frozen_params'])
                }
            json.dump(json_results, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("STRATEGY COMPARISON SUMMARY")
        print(f"{'='*60}")
        for strategy, result in results.items():
            print(f"{strategy:>12}: F1={result['evaluation']['f1_weighted']:.4f}, "
                  f"Params={result['trainable_params']:,}")
    
    else:
        # Train single model
        print(f"\nTraining with strategy: {args.strategy}")
        print(f"Multi-label mode: {args.multi_label}")
        
        model = WhisperStutterClassifier(
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
        
        trainer = StutteringTrainer(
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
        eval_results = evaluate_model(model, val_loader, device)
        
        # Save results
        results = {
            'strategy': args.strategy,
            'training': training_results,
            'evaluation': eval_results,
            'model_info': {
                'trainable_params': model.get_trainable_parameters(),
                'frozen_params': model.get_frozen_parameters()
            }
        }
        
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
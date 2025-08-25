#!/usr/bin/env python3
"""
Test CNN-BiLSTM-Attention model on UCLASS dataset.

Usage:
    python test-cnn.py --model_path results_cnn/models/best_model_f1_0.8500.pth --test_csv test_labels.csv
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_score, recall_score
import numpy as np
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import json
from pathlib import Path
import os
import argparse
import pandas as pd
from dotenv import load_dotenv

from src.components.Dataset import Sep28kDataset
from src.components.AudioPreprocessor import AudioPreprocessor
from src.model.CNNBiLSTMAttentionModel import CNNBiLSTMAttentionModel

DYSFLUENT_CLASSES = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']


def load_csv_data(csv_file: str) -> Tuple[List[str], List[List[int]], List[str]]:
    """Load test data from CSV file."""
    if not Path(csv_file).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    
    audio_paths = []
    labels = []
    file_ids = []
    
    for _, row in df.iterrows():
        audio_path = row['filepath']
        if Path(audio_path).exists():
            audio_paths.append(audio_path)
            label_vector = [int(row[col]) for col in DYSFLUENT_CLASSES]
            labels.append(label_vector)
            file_id = Path(audio_path).stem
            file_ids.append(file_id)
        else:
            print(f"Warning: Audio file not found: {audio_path}")

    print(f"Loaded {len(audio_paths)} valid test audio files")
    return audio_paths, labels, file_ids


def load_test_data(annotations_path: str, logger: logging.Logger) -> Tuple[List, List, List]:
    """Load and process test data."""
    # Load data from CSV
    audio_paths, labels, file_ids = load_csv_data(annotations_path)
    
    # Initialize audio preprocessor
    preprocessor = AudioPreprocessor()
    
    # Process audio files to spectrograms
    logger.info("Processing audio files to spectrograms...")
    spectrograms = []
    processed_labels = []
    processed_ids = []
    
    for i, (audio_path, label, file_id) in enumerate(zip(audio_paths, labels, file_ids)):
        try:
            # Process audio file into 3-second chunks (same as training)
            mel_specs = preprocessor.process_audio_file(audio_path, chunk_duration=3.0)
            
            # Ensure each spectrogram is a numpy array
            valid_specs = []
            for spec in mel_specs:
                if isinstance(spec, np.ndarray):
                    valid_specs.append(spec)
                else:
                    logger.warning(f"Skipping non-array spectrogram: {type(spec)}")
            
            if valid_specs:
                spectrograms.extend(valid_specs)
                processed_labels.extend([label] * len(valid_specs))
                processed_ids.extend([f"{file_id}_chunk_{j}" for j in range(len(valid_specs))])
                
        except Exception as e:
            logger.warning(f"Failed to process {audio_path}: {e}")
            continue
        
        # Log progress every 50 files
        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(audio_paths)} files")
    
    logger.info(f"Successfully processed {len(spectrograms)} test samples")
    
    # Log class distribution
    if processed_labels:
        label_array = np.array(processed_labels)
        class_counts = np.sum(label_array, axis=0)
        logger.info("Test dataset class distribution:")
        for i, class_name in enumerate(DYSFLUENT_CLASSES):
            logger.info(f"  {class_name}: {class_counts[i]} samples")
    
    return spectrograms, processed_labels, processed_ids


def evaluate_test_data(model: nn.Module, test_loader: DataLoader, device: str, logger: logging.Logger) -> Dict:
    """Perform inference on test data."""
    all_predictions = []
    all_probabilities = []
    all_true_labels = []
    all_file_ids = []
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            input_features = batch['input_features'].to(device)
            true_labels = batch['labels'].to(device)
            file_ids = batch['file_id']
            
            # Forward pass - CNN-BiLSTM model returns dictionary
            outputs = model(input_features)
            logits = outputs['logits']
            
            # Convert logits to probabilities and predictions
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_true_labels.extend(true_labels.cpu().numpy())
            all_file_ids.extend(file_ids)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    return {
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probabilities), 
        'true_labels': np.array(all_true_labels),
        'file_ids': all_file_ids
    }


def calculate_detailed_metrics(true_labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, logger: logging.Logger) -> Dict:
    """Calculate detailed test metrics."""
    
    # Overall metrics
    accuracy = np.mean(np.all(true_labels == predictions, axis=1))
    
    # Per-class metrics
    f1_per_class = f1_score(true_labels, predictions, average=None, zero_division=0)
    precision_per_class = precision_score(true_labels, predictions, average=None, zero_division=0)
    recall_per_class = recall_score(true_labels, predictions, average=None, zero_division=0)
    
    # Weighted averages
    f1_weighted = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    precision_weighted = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    
    # Create per-class results
    per_class_results = {}
    for i, class_name in enumerate(DYSFLUENT_CLASSES):
        per_class_results[class_name] = {
            'f1': float(f1_per_class[i]),
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i])
        }
    
    return {
        'accuracy': float(accuracy),
        'f1_weighted': float(f1_weighted),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'per_class_metrics': per_class_results
    }


def save_test_results(results: Dict, output_dir: str, logger: logging.Logger):
    """Save test results to file."""
    results_file = f"{output_dir}/test_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Test results saved to: {results_file}")
    
    # Save summary
    summary_file = f"{output_dir}/test_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("CNN-BILSTM-ATTENTION TEST RESULTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model Type: CNN-BiLSTM-Attention\n")
        f.write(f"Test Samples: {results['test_samples']}\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"  F1 (weighted): {results['f1_weighted']:.4f}\n")
        f.write(f"  Precision (weighted): {results['precision_weighted']:.4f}\n")
        f.write(f"  Recall (weighted): {results['recall_weighted']:.4f}\n\n")
        
        f.write("Per-Class Metrics:\n")
        for class_name in DYSFLUENT_CLASSES:
            metrics = results['per_class_metrics'][class_name]
            f.write(f"  {class_name}:\n")
            f.write(f"    F1: {metrics['f1']:.4f}\n")
            f.write(f"    Precision: {metrics['precision']:.4f}\n")
            f.write(f"    Recall: {metrics['recall']:.4f}\n")
    
    logger.info(f"Test summary saved to: {summary_file}")


def setup_test_logging(output_dir: str) -> logging.Logger:
    """Setup logging for test function."""
    # Create output directory first
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('cnn_test_logger')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(f"{output_dir}/test.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description='Test CNN-BiLSTM-Attention stuttering classifier')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_csv', type=str, required=True,
                       help='Path to test CSV file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--output_dir', type=str, default='./test_results_cnn',
                       help='Output directory for test results')
    
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Testing CNN-BiLSTM-Attention Model")
    
    # Setup logging
    logger = setup_test_logging(args.output_dir)
    logger.info("Starting CNN-BiLSTM-Attention model testing...")
    
    # Validate paths
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Test CSV file not found: {args.test_csv}")
    
    # Load and process test data
    logger.info("Loading test dataset...")
    test_spectrograms, test_labels, test_ids = load_test_data(args.test_csv, logger)
    logger.info(f"Test dataset loaded: {len(test_spectrograms)} samples")
    
    # Create test DataLoader
    test_dataset = Sep28kDataset(
        test_spectrograms, test_labels, test_ids, 
        is_training=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    logger.info("Creating CNN-BiLSTM-Attention model...")
    model = CNNBiLSTMAttentionModel(
        num_classes=6,
        lstm_hidden_dim=256,
        attention_heads=8,
        dropout_rate=0.3,
        classification_hidden_dim=128
    )
    
    # Load trained weights
    logger.info(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Perform inference
    logger.info("Performing model inference...")
    test_results = evaluate_test_data(model, test_loader, device, logger)
    
    # Calculate comprehensive metrics
    logger.info("Calculating test metrics...")
    detailed_metrics = calculate_detailed_metrics(
        test_results['true_labels'], 
        test_results['predictions'],
        test_results['probabilities'],
        logger
    )
    
    # Combine results
    final_results = {
        **test_results,
        **detailed_metrics,
        'model_type': 'cnn_bilstm_attention',
        'test_samples': len(test_spectrograms),
        'class_names': DYSFLUENT_CLASSES
    }
    
    # Remove large arrays from final results for JSON serialization
    final_results_json = final_results.copy()
    del final_results_json['predictions']
    del final_results_json['probabilities']
    del final_results_json['true_labels']
    del final_results_json['file_ids']
    
    # Save results
    save_test_results(final_results_json, args.output_dir, logger)
    
    # Print summary
    logger.info("Testing completed successfully!")
    logger.info(f"Model Type: CNN-BiLSTM-Attention")
    logger.info(f"Test F1 (weighted): {final_results['f1_weighted']:.4f}")
    logger.info(f"Test Accuracy: {final_results['accuracy']:.4f}")
    
    print(f"\n{'='*50}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Test Samples: {final_results['test_samples']}")
    print(f"Test F1 (weighted): {final_results['f1_weighted']:.4f}")
    print(f"Test Accuracy: {final_results['accuracy']:.4f}")
    print(f"Test Precision: {final_results['precision_weighted']:.4f}")
    print(f"Test Recall: {final_results['recall_weighted']:.4f}")
    
    print(f"\nPer-class F1 scores:")
    for class_name in DYSFLUENT_CLASSES:
        f1 = final_results['per_class_metrics'][class_name]['f1']
        print(f"  {class_name}: {f1:.4f}")
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
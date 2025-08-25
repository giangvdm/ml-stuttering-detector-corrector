#!/usr/bin/env python3
"""
Test CNN-BiLSTM-Attention model on test dataset.
Replicates the structure and output of test.py script.

Usage:
    python test_cnn.py --model_path results_cnn/models/best_model_f1_0.8500.pth --test_csv test_labels.csv
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_score, recall_score
import numpy as np
from typing import Dict, List, Tuple, Optional
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


def test_model(
    model: nn.Module,
    test_annotations_path: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size: int = 32,
    num_workers: int = 4,
    output_dir: str = './test_results'
) -> Dict:
    """
    Test function to evaluate the trained CNN-BiLSTM-Attention model on a separate test dataset.
    This function matches the structure of test.py exactly.
    
    Args:
        model: Trained CNN-BiLSTM-Attention model
        test_annotations_path: Path to test dataset annotations CSV file
        device: Device for inference ('cuda' or 'cpu')
        batch_size: Batch size for inference
        num_workers: Number of workers for DataLoader
        output_dir: Directory to save test results
        
    Returns:
        Dictionary containing comprehensive test metrics
    """
    # Setup logging
    logger = _setup_test_logging(output_dir)
    logger.info("Starting model testing...")
    
    # Validate paths
    if not os.path.exists(test_annotations_path):
        raise FileNotFoundError(f"Test annotations file not found: {test_annotations_path}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load and process test data
    logger.info("Loading test dataset...")
    test_spectrograms, test_labels, test_ids = _load_test_data(
        test_annotations_path, logger
    )
    
    logger.info(f"Test dataset loaded: {len(test_spectrograms)} samples")
    
    # Create test DataLoader
    test_dataset = Sep28kDataset(
        test_spectrograms, test_labels, test_ids, 
        is_training=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for test
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)
    
    # Perform inference
    logger.info("Performing model inference...")
    test_results = _evaluate_test_data(model, test_loader, device, logger)
    
    # Calculate comprehensive metrics
    logger.info("Calculating test metrics...")
    detailed_metrics = _calculate_detailed_metrics(
        test_results['true_labels'], 
        test_results['predictions'],
        test_results['probabilities'],
        logger
    )
    
    # Combine results
    final_results = {
        **test_results,
        **detailed_metrics,
        'test_samples': len(test_spectrograms),
        'class_names': DYSFLUENT_CLASSES
    }
    
    # Save results
    _save_test_results(final_results, output_dir, logger)
    
    logger.info("Testing completed successfully!")
    logger.info(f"Test F1 (weighted): {final_results['f1_weighted']:.4f}")
    logger.info(f"Test Accuracy: {final_results['accuracy']:.4f}")
    
    return final_results


def _setup_test_logging(output_dir: str) -> logging.Logger:
    """Setup logging for test function."""
    # Create output directory first if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(output_dir) / 'test.log'
    
    logger = logging.getLogger('stuttering_test')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def _load_test_data(
    annotations_path: str, 
    logger: logging.Logger
) -> Tuple[List[np.ndarray], List[List[int]], List[str]]:
    """Load and process test data using the existing data loading functions."""
    
    try:
        # Use the load_csv_data function from train_cnn (changed to match your module name)
        from train_cnn import load_csv_data
        
        # Load data from CSV
        audio_paths, labels, file_ids = load_csv_data(annotations_path)
        
        # Initialize audio preprocessor
        preprocessor = AudioPreprocessor()
        
        # Process audio files to spectrograms
        logger.info("Processing audio files to spectrograms...")
        spectrograms = []
        processed_labels = []
        processed_ids = []
        
        for audio_path, label, file_id in zip(audio_paths, labels, file_ids):
            try:
                # Process audio file into 3-second chunks (same as training)
                mel_specs = preprocessor.process_audio_file(audio_path, chunk_duration=3.0)
                spectrograms.extend(mel_specs)
                processed_labels.extend([label] * len(mel_specs))
                processed_ids.extend([f"{file_id}_chunk_{i}" for i in range(len(mel_specs))])
            except Exception as e:
                logger.warning(f"Failed to process {audio_path}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(spectrograms)} test samples")
        
        # Log class distribution
        label_array = np.array(processed_labels)
        class_counts = np.sum(label_array, axis=0)
        logger.info("Test dataset class distribution:")
        for i, class_name in enumerate(DYSFLUENT_CLASSES):
            logger.info(f"  {class_name}: {class_counts[i]} samples")
        
        return spectrograms, processed_labels, processed_ids
        
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise


def _evaluate_test_data(
    model: nn.Module, 
    test_loader: DataLoader, 
    device: str,
    logger: logging.Logger
) -> Dict:
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
    
    return {
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probabilities), 
        'true_labels': np.array(all_true_labels),
        'file_ids': all_file_ids
    }


def _calculate_detailed_metrics(
    true_labels: np.ndarray, 
    predictions: np.ndarray, 
    probabilities: np.ndarray, 
    logger: logging.Logger
) -> Dict:
    """Calculate detailed test metrics with proper multi-label accuracy."""
    
    # DEBUG: Print shapes and sample data
    logger.info(f"True labels shape: {true_labels.shape}")
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Sample true labels (first 3): {true_labels[:3]}")
    logger.info(f"Sample predictions (first 3): {predictions[:3]}")
    
    # Multi-label accuracy calculations
    # 1. Hamming accuracy (per-label accuracy) - RECOMMENDED for multi-label
    hamming_accuracy = np.mean(true_labels == predictions)
    
    # 2. Exact match accuracy (all labels must match) - Very strict
    exact_match_accuracy = np.mean(np.all(true_labels == predictions, axis=1))
    
    # 3. Jaccard similarity (intersection over union)
    intersection = np.sum((true_labels == 1) & (predictions == 1), axis=1)
    union = np.sum((true_labels == 1) | (predictions == 1), axis=1)
    # Handle cases where union is 0 (no positive labels in either true or pred)
    jaccard_per_sample = np.where(union == 0, 1.0, intersection / union)
    jaccard_accuracy = np.mean(jaccard_per_sample)
    
    # Per-class metrics
    f1_per_class = f1_score(true_labels, predictions, average=None, zero_division=0)
    precision_per_class = precision_score(true_labels, predictions, average=None, zero_division=0)
    recall_per_class = recall_score(true_labels, predictions, average=None, zero_division=0)
    
    # Weighted and macro averages
    f1_weighted = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
    precision_weighted = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    
    # Create per-class results (matching test.py format)
    per_class_f1 = {}
    for i, class_name in enumerate(DYSFLUENT_CLASSES):
        per_class_f1[class_name] = float(f1_per_class[i])
    
    # Log different accuracy metrics for comparison
    logger.info(f"Hamming Accuracy (recommended): {hamming_accuracy:.4f}")
    logger.info(f"Exact Match Accuracy (strict): {exact_match_accuracy:.4f}")
    logger.info(f"Jaccard Accuracy: {jaccard_accuracy:.4f}")
    
    # DEBUG: Check class distribution
    true_positives = np.sum(true_labels, axis=0)
    pred_positives = np.sum(predictions, axis=0)
    logger.info("True positive counts per class:")
    for i, class_name in enumerate(DYSFLUENT_CLASSES):
        logger.info(f"  {class_name}: true={true_positives[i]}, pred={pred_positives[i]}")
    
    return {
        'accuracy': float(hamming_accuracy),  # Use Hamming accuracy as primary metric
        'exact_match_accuracy': float(exact_match_accuracy),  # Keep for comparison
        'jaccard_accuracy': float(jaccard_accuracy),
        'f1_weighted': float(f1_weighted),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'per_class_f1': per_class_f1
    }


def _save_test_results(results: Dict, output_dir: str, logger: logging.Logger):
    """Save test results to files."""
    # Save detailed results as JSON
    results_file = f"{output_dir}/test_results.json"
    
    # Remove large arrays for JSON serialization
    results_json = results.copy()
    for key in ['predictions', 'probabilities', 'true_labels', 'file_ids']:
        if key in results_json:
            del results_json[key]
    
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    logger.info(f"Test results saved to: {results_file}")


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
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Testing CNN-BiLSTM-Attention Model")
    
    # Validate paths
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Test CSV file not found: {args.test_csv}")
    
    # Create model
    print("Creating CNN-BiLSTM-Attention model...")
    model = CNNBiLSTMAttentionModel(
        num_classes=6,
        lstm_hidden_dim=256,
        attention_heads=8,
        dropout_rate=0.3,
        classification_hidden_dim=128
    )
    
    # Load trained weights
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Run test evaluation using the main test_model function
    test_results = test_model(
        model=model,
        test_annotations_path=args.test_csv,
        device=device,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    
    # Print summary - matching test.py format exactly
    print(f"\n{'='*50}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Test Samples: {test_results['test_samples']}")
    print(f"Test F1 (weighted): {test_results['f1_weighted']:.4f}")
    print(f"Test F1 (macro): {test_results['f1_macro']:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test Precision: {test_results['precision_weighted']:.4f}")
    print(f"Test Recall: {test_results['recall_weighted']:.4f}")
    
    print(f"\nPer-class F1 scores:")
    for class_name, f1 in test_results['per_class_f1'].items():
        print(f"  {class_name}: {f1:.4f}")
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
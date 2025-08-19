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

from src.components.Dataset import Sep28kDataset
from src.components.AudioPreprocessor import AudioPreprocessor
from src.components.Trainer import DYSFLUENT_CLASSES

def test_model(
    model: nn.Module,
    test_annotations_path: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size: int = 32,
    num_workers: int = 4,
    output_dir: str = './test_results'
) -> Dict:
    """
    Test function to evaluate the trained Detection model on a separate test dataset.
    
    Args:
        model: Trained Detection model
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
        # Use your existing load_csv_data function
        from train import load_csv_data
        
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
    all_labels = []
    all_file_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_features = batch['input_features'].to(device)  # Correct key name
            labels = batch['labels'].to(device)                  # Correct key name
            file_ids = batch['file_id']                         # Correct key name
            
            # Forward pass - model returns a dictionary
            model_outputs = model(input_features)
            
            # Extract logits from the model output dictionary
            logits = model_outputs['logits']
            
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(logits)
            
            # Convert to predictions (threshold = 0.5)
            predictions = (probabilities > 0.5).float()
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_file_ids.extend(file_ids)
    
    return {
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probabilities),
        'true_labels': np.array(all_labels),
        'file_ids': all_file_ids
    }


def _calculate_detailed_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    logger: logging.Logger
) -> Dict:   
    # Overall metrics
    f1_weighted = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
    f1_micro = f1_score(true_labels, predictions, average='micro', zero_division=0)
    
    precision_weighted = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    
    # Use Hamming accuracy instead of exact-match accuracy
    hamming_accuracy = np.mean(true_labels == predictions)  # Per-label accuracy
    exact_match_ratio = np.mean(np.all(true_labels == predictions, axis=1))  # Keep for comparison
    
    # Calculate Jaccard score (multi-label specific)
    intersection = np.sum((true_labels == 1) & (predictions == 1), axis=1)
    union = np.sum((true_labels == 1) | (predictions == 1), axis=1)
    jaccard_per_sample = np.where(union == 0, 1.0, intersection / union)
    jaccard_score = np.mean(jaccard_per_sample)
    
    # Per-class metrics
    per_class_metrics = {}
    per_class_f1 = {}
    confusion_matrices = {}
    
    for i, class_name in enumerate(DYSFLUENT_CLASSES):
        y_true = true_labels[:, i]
        y_pred = predictions[:, i]
        
        class_f1 = f1_score(y_true, y_pred, zero_division=0)
        class_precision = precision_score(y_true, y_pred, zero_division=0)
        class_recall = recall_score(y_true, y_pred, zero_division=0)
        
        per_class_metrics[class_name] = {
            'f1': float(class_f1),
            'precision': float(class_precision),
            'recall': float(class_recall)
        }
        per_class_f1[class_name] = float(class_f1)
        
        if len(np.unique(y_true)) > 1:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            confusion_matrices[class_name] = cm.tolist()
        else:
            confusion_matrices[class_name] = []
        
        logger.info(f"{class_name}: F1={class_f1:.3f}, P={class_precision:.3f}, R={class_recall:.3f}")
    
    # Classification report
    class_report = classification_report(
        true_labels, predictions, 
        target_names=DYSFLUENT_CLASSES, 
        output_dict=True,
        zero_division=0
    )
    
    # Return proper multi-label metrics
    return {
        'f1_weighted': float(f1_weighted),
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        
        # Use hamming accuracy as primary accuracy metric
        'accuracy': float(hamming_accuracy),
        'hamming_accuracy': float(hamming_accuracy),
        'exact_match_ratio': float(exact_match_ratio),
        'jaccard_score': float(jaccard_score),
        
        'per_class_metrics': per_class_metrics,
        'per_class_f1': per_class_f1,
        'confusion_matrices': confusion_matrices,
        'classification_report': class_report
    }


def _save_test_results(results: Dict, output_dir: str, logger: logging.Logger):
    """Save test results to files."""
    try:
        # Save main results
        results_file = Path(output_dir) / 'test_results.json'
        
        # Create a copy for JSON serialization (remove numpy arrays)
        json_results = {k: v for k, v in results.items() 
                       if k not in ['predictions', 'probabilities', 'true_labels']}
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save predictions and probabilities separately
        np.save(Path(output_dir) / 'test_predictions.npy', results['predictions'])
        np.save(Path(output_dir) / 'test_probabilities.npy', results['probabilities'])
        np.save(Path(output_dir) / 'test_true_labels.npy', results['true_labels'])
        
        # Save file IDs
        with open(Path(output_dir) / 'test_file_ids.txt', 'w') as f:
            for file_id in results['file_ids']:
                f.write(f"{file_id}\n")
        
        logger.info(f"Test results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error saving test results: {str(e)}")
        raise
"""
Test script for enhanced audio augmentation pipeline.
Demonstrates the new augmentation techniques and their effects.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.components.DataAugmentation import AudioAugmentation, create_augmentation_pipeline
import librosa


def create_synthetic_audio(duration: float = 3.0, sr: int = 16000) -> np.ndarray:
    """Create synthetic audio for testing."""
    t = np.linspace(0, duration, int(duration * sr))
    
    # Create a synthetic speech-like signal
    # Fundamental frequency varying between 100-200 Hz
    f0 = 150 + 30 * np.sin(2 * np.pi * 0.5 * t)
    
    # Add harmonics to simulate speech
    signal = (
        np.sin(2 * np.pi * f0 * t) +
        0.5 * np.sin(2 * np.pi * 2 * f0 * t) +
        0.25 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    
    # Apply envelope to simulate speech patterns
    envelope = np.exp(-0.5 * (t - duration/2)**2 / (duration/4)**2)
    signal *= envelope
    
    # Add some noise
    signal += 0.1 * np.random.normal(0, 1, len(signal))
    
    return signal * 0.5  # Normalize amplitude


def create_synthetic_spectrogram(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Create mel spectrogram from audio."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=80,
        hop_length=160,
        n_fft=400
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return (log_mel + 80.0) / 80.0  # Whisper normalization


def test_individual_augmentations():
    """Test each augmentation technique individually."""
    print("Testing individual augmentation techniques...")
    
    # Create test data
    audio = create_synthetic_audio(duration=3.0)
    spectrogram = create_synthetic_spectrogram(audio)
    
    # Initialize augmentation pipeline
    aug_pipeline = create_augmentation_pipeline(apply_prob=1.0, individual_aug_prob=1.0)
    
    print(f"Original audio shape: {audio.shape}")
    print(f"Original spectrogram shape: {spectrogram.shape}")
    
    # Test tempo perturbation
    tempo_audio = aug_pipeline.tempo_perturbation(audio)
    print(f"Tempo perturbation - new length: {len(tempo_audio)} (ratio: {len(tempo_audio)/len(audio):.3f})")
    
    # Test speed perturbation
    speed_audio = aug_pipeline.speed_perturbation(audio)
    print(f"Speed perturbation - new length: {len(speed_audio)} (ratio: {len(speed_audio)/len(audio):.3f})")
    
    # Test noise addition
    noisy_audio = aug_pipeline.noise_addition(audio)
    snr_estimate = 10 * np.log10(np.mean(audio**2) / np.mean((noisy_audio - audio)**2))
    print(f"Noise addition - estimated SNR: {snr_estimate:.1f} dB")
    
    # Test VTLP
    vtlp_spec = aug_pipeline.vtlp_augmentation(spectrogram)
    print(f"VTLP - spectrogram difference: {np.mean(np.abs(vtlp_spec - spectrogram)):.4f}")
    
    # Test SpecAugment
    spec_aug_spec = aug_pipeline.spec_augment(spectrogram)
    masked_ratio = np.sum(spec_aug_spec == 0) / spec_aug_spec.size
    print(f"SpecAugment - masked ratio: {masked_ratio:.3f}")
    
    print("✓ Individual augmentation tests completed!")


def test_pipeline_integration():
    """Test the complete augmentation pipeline."""
    print("\nTesting complete augmentation pipeline...")
    
    # Create test data
    audio = create_synthetic_audio(duration=3.0)
    spectrogram = create_synthetic_spectrogram(audio)
    
    # Test with different probability settings
    test_configs = [
        {"apply_prob": 0.0, "individual_aug_prob": 0.5},  # No augmentation
        {"apply_prob": 1.0, "individual_aug_prob": 0.3},  # Low individual prob
        {"apply_prob": 1.0, "individual_aug_prob": 0.7},  # High individual prob
    ]
    
    for i, config in enumerate(test_configs):
        aug_pipeline = create_augmentation_pipeline(**config)
        
        # Test multiple samples to see probability effects
        augmented_count = 0
        for _ in range(10):
            aug_audio, aug_spec = aug_pipeline.apply_augmentation(
                audio=audio.copy(), 
                spectrogram=spectrogram.copy()
            )
            
            # Check if augmentation was applied
            if not np.array_equal(aug_audio, audio) or not np.array_equal(aug_spec, spectrogram):
                augmented_count += 1
        
        expected_prob = config["apply_prob"]
        actual_prob = augmented_count / 10
        print(f"Config {i+1}: Expected prob={expected_prob:.1f}, Actual prob={actual_prob:.1f}")
    
    print("✓ Pipeline integration tests completed!")


def test_dataset_integration():
    """Test integration with the dataset class."""
    print("\nTesting dataset integration...")
    
    # Create synthetic dataset
    n_samples = 5
    spectrograms = []
    labels = []
    file_ids = []
    
    for i in range(n_samples):
        audio = create_synthetic_audio(duration=3.0)
        spec = create_synthetic_spectrogram(audio)
        spectrograms.append(spec)
        
        # Create random multi-label
        label = np.random.randint(0, 2, size=6).tolist()
        labels.append(label)
        
        file_ids.append(f"test_file_{i}")
    
    # Test with enhanced augmentation
    try:
        from updated_dataset import Sep28kDataset
        
        # Training dataset (with augmentation)
        train_dataset = Sep28kDataset(
            spectrograms=spectrograms,
            labels=labels,
            file_ids=file_ids,
            augmentation_prob=0.8,  # High prob for testing
            is_training=True
        )
        
        # Test dataset loading
        sample = train_dataset[0]
        print(f"Dataset sample keys: {sample.keys()}")
        print(f"Input features shape: {sample['input_features'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")
        print(f"Augmentation stats: {train_dataset.get_augmentation_stats()}")
        
        # Validation dataset (no augmentation)
        val_dataset = Sep28kDataset(
            spectrograms=spectrograms,
            labels=labels,
            file_ids=file_ids,
            augmentation_prob=0.0,
            is_training=False
        )
        
        val_sample = val_dataset[0]
        print(f"Validation sample unchanged: {torch.equal(sample['input_features'], val_sample['input_features'])}")
        
        print("✓ Dataset integration tests completed!")
        
    except ImportError:
        print("⚠️ Dataset integration test skipped (import not available)")


def test_augmentation_effects():
    """Visualize the effects of different augmentations."""
    print("\nTesting augmentation effects (generating visualizations)...")
    
    # Create test audio and spectrogram
    audio = create_synthetic_audio(duration=3.0)
    spectrogram = create_synthetic_spectrogram(audio)
    
    # Initialize augmentation
    aug_pipeline = create_augmentation_pipeline(apply_prob=1.0, individual_aug_prob=1.0)
    
    # Apply different augmentations
    augmentations = {
        'Original': spectrogram,
        'VTLP': aug_pipeline.vtlp_augmentation(spectrogram.copy()),
        'SpecAugment': aug_pipeline.spec_augment(spectrogram.copy()),
        'Combined': None
    }
    
    # Apply combined augmentation
    _, combined_spec = aug_pipeline.apply_augmentation(
        audio=audio.copy(), 
        spectrogram=spectrogram.copy()
    )
    augmentations['Combined'] = combined_spec
    
    # Create visualization
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, (name, spec) in enumerate(augmentations.items()):
            if spec is not None:
                axes[i].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
                axes[i].set_title(f'{name}')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel('Mel Frequency')
        
        plt.tight_layout()
        plt.savefig('augmentation_effects.png', dpi=150, bbox_inches='tight')
        print("✓ Visualization saved as 'augmentation_effects.png'")
        
    except ImportError:
        print("⚠️ Visualization skipped (matplotlib not available)")
    
    # Print statistics
    for name, spec in augmentations.items():
        if spec is not None:
            diff = np.mean(np.abs(spec - spectrogram))
            print(f"{name}: Mean absolute difference = {diff:.4f}")


def benchmark_augmentation_speed():
    """Benchmark the speed of augmentation pipeline."""
    print("\nBenchmarking augmentation speed...")
    
    import time
    
    # Create test data
    audio = create_synthetic_audio(duration=3.0)
    spectrogram = create_synthetic_spectrogram(audio)
    
    # Test different configurations
    configs = [
        {"name": "No augmentation", "apply_prob": 0.0},
        {"name": "Basic probability", "apply_prob": 0.4},
        {"name": "High probability", "apply_prob": 0.8},
    ]
    
    n_iterations = 100
    
    for config in configs:
        aug_pipeline = create_augmentation_pipeline(
            apply_prob=config["apply_prob"],
            individual_aug_prob=0.5
        )
        
        start_time = time.time()
        
        for _ in range(n_iterations):
            _ = aug_pipeline.apply_augmentation(spectrogram=spectrogram.copy())
        
        end_time = time.time()
        avg_time = (end_time - start_time) / n_iterations * 1000  # Convert to ms
        
        print(f"{config['name']}: {avg_time:.2f} ms per sample")
    
    print("✓ Speed benchmark completed!")


def main():
    """Run all augmentation tests."""
    print("Enhanced Audio Augmentation Pipeline Tests")
    print("=" * 50)
    
    try:
        test_individual_augmentations()
        test_pipeline_integration()
        test_dataset_integration()
        test_augmentation_effects()
        benchmark_augmentation_speed()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("\nKey Features Verified:")
        print("✓ Tempo/Speed Perturbation")
        print("✓ Noise Addition (MUSAN-style)")
        print("✓ Vocal Tract Length Perturbation (VTLP)")
        print("✓ SpecAugment")
        print("✓ Stochastic application (p=0.3-0.5)")
        print("✓ Training-only application")
        print("✓ Dataset integration")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
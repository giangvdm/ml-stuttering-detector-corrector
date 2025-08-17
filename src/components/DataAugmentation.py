import torch
import numpy as np
import librosa
from typing import Optional, Tuple, Union
import random
from scipy import signal
from scipy.interpolate import interp1d


class AudioAugmentation:
    """
    Audio augmentation pipeline following the improved architecture specifications.
    
    Implements:
    - Tempo/Speed Perturbation
    - Noise Addition (MUSAN-style)
    - Vocal Tract Length Perturbation (VTLP)
    - SpecAugment
    
    Applied stochastically (p=0.3-0.5) during training only.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        apply_prob: float = 0.4,  # Probability to apply augmentation per sample
        individual_aug_prob: float = 0.5  # Probability for each individual augmentation
    ):
        self.sample_rate = sample_rate
        self.apply_prob = apply_prob
        self.individual_aug_prob = individual_aug_prob
        
        # Precomputed noise samples for efficiency
        self._noise_samples = self._generate_noise_bank()
    
    def _generate_noise_bank(self) -> list:
        """Generate a bank of noise samples for efficient noise addition."""
        noise_bank = []
        
        # Generate different types of noise
        for _ in range(10):  # Create 10 different noise samples
            duration = 5.0  # 5 seconds of noise
            samples = int(duration * self.sample_rate)
            
            # White noise
            white_noise = np.random.normal(0, 0.1, samples)
            
            # Pink noise (1/f noise)
            pink_noise = self._generate_pink_noise(samples)
            
            # Room tone simulation (low-frequency rumble)
            room_tone = self._generate_room_tone(samples)
            
            noise_bank.extend([white_noise, pink_noise, room_tone])
        
        return noise_bank
    
    def _generate_pink_noise(self, samples: int) -> np.ndarray:
        """Generate pink noise (1/f noise)."""
        white = np.random.randn(samples)
        # Simple pink noise filter approximation
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        pink = signal.lfilter(b, a, white)
        return pink * 0.1
    
    def _generate_room_tone(self, samples: int) -> np.ndarray:
        """Generate room tone (low-frequency ambient noise)."""
        # Low-frequency noise between 20-200 Hz
        t = np.linspace(0, samples / self.sample_rate, samples)
        room_tone = (
            0.05 * np.sin(2 * np.pi * 60 * t) +  # 60 Hz hum
            0.03 * np.sin(2 * np.pi * 120 * t) + # 120 Hz harmonic
            0.02 * np.random.normal(0, 1, samples)  # Low-level random noise
        )
        return room_tone
    
    def tempo_perturbation(
        self, 
        audio: np.ndarray, 
        tempo_range: Tuple[float, float] = (0.8, 1.2)
    ) -> np.ndarray:
        """
        Apply tempo perturbation without changing pitch.
        
        Args:
            audio: Input audio signal
            tempo_range: Range of tempo multipliers (0.8 = 20% slower, 1.2 = 20% faster)
            
        Returns:
            Tempo-modified audio
        """
        tempo_factor = random.uniform(*tempo_range)
        
        # Use librosa's time_stretch for tempo change without pitch change
        return librosa.effects.time_stretch(audio, rate=tempo_factor)
    
    def speed_perturbation(
        self, 
        audio: np.ndarray, 
        speed_range: Tuple[float, float] = (0.9, 1.1)
    ) -> np.ndarray:
        """
        Apply speed perturbation (changes both tempo and pitch).
        
        Args:
            audio: Input audio signal
            speed_range: Range of speed multipliers
            
        Returns:
            Speed-modified audio
        """
        speed_factor = random.uniform(*speed_range)
        
        # Resample to change speed (affects both tempo and pitch)
        original_length = len(audio)
        new_length = int(original_length / speed_factor)
        
        # Interpolate to new length
        old_indices = np.linspace(0, original_length - 1, original_length)
        new_indices = np.linspace(0, original_length - 1, new_length)
        interpolator = interp1d(old_indices, audio, kind='linear', bounds_error=False, fill_value=0)
        
        return interpolator(new_indices)
    
    def noise_addition(
        self, 
        audio: np.ndarray, 
        snr_range: Tuple[float, float] = (15, 35)
    ) -> np.ndarray:
        """
        Add background noise with specified SNR range.
        
        Args:
            audio: Input audio signal
            snr_range: Signal-to-noise ratio range in dB
            
        Returns:
            Audio with added noise
        """
        # Select random noise from bank
        noise = random.choice(self._noise_samples)
        
        # Ensure noise is same length as audio
        if len(noise) >= len(audio):
            start_idx = random.randint(0, len(noise) - len(audio))
            noise = noise[start_idx:start_idx + len(audio)]
        else:
            # Repeat noise if it's shorter than audio
            repeats = int(np.ceil(len(audio) / len(noise)))
            noise = np.tile(noise, repeats)[:len(audio)]
        
        # Calculate current signal power
        signal_power = np.mean(audio ** 2)
        
        # Calculate target noise power based on desired SNR
        target_snr = random.uniform(*snr_range)
        noise_power = signal_power / (10 ** (target_snr / 10))
        
        # Scale noise to achieve target SNR
        current_noise_power = np.mean(noise ** 2)
        if current_noise_power > 0:
            noise_scale = np.sqrt(noise_power / current_noise_power)
            noise = noise * noise_scale
        
        return audio + noise
    
    def vtlp_augmentation(
        self, 
        spectrogram: np.ndarray, 
        warp_range: Tuple[float, float] = (0.9, 1.1)
    ) -> np.ndarray:
        """
        Apply Vocal Tract Length Perturbation (VTLP) to spectrogram.
        
        Args:
            spectrogram: Input mel spectrogram [n_mels, time]
            warp_range: Range of frequency warping factors
            
        Returns:
            VTLP-augmented spectrogram
        """
        warp_factor = random.uniform(*warp_range)
        
        n_mels, n_frames = spectrogram.shape
        
        # Create frequency warping matrix
        old_mel_indices = np.arange(n_mels)
        new_mel_indices = old_mel_indices * warp_factor
        
        # Clip to valid range
        new_mel_indices = np.clip(new_mel_indices, 0, n_mels - 1)
        
        # Interpolate each time frame
        warped_spec = np.zeros_like(spectrogram)
        for t in range(n_frames):
            interpolator = interp1d(
                old_mel_indices, 
                spectrogram[:, t], 
                kind='linear', 
                bounds_error=False, 
                fill_value=0
            )
            warped_spec[:, t] = interpolator(new_mel_indices)
        
        return warped_spec
    
    def spec_augment(
        self, 
        spectrogram: np.ndarray,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        num_freq_masks: int = 1,
        num_time_masks: int = 1
    ) -> np.ndarray:
        """
        Apply SpecAugment: frequency and time masking.
        
        Args:
            spectrogram: Input mel spectrogram [n_mels, time]
            freq_mask_param: Maximum width of frequency mask
            time_mask_param: Maximum width of time mask
            num_freq_masks: Number of frequency masks to apply
            num_time_masks: Number of time masks to apply
            
        Returns:
            SpecAugment-applied spectrogram
        """
        augmented_spec = spectrogram.copy()
        n_mels, n_frames = augmented_spec.shape
        
        # Apply frequency masks
        for _ in range(num_freq_masks):
            f = random.randint(0, freq_mask_param)
            f0 = random.randint(0, n_mels - f)
            augmented_spec[f0:f0 + f, :] = 0
        
        # Apply time masks
        for _ in range(num_time_masks):
            t = random.randint(0, min(time_mask_param, n_frames // 5))  # Limit to 20% of total time
            t0 = random.randint(0, n_frames - t)
            augmented_spec[:, t0:t0 + t] = 0
        
        return augmented_spec
    
    def apply_augmentation(
        self, 
        audio: Optional[np.ndarray] = None, 
        spectrogram: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply the complete augmentation pipeline.
        
        Args:
            audio: Input audio signal (optional)
            spectrogram: Input mel spectrogram (optional)
            
        Returns:
            Augmented audio and/or spectrogram
        """
        # Decide whether to apply augmentation
        if random.random() > self.apply_prob:
            if audio is not None and spectrogram is not None:
                return audio, spectrogram
            elif audio is not None:
                return audio
            else:
                return spectrogram
        
        augmented_audio = audio
        augmented_spec = spectrogram
        
        # Apply audio-domain augmentations
        if audio is not None:
            # Tempo perturbation
            if random.random() < self.individual_aug_prob:
                augmented_audio = self.tempo_perturbation(augmented_audio)
            
            # Speed perturbation (alternative to tempo)
            elif random.random() < self.individual_aug_prob * 0.5:
                augmented_audio = self.speed_perturbation(augmented_audio)
            
            # Noise addition
            if random.random() < self.individual_aug_prob:
                augmented_audio = self.noise_addition(augmented_audio)
        
        # Apply spectrogram-domain augmentations
        if spectrogram is not None:
            # VTLP
            if random.random() < self.individual_aug_prob:
                augmented_spec = self.vtlp_augmentation(augmented_spec)
            
            # SpecAugment
            if random.random() < self.individual_aug_prob:
                augmented_spec = self.spec_augment(augmented_spec)
        
        # Return appropriate results
        if audio is not None and spectrogram is not None:
            return augmented_audio, augmented_spec
        elif audio is not None:
            return augmented_audio
        else:
            return augmented_spec


# Convenience function for easy integration
def create_augmentation_pipeline(
    sample_rate: int = 16000,
    apply_prob: float = 0.4,
    individual_aug_prob: float = 0.5
) -> AudioAugmentation:
    """
    Factory function to create the enhanced augmentation pipeline.
    
    Args:
        sample_rate: Audio sample rate
        apply_prob: Probability to apply augmentation per sample (0.3-0.5 as per spec)
        individual_aug_prob: Probability for each individual augmentation
        
    Returns:
        Configured augmentation pipeline
    """
    return AudioAugmentation(
        sample_rate=sample_rate,
        apply_prob=apply_prob,
        individual_aug_prob=individual_aug_prob
    )
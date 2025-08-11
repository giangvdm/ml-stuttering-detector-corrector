import torch
import numpy as np
from typing import Union, List, Tuple, Optional
import librosa
import warnings


class AudioPreprocessor:
    """
    Audio preprocessing pipeline for Whisper-based stuttering classification.
    
    Handles:
    - Audio loading and resampling
    - Log-mel spectrogram generation
    - Normalization and padding
    - Batch processing
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        target_sample_rate: int = 16000,
        n_mels: int = 80,
        hop_length: int = 160,
        chunk_length: int = 30,  # seconds
        normalize: bool = True
    ):
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.normalize = normalize
        
        # Calculate frame parameters (matching Whisper)
        self.n_fft = 400  # Whisper default
        self.chunk_length_samples = chunk_length * target_sample_rate
        
        print("AudioPreprocessor initialized with efficient librosa backend")
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sample rate.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            # Use librosa for robust audio loading
            audio, sr = librosa.load(audio_path, sr=self.target_sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            raise ValueError(f"Failed to load audio from {audio_path}: {e}")
    
    def audio_to_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to log-mel spectrogram for 3-second clips as used in Whisper in Focus.
        
        Args:
            audio: Audio array (typically 3 seconds = 48,000 samples at 16kHz)
            
        Returns:
            Log-mel spectrogram [n_mels, time_steps] - variable time steps based on audio length
        """
        # Generate mel spectrogram using librosa (matching Whisper's parameters)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=self.target_sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            fmin=0,
            fmax=self.target_sample_rate // 2
        )
        
        # Convert to log scale (matching Whisper)
        log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Normalize to match Whisper's expected input range
        log_mel = (log_mel + 80.0) / 80.0  # Whisper normalization
        
        return log_mel
    
    def chunk_audio(self, audio: np.ndarray, chunk_duration: float = 3.0) -> List[np.ndarray]:
        """
        Split audio into 3-second chunks as used in Whisper in Focus paper.
        
        Args:
            audio: Audio array
            chunk_duration: Duration of each chunk in seconds (3.0 as per paper)
            
        Returns:
            List of audio chunks
        """
        chunk_samples = int(chunk_duration * self.target_sample_rate)
        chunks = []
        
        for start in range(0, len(audio), chunk_samples):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            
            # Only keep chunks that are at least 3 seconds (as per paper)
            if len(chunk) >= chunk_samples:
                chunks.append(chunk)
            elif len(chunk) >= chunk_samples * 0.9:  # Accept chunks that are at least 90% of target length
                # Pad to exact length
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
                chunks.append(chunk)
            # Reject chunks shorter than 90% of target length
        
        return chunks
    
    def process_audio_file(
        self, 
        audio_path: str, 
        chunk_duration: float = 3.0
    ) -> List[np.ndarray]:
        """
        Complete preprocessing pipeline for a single audio file.
        Following the Whisper in Focus approach: 3-second chunks.
        
        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of chunks in seconds (3.0 as per paper)
            
        Returns:
            List of log-mel spectrograms with shape [80, ~300] each (300 for 3-second clips)
        """
        # Load audio
        audio, _ = self.load_audio(audio_path)
        
        # Split into chunks
        chunks = self.chunk_audio(audio, chunk_duration)
        
        # Convert each chunk to mel spectrogram
        mel_spectrograms = []
        for chunk in chunks:
            mel_spec = self.audio_to_mel_spectrogram(chunk)
            mel_spectrograms.append(mel_spec)
        
        return mel_spectrograms
    
    def batch_process(
        self, 
        audio_paths: List[str], 
        chunk_duration: float = 3.0
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Process multiple audio files in batch.
        
        Args:
            audio_paths: List of audio file paths
            chunk_duration: Duration of chunks in seconds
            
        Returns:
            Tuple of (batched_tensors, file_identifiers)
        """
        all_spectrograms = []
        file_identifiers = []
        
        for audio_path in audio_paths:
            try:
                spectrograms = self.process_audio_file(audio_path, chunk_duration)
                all_spectrograms.extend(spectrograms)
                
                # Create identifiers for each chunk
                for i in range(len(spectrograms)):
                    file_identifiers.append(f"{audio_path}_chunk_{i}")
                    
            except Exception as e:
                warnings.warn(f"Failed to process {audio_path}: {e}")
                continue
        
        if not all_spectrograms:
            raise ValueError("No audio files could be processed successfully")
        
        # Convert to tensor and add batch dimension
        tensor_batch = torch.stack([torch.from_numpy(spec) for spec in all_spectrograms])
        
        return tensor_batch, file_identifiers
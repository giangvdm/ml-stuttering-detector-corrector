import torch
import torchaudio

class SpectrogramProcessor:
    def __init__(self, 
                 sample_rate=16000,
                 n_fft=400,           # 25ms window at 16kHz
                 hop_length=160,      # 10ms hop at 16kHz  
                 n_mels=128,          # Number of mel filter banks
                 f_min=0.0,           # Minimum frequency
                 f_max=8000.0,        # Maximum frequency (Nyquist for 16kHz)
                 normalize=True):     # Apply normalization (-1 to 1)
        """
        Initialize spectrogram processor with audio processing parameters.
        
        Args:
            sample_rate: Target sample rate for audio
            n_fft: FFT window size (25ms = 400 samples at 16kHz)
            hop_length: Hop size between windows (10ms = 160 samples at 16kHz)
            n_mels: Number of mel filter banks
            f_min: Minimum frequency for mel scale
            f_max: Maximum frequency for mel scale
            normalize: Whether to normalize spectrograms to [-1, 1] range
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.normalize = normalize
        
        # Create mel spectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,  # Power spectrogram (magnitude squared)
            normalized=False,
            center=True,
            pad_mode="reflect"
        )
        
        print(f"SpectrogramProcessor initialized:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Window size: {n_fft} samples ({n_fft/sample_rate*1000:.1f}ms)")
        print(f"  Hop length: {hop_length} samples ({hop_length/sample_rate*1000:.1f}ms)")
        print(f"  Mel bins: {n_mels}")
        print(f"  Frequency range: {f_min}-{f_max} Hz")
    
    def process_waveform(self, waveform):
        """
        Convert waveform to mel spectrogram.
        
        Args:
            waveform: 1D tensor of audio samples [samples]
            
        Returns:
            mel_spec: 2D tensor [n_mels, time_frames]
        """
        # Ensure waveform is 1D
        if waveform.dim() > 1:
            waveform = waveform.squeeze()
        
        # Convert to mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        
        # Convert to log scale (dB)
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        # Normalize to [-1, 1] range if requested
        if self.normalize:
            mel_spec = self._normalize_spectrogram(mel_spec)
        
        return mel_spec
    
    def _normalize_spectrogram(self, mel_spec):
        """
        Normalize mel spectrogram to [-1, 1] range.
        
        Args:
            mel_spec: 2D tensor [n_mels, time_frames]
            
        Returns:
            normalized_spec: 2D tensor normalized to [-1, 1]
        """
        # Get min and max values
        spec_min = mel_spec.min()
        spec_max = mel_spec.max()
        
        # Avoid division by zero
        if spec_max - spec_min == 0:
            return torch.zeros_like(mel_spec)
        
        # Normalize to [0, 1] then scale to [-1, 1]
        normalized = (mel_spec - spec_min) / (spec_max - spec_min)  # [0, 1]
        normalized = 2.0 * normalized - 1.0  # [-1, 1]
        
        return normalized
    
    def process_batch(self, waveforms):
        """
        Process a batch of waveforms to spectrograms.
        
        Args:
            waveforms: Tensor of shape [batch_size, samples]
            
        Returns:
            spectrograms: Tensor of shape [batch_size, n_mels, time_frames]
        """
        batch_size = waveforms.shape[0]
        spectrograms = []
        
        for i in range(batch_size):
            spec = self.process_waveform(waveforms[i])
            spectrograms.append(spec)
        
        # Stack into batch tensor
        return torch.stack(spectrograms, dim=0)

    def get_output_shape(self, input_length):
        """
        Calculate expected spectrogram shape for given input length.
        
        Args:
            input_length: Number of audio samples
            
        Returns:
            tuple: (n_mels, time_frames)
        """
        time_frames = (input_length - self.n_fft) // self.hop_length + 1
        return (self.n_mels, time_frames)
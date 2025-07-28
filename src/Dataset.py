import torch
import torchaudio
import pandas as pd
import os

class SEP28kDataset(torch.utils.data.Dataset):
    """Dataset class that now loads pre-computed transcripts."""
    def __init__(self, csv_file, audio_root_dir):
        self.df = pd.read_csv(csv_file).dropna(subset=['transcript'])
        self.audio_root_dir = audio_root_dir
        self.labels = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection', 'Fluent']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_root_dir, row['filepath'])

        if not os.path.exists(audio_path):
            return None

        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        label_vector = torch.tensor([row[label] for label in self.labels], dtype=torch.float32)
        transcript = row['transcript']
        
        return {'waveform': waveform.squeeze(0), 'transcript': transcript, 'label': label_vector}
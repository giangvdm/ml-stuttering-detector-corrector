import torch
import torchaudio
import pandas as pd
import os

class SEP28kDataset(torch.utils.data.Dataset):
    """Dataset class that now loads pre-computed transcripts."""
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file).dropna(subset=['Transcript'])
        self.labels = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row['filepath']

        if not os.path.exists(audio_path):
            return None

        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        label_vector = torch.tensor([row[label] for label in self.labels], dtype=torch.float32)
        transcript = row['Transcript']
        
        return {'waveform': waveform.squeeze(0), 'transcript': transcript, 'label': label_vector}
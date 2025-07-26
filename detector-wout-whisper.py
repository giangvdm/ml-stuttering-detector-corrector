import torch
import torch.nn as nn
import torchaudio
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

class Wav2Vec2Encoder(nn.Module):
    """Acoustic stream using Wav2Vec2.0 for extracting acoustic embeddings."""
    def __init__(self):
        super().__init__()
        self.wav2vec2 = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            
    def forward(self, audio_input):
        features, _ = self.wav2vec2.extract_features(audio_input)
        return features[-1]

class BertLinguisticEncoder(nn.Module):
    """Linguistic stream using BERT for extracting linguistic embeddings."""
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def forward(self, text_input, device):
        encoded = self.tokenizer(
            text_input, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(device)
        outputs = self.bert(**encoded)
        return outputs.last_hidden_state

class FeatureFusion(nn.Module):
    """Feature fusion module for concatenating acoustic and linguistic embeddings."""
    def __init__(self, acoustic_dim=768, linguistic_dim=768, fusion_dim=1024):
        super().__init__()
        self.acoustic_proj = nn.Linear(acoustic_dim, fusion_dim // 2)
        self.linguistic_proj = nn.Linear(linguistic_dim, fusion_dim // 2)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(fusion_dim)
        
    def forward(self, acoustic_features, linguistic_features):
        acoustic_proj = self.acoustic_proj(acoustic_features)
        target_len = acoustic_proj.size(1)
        linguistic_upsampled = nn.functional.interpolate(
            linguistic_features.transpose(1, 2), size=target_len, mode='linear'
        ).transpose(1, 2)
        linguistic_proj = self.linguistic_proj(linguistic_upsampled)
        fused_features = torch.cat([acoustic_proj, linguistic_proj], dim=-1)
        fused_features = self.layer_norm(fused_features)
        fused_features = self.dropout(fused_features)
        return fused_features

class BiLSTMAttentionClassifier(nn.Module):
    """BiLSTM for dysfluency classification."""
    def __init__(self, input_dim=1024, hidden_dim=256, num_layers=2, num_classes=6):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.3 if num_layers > 1 else 0
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        logits = self.classifier(lstm_out)
        return logits

class DysfluencyDetectorSEP(nn.Module):
    """Complete dysfluency detection system for SEP-28k."""
    def __init__(self, num_classes=6, device='cpu'):
        super().__init__()
        self.device = device
        self.wav2vec_encoder = Wav2Vec2Encoder().to(device)
        self.bert_encoder = BertLinguisticEncoder().to(device)
        self.feature_fusion = FeatureFusion().to(device)
        self.classifier = BiLSTMAttentionClassifier(num_classes=num_classes).to(device)
        
    def forward(self, waveforms, transcripts):
        """Forward pass now takes waveforms and pre-computed transcripts."""
        acoustic_features = self.wav2vec_encoder(waveforms)
        linguistic_features = self.bert_encoder(transcripts, self.device)
        fused_features = self.feature_fusion(acoustic_features, linguistic_features)
        frame_logits = self.classifier(fused_features)
        clip_logits = torch.mean(frame_logits, dim=1)
        return clip_logits

class SEP28kDataset(torch.utils.data.Dataset):
    """Dataset class that now loads pre-computed transcripts."""
    def __init__(self, csv_file, audio_root_dir):
        self.df = pd.read_csv(csv_file).dropna(subset=['transcript']) # Drop rows where transcription failed
        self.audio_root_dir = audio_root_dir
        self.labels = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection', 'Fluent']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_root_dir, row['filepath'])

        # --- ADDED CHECK ---
        # If the audio file does not exist, return None.
        # The collate_fn will handle this by skipping the item.
        if not os.path.exists(audio_path):
            # It's good practice to log or print a warning for debugging.
            # print(f"Warning: File not found, skipping: {audio_path}")
            return None
        # --- END OF CHECK ---

        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        label_vector = torch.tensor([row[label] for label in self.labels], dtype=torch.float32)
        transcript = row['transcript']
        
        return {'waveform': waveform.squeeze(0), 'transcript': transcript, 'label': label_vector}

def collate_fn(batch):
    """Custom collate function to handle variable length audio and text."""
    # --- ADDED FILTER ---
    # Filter out None items which are returned by __getitem__ when a file is missing.
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if the whole batch was invalid
    # --- END OF FILTER ---

    waveforms = [item['waveform'] for item in batch]
    transcripts = [item['transcript'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    # Pad waveforms to the same length
    padded_waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    
    return {'waveforms': padded_waveforms, 'transcripts': transcripts, 'labels': labels}

class SEP28kTrainer:
    """Training utility for the SEP-28k detector."""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        waveforms = batch['waveforms'].to(self.device)
        transcripts = batch['transcripts']
        labels = batch['labels'].to(self.device)
        
        logits = self.model(waveforms, transcripts)
        loss = self.criterion(logits, labels)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                # --- ADDED CHECK ---
                # Skip if the batch is empty (all files were missing)
                if batch is None:
                    continue
                # --- END OF CHECK ---
                waveforms = batch['waveforms'].to(self.device)
                transcripts = batch['transcripts']
                labels = batch['labels'].to(self.device)
                
                logits = self.model(waveforms, transcripts)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        if not all_preds:
            return 0, 0 # Handle case where validation set is empty or all files are missing

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        accuracy = (all_preds == all_labels).all(dim=1).float().mean().item()
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy

def run_training_loop(trainer, train_loader, val_loader, num_epochs):
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        trainer.model.train()
        train_losses = []
        # Use tqdm for a training progress bar
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            # --- ADDED CHECK ---
            # Skip if the batch is empty (all files were missing)
            if batch is None:
                continue
            # --- END OF CHECK ---
            loss = trainer.train_step(batch)
            train_losses.append(loss)
        
        avg_train_loss = np.mean(train_losses)
        print(f"  Average Training Loss: {avg_train_loss:.4f}")
        
        print("  Validating...")
        val_loss, val_accuracy = trainer.evaluate(val_loader)
        print(f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(trainer.model.state_dict(), 'best_dysfluency_model.pth')
            print(f"  New best model saved! (Validation Accuracy: {val_accuracy:.4f})")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = DysfluencyDetectorSEP(num_classes=6, device=device)

    csv_file = "sep28k_labels_with_transcripts.csv"
    audio_root_dir = "." 
    
    if not os.path.exists(csv_file):
        print(f"ERROR: Transcript file not found at '{csv_file}'. Please run the transcription script first.")
    else:
        dataset = SEP28kDataset(csv_file=csv_file, audio_root_dir=audio_root_dir)
        
        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)
        val_size = int(0.1 * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        print(f"\nDataset split:")
        print(f"  Training set size: {len(train_dataset)}")
        print(f"  Validation set size: {len(val_dataset)}")
        print(f"  Test set size: {len(test_dataset)}")
        
        # Now we can use a larger batch size for faster training.
        batch_size = 16 
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

        trainer = SEP28kTrainer(model, device=device)

        num_training_epochs = 5
        run_training_loop(trainer, train_dataloader, val_dataloader, num_epochs=num_training_epochs)
        
        print("\nTraining complete.")

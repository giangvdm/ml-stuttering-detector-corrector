import torch
import torch.nn as nn
import torchaudio
import whisper
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import os
from typing import Dict, List

# --- Jargon Check ---
# Tensor: A multi-dimensional array, the fundamental data structure in PyTorch.
# .to(device): A PyTorch method to move a tensor or model to a specific device (e.g., 'cpu' or 'cuda').
# nn.Module: The base class for all neural network modules in PyTorch.

class Wav2Vec2Encoder(nn.Module):
    """Acoustic stream using Wav2Vec2.0 for extracting acoustic embeddings."""
    def __init__(self):
        super().__init__()
        self.wav2vec2 = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
        # Freeze parameters as we are only using it for feature extraction.
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            
    def forward(self, audio_input):
        """Extract acoustic embeddings from raw audio."""
        # The model's feature extractor is used here.
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
        """Extract linguistic embeddings from text."""
        if isinstance(text_input, str):
            text_input = [text_input]
        
        # Tokenize and crucially, move the resulting tensors to the correct device.
        encoded = self.tokenizer(
            text_input, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(device)
        
        outputs = self.bert(**encoded)
        return outputs.last_hidden_state

class FeatureFusion(nn.Module):
    """Feature fusion module for concatenating acoustic and linguistic embeddings."""
    def __init__(self, acoustic_dim=768, linguistic_dim=768, fusion_dim=1024):
        super().__init__()
        # Projection layers can help the model learn to combine the features more effectively.
        self.acoustic_proj = nn.Linear(acoustic_dim, fusion_dim // 2)
        self.linguistic_proj = nn.Linear(linguistic_dim, fusion_dim // 2)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(fusion_dim)
        
    def forward(self, acoustic_features, linguistic_features):
        """Fuse acoustic and linguistic features."""
        acoustic_proj = self.acoustic_proj(acoustic_features)
        
        # Upsample linguistic features to match the acoustic sequence length.
        # This is a more robust way to handle alignment.
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
    """BiLSTM with attention mechanism for dysfluency classification."""
    def __init__(self, input_dim=1024, hidden_dim=256, num_layers=2, num_classes=6):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.3 if num_layers > 1 else 0
        )
        # Final classifier layer.
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        """Forward pass through BiLSTM."""
        lstm_out, _ = self.bilstm(x)
        logits = self.classifier(lstm_out)
        return logits

class DysfluencyDetectorSEP(nn.Module):
    """Complete dysfluency detection system for SEP-28k."""
    def __init__(self, num_classes=6, device='cpu'):
        super().__init__()
        self.device = device
        
        # Initialize components and move them to the specified device.
        self.wav2vec_encoder = Wav2Vec2Encoder().to(device)
        self.whisper_asr = whisper.load_model("base", device=device)
        self.bert_encoder = BertLinguisticEncoder().to(device)
        self.feature_fusion = FeatureFusion().to(device)
        self.classifier = BiLSTMAttentionClassifier(num_classes=num_classes).to(device)
        
    def forward(self, audio_path, return_frame_level=False):
        """Forward pass with automatic transcription."""
        # 1. Acoustic Stream
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000).to(self.device)
            waveform = resampler(waveform)
        
        # Move waveform to the correct device BEFORE passing to the model.
        acoustic_features = self.wav2vec_encoder(waveform.to(self.device))
        
        # 2. Linguistic Stream
        transcript = self.whisper_asr.transcribe(audio_path)["text"]
        linguistic_features = self.bert_encoder(transcript, self.device)
        
        # 3. Fusion
        fused_features = self.feature_fusion(acoustic_features, linguistic_features)
        
        # 4. Classification
        frame_logits = self.classifier(fused_features)
        
        if return_frame_level:
            return frame_logits
        
        # Pool frame-level predictions to get a single clip-level prediction.
        clip_logits = torch.mean(frame_logits, dim=1)
        return clip_logits

class SEP28kDataset(torch.utils.data.Dataset):
    """Dataset class for SEP-28k dysfluency dataset."""
    def __init__(self, csv_file, audio_root_dir):
        self.df = pd.read_csv(csv_file)
        self.audio_root_dir = audio_root_dir
        self.labels = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection', 'Fluent']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_root_dir, row['filepath'])
        
        # Create the multi-hot label vector
        label_vector = torch.tensor([row[label] for label in self.labels], dtype=torch.float32)
        
        return {'audio_path': audio_path, 'label': label_vector}

class SEP28kTrainer:
    """Training utility for the SEP-28k detector."""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        # Use BCEWithLogitsLoss for multi-label classification.
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Assuming batch size is 1 for simplicity with file paths
        audio_path = batch['audio_path'][0]
        labels = batch['label'].to(self.device)
        
        logits = self.model(audio_path)
        loss = self.criterion(logits, labels)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def evaluate(self, dataloader):
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                audio_path = batch['audio_path'][0]
                labels = batch['label'].to(self.device)
                
                logits = self.model(audio_path)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Apply sigmoid and threshold to get binary predictions
                preds = (torch.sigmoid(logits) > 0.5).float()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        # Calculate accuracy (exact match ratio)
        accuracy = (all_preds == all_labels).all(dim=1).float().mean().item()
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy

def run_training_loop(trainer, train_loader, val_loader, num_epochs):
    """
    Main function to run the training and validation loop.
    """
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # --- Training Phase ---
        trainer.model.train()
        train_losses = []
        for i, batch in enumerate(train_loader):
            loss = trainer.train_step(batch)
            train_losses.append(loss)
            
            # Progress indicator
            print(f"\r  Training... Batch {i+1}/{len(train_loader)}, Loss: {loss:.4f}", end="")
        
        avg_train_loss = np.mean(train_losses)
        print(f"\n  Average Training Loss: {avg_train_loss:.4f}")
        
        # --- Validation Phase ---
        print("  Validating...")
        val_loss, val_accuracy = trainer.evaluate(val_loader)
        
        print(f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # --- Save Best Model ---
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(trainer.model.state_dict(), 'best_dysfluency_model.pth')
            print(f"  New best model saved! (Validation Accuracy: {val_accuracy:.4f})")


# --- Example Main Execution Block ---
if __name__ == "__main__":
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialise model
    model = DysfluencyDetectorSEP(num_classes=6, device=device)

    # Dataset and Dataloader creation
    csv_file = "sep28k_labels_processed.csv"
    audio_root_dir = "ml-stuttering-events-dataset" 
    
    if not os.path.exists(csv_file):
        print(f"ERROR: Processed CSV file not found at '{csv_file}'. Please run the preprocessing script first.")
    else:
        dataset = SEP28kDataset(csv_file=csv_file, audio_root_dir=audio_root_dir)
        
        # --- Dataset Splitting (80% train, 10% val, 10% test) ---
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
        
        # Create DataLoaders
        # NOTE: Batch size is 1 because Whisper processes files by path.
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
        # test_dataloader is not created as testing is not implemented yet.

        # Initialise Trainer
        trainer = SEP28kTrainer(model, device=device)

        # --- Start Training ---
        num_training_epochs = 5 # You can change this value
        run_training_loop(trainer, train_dataloader, val_dataloader, num_epochs=num_training_epochs)
        
        print("\nTraining complete.")

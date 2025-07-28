import torch
import torch.nn as nn
from sklearn.metrics import f1_score

class SEP28kTrainer:
    """Training utility for the SEP-28k detector."""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        self.labels = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection', 'Fluent']
        
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
        """Evaluate model on validation set using F1-score."""
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                if batch is None:
                    continue
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
            return 0, 0, {}

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        # --- NEW METRICS ---
        # Calculate the macro-averaged F1 score. This is the main performance metric.
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # Calculate the F1 score for each class individually for detailed analysis.
        per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        per_class_f1_dict = {label: score for label, score in zip(self.labels, per_class_f1)}
        
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, macro_f1, per_class_f1_dict
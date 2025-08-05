import torch.nn as nn
from transformers import BertTokenizer, BertModel

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
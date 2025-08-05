import torch
from src.DysfluencyDetector import DysfluencyDetector
from src.model.Dataset import SEP28kDataset
from src.model.SpectrogramProcessor import SpectrogramProcessor
from src.model.VGGBackbone import VGGBackbone
from src.model.ResNetFeatureExtractor import ResNetFeatureExtractor
from src.model.BiLSTMEncoder import BiLSTMEncoder
from src.model.AttentionMechanism import AttentionMechanism
from src.model.ClassificationHead import ClassificationHead

dataset = SEP28kDataset('all_labels.csv', target_disfluency='Block')
# sample = dataset[14]
# print(f"Waveform shape: {sample['waveform'].shape}")
# print(f"Label: {sample['label']}")
# print(f"Label type: {type(sample['label'])}")
# print(f"Audio file: {sample['filepath']}")

# disfluency_types = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection']

# for disfluency in disfluency_types:
#     splits = SEP28kDataset.create_stratified_splits(
#         'all_labels.csv',
#         disfluency,
#         f'./splits/{disfluency}/'
#     )
#     print(f"Created splits for {disfluency}")

# Test the full workflow
dataset = SEP28kDataset('./splits/Block/Block_train.csv', target_disfluency='Block')
dataset.get_dataset_info()

# Apply upsampling
dataset.apply_upsampling()
dataset.get_dataset_info()

# Get weights for loss function
pos_weight = dataset.get_class_weights()

sample = dataset[0]
# Process spectrogram
processor = SpectrogramProcessor()
spectrogram = processor.process_waveform(sample['waveform'])

print(f"Waveform shape: {sample['waveform'].shape}")
print(f"Spectrogram shape: {spectrogram.shape}")

waveforms = torch.stack([sample['waveform'] for _ in range(3)])
batch_spectrograms = processor.process_batch(waveforms)
print(f"Batch spectrograms shape: {batch_spectrograms.shape}")

# Test with your spectrogram
backbone = VGGBackbone(input_channels=1)

# Add batch and channel dimensions to spectrogram
test_input = spectrogram.unsqueeze(0).unsqueeze(0)  # [1, 1, 128, 301]
print(f"Input shape: {test_input.shape}")

# Forward pass
features = backbone(test_input)
print(f"Output features shape: {features.shape}")

# Use the features from VGGBackbone test
feature_extractor = ResNetFeatureExtractor(input_size=features.shape[1])

# Process VGG features
processed_features = feature_extractor(features)
print(f"VGG features shape: {features.shape}")
print(f"Processed features shape: {processed_features.shape}")

# Use processed features from ResNet (shape: [1, 2, 1024])
bilstm = BiLSTMEncoder(input_size=1024, hidden_size=256, num_layers=2)

# Forward pass
lstm_output, (h_n, c_n) = bilstm(processed_features)

print(f"Input shape: {processed_features.shape}")
print(f"LSTM output shape: {lstm_output.shape}")
print(f"Hidden state shape: {h_n.shape}")
print(f"Cell state shape: {c_n.shape}")

# Use LSTM output from previous test (shape: [1, 2, 512])
attention = AttentionMechanism(input_size=512, attention_size=256)

# Forward pass
attended_output, attention_weights = attention(lstm_output)

print(f"LSTM output shape: {lstm_output.shape}")
print(f"Attended output shape: {attended_output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
print(f"Attention weights: {attention_weights}")
print(f"Attention weights sum: {attention_weights.sum()}")  # Should be 1.0

# Use attended output from attention test (shape: [1, 512])
classifier = ClassificationHead(input_size=512, hidden_dim=4096, dropout=0.5)

# Forward pass
logits = classifier(attended_output)
probabilities = classifier.predict_proba(attended_output)
predictions = classifier.predict(attended_output)

print(f"Attention output shape: {attended_output.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Logits value: {logits.item():.4f}")
print(f"Probability shape: {probabilities.shape}")
print(f"Probability value: {probabilities.item():.4f}")
print(f"Binary prediction: {predictions.item()}")

# Create the complete model
model = DysfluencyDetector(input_channels=1)

# Test with the spectrogram (add batch and channel dimensions)
test_input = spectrogram.unsqueeze(0).unsqueeze(0)  # [1, 1, 128, 301]
print(f"Input shape: {test_input.shape}")

# Forward pass
logits = model(test_input)
probabilities = model.predict_proba(test_input)

print(f"Model output (logits): {logits.shape} - {logits.item():.4f}")
print(f"Model probability: {probabilities.item():.4f}")

# Test intermediate outputs
intermediate = model.get_intermediate_outputs(test_input)
for key, value in intermediate.items():
    if isinstance(value, torch.Tensor):
        print(f"{key}: {value.shape}")

# # Demo training
# def train_disfluency_model(disfluency_type):
#     for disfluency in disfluency_types:
#         # Load class-specific splits
#         train_dataset = SEP28kDataset(f'./splits/{disfluency}/{disfluency}_train.csv', 
#                                     target_disfluency=disfluency)
#         train_dataset.apply_upsampling()  # Balance training set
        
#         val_dataset = SEP28kDataset(f'./splits/{disfluency}/{disfluency}_val.csv',
#                                     target_disfluency=disfluency)
        
#         # Train model...
#         return trained_model
    
# block_model = train_disfluency_model('Block')
# soundrep_model = train_disfluency_model('SoundRep')
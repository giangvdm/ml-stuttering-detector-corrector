from src.model.Dataset import SEP28kDataset

dataset = SEP28kDataset('all_labels.csv', target_disfluency='Block')
# sample = dataset[14]
# print(f"Waveform shape: {sample['waveform'].shape}")
# print(f"Label: {sample['label']}")
# print(f"Label type: {type(sample['label'])}")
# print(f"Audio file: {sample['filepath']}")

disfluency_types = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection']

for disfluency in disfluency_types:
    splits = SEP28kDataset.create_stratified_splits(
        'all_labels.csv',
        disfluency,
        f'./splits/{disfluency}/'
    )
    print(f"Created splits for {disfluency}")

# Test the full workflow
dataset = SEP28kDataset('./splits/SoundRep/SoundRep_train.csv', target_disfluency='SoundRep')
dataset.get_dataset_info()

# Apply upsampling
dataset.apply_upsampling()
dataset.get_dataset_info()

# Get weights for loss function
pos_weight = dataset.get_class_weights()

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
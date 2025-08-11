import pandas as pd
import os
import kagglehub
from dotenv import load_dotenv

def preprocess_labels(input_csv_path, output_csv_path, base_audio_path):
    """
    Preprocesses the label data from a CSV file with quality filtering.
    
    This function reads a CSV file containing label information, applies
    strict quality filtering based on annotator agreement, and processes 
    the remaining data to create a structured DataFrame suitable for 
    frame-level disfluency detection.

    Args:
        input_csv_path (str): The path to the input CSV file.
        output_csv_path (str): The path to the output, processed CSV file.
        base_audio_path (str): The base directory path where audio clips are stored.

    Returns:
        bool: True if processing successful, False otherwise.
    """
    
    print(f"Reading raw annotation file from: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{input_csv_path}'. Please check the path.")
        return False

    print(f"Original dataset size: {len(df)} rows")
    print("--- Quality Filtering Pipeline ---")

    # --- 1. Aggressive filtering of problematic non-dysfluent labels ---
    # These labels indicate poor data quality - be very strict
    strict_filter_labels = ['Unsure', 'NoSpeech', 'PoorAudioQuality', 'Music']
    
    initial_size = len(df)
    for label in strict_filter_labels:
        if label in df.columns:
            # Drop any sample where even 1 annotator marked these labels
            df = df[df[label] == 0]
            print(f"  After removing '{label}' samples: {len(df)} rows (-{initial_size - len(df)})")
            initial_size = len(df)

    # --- 2. Define the target classes for dysfluencies ---
    dysfluency_classes = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection']
    fluent_classes = ['NoStutteredWords']  # Keep separate for clarity
    all_target_classes = dysfluency_classes + fluent_classes

    # --- 3. Apply ≥3 annotator agreement threshold for dysfluent labels ---
    print("\n--- Applying ≥3 Annotator Agreement Filtering ---")
    
    # Create boolean masks for high-confidence samples
    high_confidence_masks = {}
    for label in dysfluency_classes:
        if label in df.columns:
            high_confidence_masks[label] = df[label] >= 3
            positive_samples = (df[label] >= 3).sum()
            print(f"  {label}: {positive_samples} high-confidence positive samples")

    # For fluent samples, we want high confidence in the absence of disfluencies
    fluent_mask = (df['NoStutteredWords'] >= 3)
    high_confidence_masks['NoStutteredWords'] = fluent_mask
    fluent_samples = fluent_mask.sum()
    print(f"  NoStutteredWords: {fluent_samples} high-confidence fluent samples")

    # --- 4. Create quality-filtered dataset ---
    # Keep samples that have high confidence in AT LEAST ONE category
    any_high_confidence = pd.Series([False] * len(df), index=df.index)
    for mask in high_confidence_masks.values():
        any_high_confidence |= mask

    filtered_df = df[any_high_confidence].copy()
    print(f"\nDataset after quality filtering: {len(filtered_df)} rows")
    print(f"Reduction: {len(df) - len(filtered_df)} samples ({((len(df) - len(filtered_df))/len(df)*100):.1f}%)")

    # --- 5. Create processed data with binary labels ---
    processed_data = []
    
    print("\n--- Creating Binary Labels ---")
    for index, row in filtered_df.iterrows():
        # Construct the full file path
        filename = f"{row['Show']}_{row['EpId']}_{row['ClipId']}.wav"
        filepath = os.path.join(base_audio_path, filename)

        # Apply strict binary labeling: ≥2 annotators = 1, else = 0
        label_vector = {}
        for label in all_target_classes:
            if label in row:
                is_present = 1 if row[label] >= 2 else 0
                label_vector[label] = is_present
            else:
                label_vector[label] = 0

        # Store processed info
        processed_row = {'filepath': filepath}
        processed_row.update(label_vector)
        
        # Add metadata for potential future use
        processed_row['original_confidence'] = {
            label: row[label] for label in all_target_classes if label in row
        }
        
        processed_data.append(processed_row)

    # --- 6. Create final DataFrame and perform sanity checks ---
    processed_df = pd.DataFrame(processed_data)
    
    # Reorder columns for clarity
    final_columns = ['filepath'] + all_target_classes
    processed_df = processed_df[final_columns]

    # --- 7. Filter out data with missing audio files ---
    valid_files_mask = processed_df['filepath'].apply(os.path.exists)
    processed_df = processed_df[valid_files_mask].reset_index(drop=True)
    missing_count = (~valid_files_mask).sum()

    if missing_count > 0:
        print(f"Removed {missing_count} samples with missing audio files")
    else:
        print(f"No missing audio files - 0 rows removed")
    print(f"Dataset after missing file filtering: {len(processed_df)} rows")

    # --- 8. Data quality report ---
    print(f"\n--- Final Dataset Statistics ---")
    print(f"Total processed samples: {len(processed_df)}")
    print(f"Samples with at least one disfluency: {(processed_df[dysfluency_classes].sum(axis=1) > 0).sum()}")
    print(f"Purely fluent samples: {(processed_df['NoStutteredWords'] == 1).sum()}")
    
    print("\n--- Label Distribution (High-Confidence Only) ---")
    label_counts = processed_df[all_target_classes].sum()
    for label, count in label_counts.items():
        percentage = (count / len(processed_df)) * 100
        print(f"  {label}: {count} samples ({percentage:.1f}%)")

    # --- 9. Check for multi-label samples ---
    multi_disfluency_count = (processed_df[dysfluency_classes].sum(axis=1) > 1).sum()
    print(f"\nSamples with multiple disfluency types: {multi_disfluency_count}")
    
    # --- 10. Save processed data ---
    print(f"\nSaving processed data to: {output_csv_path}")
    processed_df.to_csv(output_csv_path, index=False)
    
    # Also save a copy locally for quick reference
    local_copy = "./all_labels.csv"
    processed_df.to_csv(local_copy, index=False)
    print(f"Local copy saved to: {local_copy}")
    
    # --- 11. Save quality report ---
    quality_report = {
        'original_samples': len(df),
        'final_samples': len(processed_df),
        'reduction_percentage': ((len(df) - len(processed_df))/len(df)*100),
        'label_distribution': label_counts.to_dict(),
        'multi_label_samples': multi_disfluency_count
    }
    
    report_path = output_csv_path.replace('.csv', '_quality_report.txt')
    with open(report_path, 'w') as f:
        f.write("SEP-28k Quality Filtering Report\n")
        f.write("=" * 40 + "\n\n")
        for key, value in quality_report.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Quality report saved to: {report_path}")
    
    return True


def validate_audio_files(csv_path, sample_size=100):
    """
    Validates that audio files referenced in the CSV actually exist.
    
    Args:
        csv_path (str): Path to the processed CSV file
        sample_size (int): Number of files to check (random sample)
    """
    print(f"\n--- Audio File Validation ---")
    df = pd.read_csv(csv_path)
    
    if len(df) < sample_size:
        sample_size = len(df)
    
    sample_df = df.sample(n=sample_size)
    missing_files = []
    
    for idx, row in sample_df.iterrows():
        if not os.path.exists(row['filepath']):
            missing_files.append(row['filepath'])
    
    print(f"Checked {sample_size} audio files")
    print(f"Missing files: {len(missing_files)}")
    
    if missing_files:
        print("First 5 missing files:")
        for file in missing_files[:5]:
            print(f"  {file}")
    
    return len(missing_files) == 0


if __name__ == "__main__":
    load_dotenv()

    dataset_root_dir = os.getenv("DATASET_ROOT_DIR")
    audio_root_dir = os.getenv("AUDIO_ROOT_DIR")
    real_audio_root_dir = os.path.join(dataset_root_dir, audio_root_dir)
    merged_labels_file = os.getenv("MERGED_LABEL_FILE_NAME")
    merged_labels_file = os.path.join(dataset_root_dir, merged_labels_file)
    all_labels_file = os.getenv("PROCESSED_LABEL_FILE_NAME")
    all_labels_file = os.path.join(dataset_root_dir, all_labels_file)
    
    output_path = os.path.join(dataset_root_dir, all_labels_file)

    # Download latest version
    path = kagglehub.dataset_download("vudominhgiang/sep-28k-maintained")
    print("Path to dataset files:", path)
    
    # Process labels with quality filtering
    success = preprocess_labels(merged_labels_file, output_path, real_audio_root_dir)
    
    if success:
        # Validate a sample of audio files
        validate_audio_files(output_path, sample_size=50)
        
        print("\n" + "="*50)
        print("PREPROCESSING COMPLETE")
        print("="*50)
    else:
        print("ERROR: Preprocessing failed!")
import pandas as pd
import os
import kagglehub
from dotenv import load_dotenv

def preprocess_labels(input_csv_path, output_csv_path, base_audio_path):
    """
    Preprocesses the label data from a CSV file.

    This function reads a CSV file containing label information, filters out
    unwanted labels, and processes the remaining data to create a
    structured DataFrame.

    Args:
        input_csv_path (str): The path to the input CSV file.
        output_csv_path (str): The path to the output, processed CSV file.
        base_audio_path (str): The base directory path where audio clips are stored.

    Returns:
        pandas.DataFrame: A DataFrame containing the preprocessed label data.
    """
    
    print(f"Reading raw annotation file from: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{input_csv_path}'. Please check the path.")
        return None

    print(f"Original dataset size: {len(df)} rows")

    # --- 1. Drop records with least distributed, irrelevant labels ---
    labels_to_drop = ['Unsure', 'NoSpeech', 'PoorAudioQuality', 'Music']
    
    # Drop rows where any of these labels have a value > 0
    for label in labels_to_drop:
        if label in df.columns:
            df = df[df[label] == 0]
            print(f"Dataset size after dropping '{label}' records: {len(df)} rows")

    # --- 2. Define the target classes for dysfluencies ---
    dysfluency_classes = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']

    # --- 3. Create a new DataFrame to store processed data ---
    processed_data = []

    # --- 4. Iterate over each row to create file paths and multi-hot labels ---
    for index, row in df.iterrows():
        # a. Construct the full file path
        filename = f"{row['Show']}_{row['EpId']}_{row['ClipId']}.wav"
        filepath = os.path.join(base_audio_path, filename)

        # b. Binarize the labels (any value > 0 becomes 1)
        label_vector = {}
        for label in dysfluency_classes:
            # If at least one annotator marked the label, we consider it present.
            is_present = 1 if row[label] > 0 else 0
            label_vector[label] = is_present

        # d. Append the processed info to our list
        processed_row = {'filepath': filepath}
        processed_row.update(label_vector)
        processed_data.append(processed_row)

    # --- 5. Create the final DataFrame ---
    processed_df = pd.DataFrame(processed_data)
    
    # Reorder columns for clarity
    final_columns = ['filepath'] + dysfluency_classes
    processed_df = processed_df[final_columns]

    print(f"Processed {len(processed_df)} records from {input_csv_path}")
    print("\n--- Sample of Merged Data ---")
    print(processed_df.head())
    print("--- Label Distribution ---")
    print(processed_df[dysfluency_classes].sum())

    print(f"Saving merged data to: {output_csv_path}")
    processed_df.to_csv(output_csv_path, index=False)
    print(f"Also saved a copy to current directory for quick reference")
    processed_df.to_csv("./all_labels.csv", index=False)
    
    return True


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
    
    preprocess_labels(merged_labels_file, output_path, real_audio_root_dir)
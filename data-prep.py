import pandas as pd
import os

def preprocess_sep28k_labels(input_csv_path, output_csv_path, clips_root_dir="clips"):
    """
    Preprocesses the raw SEP-28k annotation CSV for multi-label classification.

    Args:
        input_csv_path (str): Path to the original SEP-28k labels CSV file.
        output_csv_path (str): Path to save the new, processed CSV file.
        clips_root_dir (str): The root directory where the audio clips are stored.
    """
    print(f"Reading raw annotation file from: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{input_csv_path}'. Please check the path.")
        return

    print(f"Original dataset size: {len(df)} rows")

    # --- 1.1 Filter out clips with any non-dysfluency or quality issue labels ---
    # Define all columns that indicate an issue or are not a target dysfluency.
    quality_issue_cols = [
        'PoorAudioQuality', 'Unsure', 'DifficultToUnderstand',
        'NaturalPause', 'Music', 'NoSpeech'
    ]
    # Keep rows only if the sum of all these columns is 0.
    # This ensures no annotator has flagged the clip with any of these labels.
    df = df[df[quality_issue_cols].sum(axis=1) == 0]
    print(f"Dataset size after filtering for quality and non-dysfluency flags: {len(df)} rows")

    # --- 1.2 Filter for specific shows ---
    # Define the list of show names you want to keep.
    shows_to_keep = ["HeStutters", "HVSA", "MyStutteringLife", "StutterTalk", "WomenWhoStutter"] 
    df = df[df['Show'].isin(shows_to_keep)]
    print(f"Dataset size after filtering for specific shows {shows_to_keep}: {len(df)} rows")

    # --- 2. Define the target classes for dysfluencies ---
    dysfluency_classes = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection']
    all_classes = dysfluency_classes + ['Fluent']

    # --- 3. Create a new DataFrame to store processed data ---
    processed_data = []

    # --- 4. Iterate over each row to create file paths and multi-hot labels ---
    for index, row in df.iterrows():
        # a. Construct the full file path
        filename = f"{row['Show']}_{row['EpId']}_{row['ClipId']}.wav"
        filepath = os.path.join(clips_root_dir, row['Show'], str(row['EpId']), filename)

        # b. Binarize the labels (any value > 0 becomes 1)
        is_dysfluent = False
        label_vector = {}
        for label in dysfluency_classes:
            # If at least one annotator marked the label, we consider it present.
            is_present = 1 if row[label] > 0 else 0
            if is_present == 1:
                is_dysfluent = True
            label_vector[label] = is_present

        # c. Determine the 'Fluent' label
        # A clip is considered fluent if no dysfluency labels were marked.
        label_vector['Fluent'] = 1 if not is_dysfluent else 0

        # d. Append the processed info to our list
        processed_row = {'filepath': filepath}
        processed_row.update(label_vector)
        processed_data.append(processed_row)

    # --- 5. Create the final DataFrame and save it ---
    processed_df = pd.DataFrame(processed_data)
    
    # Reorder columns for clarity
    final_columns = ['filepath'] + all_classes
    processed_df = processed_df[final_columns]

    print(f"\nSaving processed data to: {output_csv_path}")
    processed_df.to_csv(output_csv_path, index=False)
    print("Preprocessing complete.")
    print("\n--- Sample of Processed Data ---")
    print(processed_df.head())
    print("\n--- Label Distribution ---")
    print(processed_df[all_classes].sum())


if __name__ == "__main__":
    # Define the input and output file paths
    raw_labels_file = "ml-stuttering-events-dataset/SEP-28k_labels.csv"
    processed_labels_file = "sep28k_labels_processed.csv"

    preprocess_sep28k_labels(raw_labels_file, processed_labels_file)

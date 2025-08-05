import pandas as pd
import whisper
import torch
import os
from tqdm import tqdm
from dotenv import load_dotenv

def generate_transcripts(input_csv, output_csv, audio_root_dir):
    """
    Generates transcripts for all audio files in a processed CSV and saves them to a new file.

    Args:
        input_csv (str): Path to the processed labels CSV.
        output_csv (str): Path to save the new CSV with a 'transcript' column.
        audio_root_dir (str): The root directory where all audio files reside.
    """
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"ERROR: Input CSV not found at '{input_csv}'. Please run the initial preprocessing script first.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading Whisper model onto device: {device}")
    asr_model = whisper.load_model("base", device=device)

    transcripts = []
    print(f"Generating transcripts for {len(df)} audio files...")

    # Use tqdm for a progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Transcribing"):
        audio_path = os.path.join(audio_root_dir, row['filepath'])
        if os.path.exists(audio_path):
            try:
                result = asr_model.transcribe(audio_path)
                transcripts.append(result['text'])
            except Exception as e:
                print(f"Could not transcribe {audio_path}. Error: {e}")
                transcripts.append("")
        else:
            print(f"File not found: {audio_path}")
            transcripts.append("")

    # Add the new transcript column to the DataFrame
    df['transcript'] = transcripts

    print(f"\nSaving new dataset with transcripts to: {output_csv}")
    df.to_csv(output_csv, index=False)
    print("Transcription complete.")
    print("\n--- Sample of new data ---")
    print(df.head())

if __name__ == "__main__":
    load_dotenv()

    dataset_root_dir = os.getenv("DATASET_ROOT_DIR")
    audio_root_dir = os.getenv("AUDIO_ROOT_DIR")
    real_audio_root_dir = os.path.join(dataset_root_dir, audio_root_dir)
    all_labels_file = os.getenv("PROCESSED_LABEL_FILE_NAME")
    transcribed_labels_file = os.getenv("TRANSCRIBED_LABEL_FILE_NAME")
    
    generate_transcripts(all_labels_file, transcribed_labels_file, real_audio_root_dir)

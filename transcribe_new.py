import pandas as pd
import whisper
import torch
import os
from tqdm import tqdm
from dotenv import load_dotenv

def generate_transcripts(input_csv, output_csv, audio_root_dir, model_size="medium"):
    """
    Generates transcripts for short audio clips (2-3 seconds) and saves them to a new file.
    Optimized for short clips with better Whisper settings.

    Args:
        input_csv (str): Path to the processed labels CSV.
        output_csv (str): Path to save the new CSV with a 'transcript' column.
        audio_root_dir (str): The root directory where all audio files reside.
        model_size (str): Whisper model size to use.
    """
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"ERROR: Input CSV not found at '{input_csv}'. Please run the initial preprocessing script first.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading Whisper model '{model_size}' onto device: {device}")
    
    try:
        asr_model = whisper.load_model(model_size, device=device)
    except Exception as e:
        print(f"Error loading model {model_size}: {e}")
        print("Falling back to 'base' model...")
        asr_model = whisper.load_model("base", device=device)

    transcripts = []
    failed_files = []
    print(f"Generating transcripts for {len(df)} short audio clips...")

    # Use tqdm for a progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Transcribing"):
        audio_path = os.path.join(audio_root_dir, row['filepath'])
        
        if os.path.exists(audio_path):
            try:
                # Load and normalize audio to handle loud/crackling audio
                audio = whisper.load_audio(audio_path)
                
                # Simple normalization to prevent clipping from loud audio
                if len(audio) > 0:
                    max_val = max(abs(audio.max()), abs(audio.min()))
                    if max_val > 0:
                        audio = audio / max_val * 0.9  # Normalize to 90% to prevent clipping
                
                # For short clips, use settings optimized for brief audio
                result = asr_model.transcribe(
                    audio,
                    language='en',  # Specify language if known
                    task='transcribe',
                    temperature=0.2,  # Slightly less deterministic for short clips
                    best_of=3,  # Fewer attempts for short clips
                    beam_size=3,  # Smaller beam for efficiency
                    condition_on_previous_text=False,  # Important for short clips
                    compression_ratio_threshold=1.8,  # Lower threshold for short clips
                    logprob_threshold=-0.8,  # Less strict for short clips
                    no_speech_threshold=0.5,  # Lower threshold to catch quiet speech
                    word_timestamps=False  # Don't need word-level timing for short clips
                )
                
                transcript = result['text'].strip()
                transcripts.append(transcript)
                
                # Debug: print empty transcriptions
                if not transcript:
                    failed_files.append(audio_path)
                    
            except Exception as e:
                print(f"Could not transcribe {audio_path}. Error: {e}")
                transcripts.append("")
                failed_files.append(audio_path)
        else:
            print(f"File not found: {audio_path}")
            transcripts.append("")
            failed_files.append(audio_path)

    # Add the new transcript column to the DataFrame
    df['Transcript'] = transcripts

    print(f"\nSaving new dataset with transcripts to: {output_csv}")
    df.to_csv(output_csv, index=False)
    
    # Print statistics
    successful_transcripts = sum(1 for t in transcripts if t.strip())
    print(f"Transcription complete. Success rate: {successful_transcripts}/{len(transcripts)} ({successful_transcripts/len(transcripts)*100:.1f}%)")
    
    if failed_files:
        print(f"\n{len(failed_files)} files failed transcription or returned empty results:")
        for i, failed_file in enumerate(failed_files[:10]):  # Show first 10
            print(f"  {failed_file}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    print("\n--- Sample of new data ---")
    print(df.head())

if __name__ == "__main__":
    load_dotenv()

    dataset_root_dir = os.getenv("DATASET_ROOT_DIR")
    audio_root_dir = os.getenv("AUDIO_ROOT_DIR")
    real_audio_root_dir = os.path.join(dataset_root_dir, audio_root_dir)
    all_labels_file = os.getenv("PROCESSED_LABEL_FILE_NAME")
    transcribed_labels_file = os.getenv("TRANSCRIBED_LABEL_FILE_NAME")
    
    # Model size
    # You can try "large" or "large-v2" if you have GPU memory
    model_size = "medium"
    
    generate_transcripts(all_labels_file, transcribed_labels_file, real_audio_root_dir, model_size)
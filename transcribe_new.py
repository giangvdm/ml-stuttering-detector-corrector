import pandas as pd
import torch
import os
import numpy as np
import tempfile
import subprocess
import time
from datetime import timedelta
import re
from tqdm import tqdm
from dotenv import load_dotenv
import whisper
from transformers import pipeline
import librosa
from typing import List, Tuple, Optional

# Import punctuation restoration if available
try:
    from src.utils.punc import SbertPuncCase, PUNCTUATION
    PUNC_AVAILABLE = True
except ImportError:
    PUNC_AVAILABLE = False
    print("Warning: punc.py not available. Punctuation restoration will be skipped.")

# Audio and video file extensions
audio_exts = ['mp3', 'aac', 'wav', 'ogg', 'm4a', 'opus']
video_exts = ['mp4', 'mov', 'avi', 'mkv', 'webm']

def extract_audio_from_video(video_path: str) -> str:
    """
    Extract audio from video file using FFmpeg and save it as a temporary WAV file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Path to the temporary audio file
    """
    print(f"Extracting audio from video file...", end="")
    try:
        # Create a temporary file for the audio
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio.close()
        
        # Use FFmpeg to extract audio
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # Disable video
            '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
            '-ar', '16000',  # Sample rate 16kHz
            '-ac', '1',  # Mono audio
            '-y',  # Overwrite output file
            temp_audio.name
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        print("done")
        return temp_audio.name
    except subprocess.CalledProcessError as e:
        print(f"failed: {e.stderr.decode()}")
        raise
    except Exception as e:
        print(f"failed: {e}")
        raise

def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file in seconds using FFmpeg.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Duration in seconds
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', audio_path,
            '-f', 'null',
            '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse FFmpeg output to find duration
        for line in result.stderr.split('\n'):
            if 'Duration:' in line:
                time_str = line.split('Duration:')[1].split(',')[0].strip()
                h, m, s = time_str.split(':')
                duration = int(h) * 3600 + int(m) * 60 + float(s)
                return duration
        return 0
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return 0

def match_ext(filename: str, extensions):
    """Check if file has matching extension"""
    return filename.lower().split('.')[-1] in extensions

def fix_punctuation(text: str, sbertpunc) -> str:
    """
    Fix punctuation and case using the SbertPuncCase model.
    """
    if not PUNC_AVAILABLE or sbertpunc is None:
        return text
        
    text = text.splitlines()
    target_text = ''
    chunks = 0
    
    for sentence in text:
        contains_punc = False
        for sign in PUNCTUATION:
            if sign in sentence:
                contains_punc = True
        
        if (len(sentence) > 80 and contains_punc == False) or len(sentence) > 300:
            chunks += 1
            punctuated_text = sbertpunc.punctuate(sentence.strip(PUNCTUATION))
            punctuated_text_nodoubles = re.sub(",([,\.\!\?])", "\\1", punctuated_text)
            punctuated_text_lines = re.sub("([\.\!\?]) ", "\\1\n", punctuated_text_nodoubles)
            target_text += punctuated_text_lines + '\n'
        else:
            target_text += sentence + '\n'
    
    if chunks > 0:
        print(f"Fixed punctuation and case in {chunks} chunks")
    return target_text

def load_and_preprocess_audio(audio_path: str) -> Optional[np.ndarray]:
    """
    Load and preprocess a single audio file.
    
    Args:
        audio_path (str): Path to the audio file
        
    Returns:
        np.ndarray: Preprocessed audio array, or None if failed
    """
    try:
        # Load audio with librosa for better quality
        try:
            # Load at 16kHz (common sample rate for speech models)
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            audio = audio.astype(np.float32)
        except ImportError:
            # Fallback method without librosa
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            if sr != 16000:
                print(f"Warning: Audio sample rate is {sr}, expected 16000")
            audio = audio.astype(np.float32)
        
        # Better audio preprocessing
        if len(audio) > 0:
            # Remove DC offset
            audio = audio - np.mean(audio)
            
            # RMS-based normalization (gentler than peak normalization)
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                target_rms = 0.1  # Conservative target
                audio = audio * (target_rms / rms)
            
            # Clip to prevent any overflow
            audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return None

def process_single_audio_file(audio_path: str, model, model_type: str, language: str = None, 
                            raw_output: bool = False, sbertpunc=None) -> Tuple[str, bool]:
    """
    Process a single audio file and return transcript.
    
    Args:
        audio_path: Path to the audio file
        model: Loaded transcription model
        model_type: 'whisper' or 'huggingface'
        language: Language code for transcription (optional)
        raw_output: Whether to skip punctuation restoration
        sbertpunc: Punctuation restoration model
        
    Returns:
        Tuple of (transcript, success_flag)
    """
    try:
        file_ext = audio_path.split('.')[-1].lower()
        temp_audio_file = None
        should_cleanup = False
        
        # Handle video files by extracting audio
        if file_ext in video_exts:
            try:
                temp_audio_file = extract_audio_from_video(audio_path)
                actual_audio_path = temp_audio_file
                should_cleanup = True
            except Exception as e:
                print(f"Error extracting audio from video {audio_path}: {e}")
                return "", False
        else:
            actual_audio_path = audio_path
        
        # Check if file exists
        if not os.path.exists(actual_audio_path):
            print(f"File not found: {actual_audio_path}")
            return "", False
        
        # Get audio duration for statistics
        audio_duration = get_audio_duration(actual_audio_path)
        
        # Start timing
        start_time = time.time()
        
        # Process based on model type
        if model_type == 'huggingface':
            result = model(
                actual_audio_path,
                return_timestamps=True,
            )
            segments = []
            for chunk in result["chunks"]:
                if chunk["timestamp"][0] is not None:
                    segments.append({
                        "start": chunk["timestamp"][0],
                        "text": chunk["text"]
                    })
                else:
                    segments.append({
                        "start": 0,
                        "text": chunk["text"]
                    })
        else:  # whisper
            result = model.transcribe(actual_audio_path, verbose=False, language=language)
            segments = result['segments']
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Extract and clean text
        rawtext = ' '.join([segment['text'].strip() for segment in segments])
        rawtext = re.sub(" +", " ", rawtext)
        
        # Apply basic sentence splitting
        alltext = re.sub("([\.\!\?]) ", "\\1\n", rawtext)
        
        # Apply punctuation restoration if not raw output
        if not raw_output:
            alltext = fix_punctuation(alltext, sbertpunc)
        
        # Clean up final text
        final_text = alltext.strip()
        final_text = ' '.join(final_text.split())  # Normalize whitespace
        
        return final_text, True
        
    except Exception as e:
        print(f"Error processing audio {audio_path}: {e}")
        return "", False
    finally:
        # Clean up temporary audio file if it was created
        if should_cleanup and temp_audio_file and os.path.exists(temp_audio_file):
            try:
                os.unlink(temp_audio_file)
            except:
                pass

def process_batch(audio_paths: List[str], model, model_type: str, language: str = None,
                 raw_output: bool = False, sbertpunc=None) -> List[Tuple[str, bool]]:
    """
    Process a batch of audio files and return transcripts.
    
    Args:
        audio_paths: List of audio file paths
        model: Loaded transcription model
        model_type: 'whisper' or 'huggingface'
        language: Language code for transcription (optional)
        raw_output: Whether to skip punctuation restoration
        sbertpunc: Punctuation restoration model
        
    Returns:
        List of (transcript, success_flag) tuples
    """
    results = []
    
    for audio_path in audio_paths:
        transcript, success = process_single_audio_file(
            audio_path, model, model_type, language, raw_output, sbertpunc
        )
        results.append((transcript, success))
    
    return results

def generate_transcripts(input_csv, output_csv, audio_root_dir, model_name="large-v3-turbo", 
                        batch_size=32, language=None, raw_output=False, device='cuda'):
    """
    Generates transcripts for audio clips using Whisper or HuggingFace models with batch processing.
    Mimics the approach from batch-speech-to-text.py while maintaining the current pipeline structure.

    Args:
        input_csv (str): Path to the processed labels CSV.
        output_csv (str): Path to save the new CSV with a 'transcript' column.
        audio_root_dir (str): The root directory where all audio files reside.
        model_name (str): Model name (Whisper model or HuggingFace model path).
        batch_size (int): Number of audio files to process in each batch.
        language (str): Language code for transcription (optional).
        raw_output (bool): Whether to skip punctuation restoration.
        device (str): Device to run on ('cpu' or 'cuda').
    """
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"ERROR: Input CSV not found at '{input_csv}'. Please run the initial preprocessing script first.")
        return

    # Determine if the model is from Hugging Face by checking if the name contains '/'
    use_huggingface = '/' in model_name
    model_type = 'huggingface' if use_huggingface else 'whisper'
    
    # Set device
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Load punctuation restoration model if needed
    sbertpunc = None
    if not raw_output and PUNC_AVAILABLE:
        print("Loading punctuation and case model...", end=" ")
        try:
            sbertpunc = SbertPuncCase().to("cpu")
            print("done")
        except Exception as e:
            print(f"failed: {e}")
            print("Continuing without punctuation restoration...")
    
    lang_text = f"to {language} " if language else ""
    print(f"Starting speech-to-text recognition {lang_text}on {device} device")
    print(f"Model type: {model_type}")
    print(f"Batch size: {batch_size}")
    
    # Load transcription model
    if use_huggingface:
        print(f"Loading huggingface model: {model_name}...", end=" ")
        try:
            model = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device=device
            )
            print("done")
        except Exception as e:
            print(f"failed: {e}")
            return
    else:
        print(f"Loading whisper model: {model_name}...", end=" ")
        try:
            model = whisper.load_model(model_name).to(device)
            print("done")
        except Exception as e:
            print(f"failed: {e}")
            return

    transcripts = []
    failed_files = []
    empty_transcripts = []
    successful_transcripts = 0
    total_files = len(df)
    
    print(f"Generating transcripts for {total_files} audio files...")

    # Process in batches
    num_batches = (total_files + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_files)
        batch_rows = df.iloc[start_idx:end_idx]
        
        # Prepare batch paths
        batch_paths = []
        for idx, row in batch_rows.iterrows():
            audio_path = os.path.join(audio_root_dir, row['filepath'])
            batch_paths.append(audio_path)
        
        # Process batch
        batch_results = process_batch(
            batch_paths, model, model_type, language, raw_output, sbertpunc
        )
        
        # Collect results
        for i, (transcript, success) in enumerate(batch_results):
            if success and transcript.strip():
                transcripts.append(transcript)
                successful_transcripts += 1
            elif success and not transcript.strip():
                transcripts.append("")
                empty_transcripts.append(batch_paths[i])
            else:
                transcripts.append("")
                failed_files.append(batch_paths[i])
    
    # Add the new transcript column to the DataFrame
    df['Transcript'] = transcripts

    print(f"\nSaving new dataset with transcripts to: {output_csv}")
    df.to_csv(output_csv, index=False)
    
    # Print detailed statistics
    print(f"\n=== TRANSCRIPTION RESULTS ===")
    print(f"Total files processed: {total_files}")
    print(f"Successful transcriptions: {successful_transcripts}")
    print(f"Empty transcriptions: {len(empty_transcripts)}")
    print(f"Failed transcriptions: {len(failed_files)}")
    print(f"Success rate: {successful_transcripts/total_files*100:.1f}%")
    print(f"Batches processed: {num_batches}")
    print(f"Average files per batch: {total_files/num_batches:.1f}")
    
    if empty_transcripts:
        print(f"\n{len(empty_transcripts)} files returned empty transcriptions:")
        for i, empty_file in enumerate(empty_transcripts[:5]):  # Show first 5
            print(f"  {empty_file}")
        if len(empty_transcripts) > 5:
            print(f"  ... and {len(empty_transcripts) - 5} more")
    
    if failed_files:
        print(f"\n{len(failed_files)} files failed due to errors:")
        for i, failed_file in enumerate(failed_files[:5]):  # Show first 5
            print(f"  {failed_file}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")
    
    print("\n--- Sample of new data ---")
    print(df[['filepath', 'Transcript']].head(10))
    
    # Show some example transcripts
    successful_samples = df[df['Transcript'].str.len() > 0]['Transcript'].head(5)
    if len(successful_samples) > 0:
        print("\n--- Example successful transcripts ---")
        for i, transcript in enumerate(successful_samples, 1):
            print(f"{i}. \"{transcript}\"")

if __name__ == "__main__":
    load_dotenv()

    dataset_root_dir = os.getenv("DATASET_ROOT_DIR")
    audio_root_dir = os.getenv("AUDIO_ROOT_DIR")
    real_audio_root_dir = os.path.join(dataset_root_dir, audio_root_dir)
    all_labels_file = os.getenv("PROCESSED_LABEL_FILE_NAME")
    transcribed_labels_file = os.getenv("TRANSCRIBED_LABEL_FILE_NAME")
    
    # Model options:
    # Whisper models: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3-turbo"
    # HuggingFace models: "openai/whisper-large-v3", "openai/whisper-medium", etc.
    model_name = "large-v3-turbo"
    
    # Language code (optional) - e.g., "en", "ru", "es", etc.
    language = "en"
    
    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Batch size - adjust based on your system capabilities
    # For longer audio files, use smaller batch sizes
    batch_size = 128
    
    # Raw output (skip punctuation restoration)
    raw_output = False
    
    generate_transcripts(
        all_labels_file, 
        transcribed_labels_file, 
        real_audio_root_dir, 
        model_name=model_name,
        batch_size=batch_size,
        language=language,
        raw_output=raw_output,
        device=device
    )
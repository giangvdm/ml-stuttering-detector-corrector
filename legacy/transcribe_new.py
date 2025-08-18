import pandas as pd
import torch
import os
import numpy as np
import tempfile
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import re
from tqdm import tqdm
from dotenv import load_dotenv
import whisper
from transformers import pipeline
from typing import List, Tuple, Optional, Dict
import gc
import warnings
warnings.filterwarnings("ignore")

# Import punctuation restoration if available
try:
    from src.utils.punc import SbertPuncCase, PUNCTUATION
    PUNC_AVAILABLE = True
except ImportError:
    PUNC_AVAILABLE = False
    print("Warning: punc.py not available. Punctuation restoration will be skipped.")

# Audio and video file extensions
AUDIO_EXTS = {'mp3', 'aac', 'wav', 'ogg', 'm4a', 'opus'}
VIDEO_EXTS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}

class OptimizedTranscriber:
    """Optimized transcription pipeline with parallel processing and memory management."""
    
    def __init__(self, model_name: str = "large-v3-turbo", device: str = 'cuda', 
                 num_workers: int = None, use_memory_optimization: bool = True):
        self.model_name = model_name
        self.device = device
        self.use_huggingface = '/' in model_name
        self.model_type = 'huggingface' if self.use_huggingface else 'whisper'
        self.num_workers = num_workers or min(mp.cpu_count(), 4)
        self.use_memory_optimization = use_memory_optimization
        
        # Set device
        if not torch.cuda.is_available() and device == 'cuda':
            print("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load transcription and punctuation models with memory optimization."""
        print(f"Loading {self.model_type} model: {self.model_name}...", end=" ")
        
        try:
            if self.use_huggingface:
                # Use torch_dtype for memory optimization
                model_kwargs = {"device": self.device}
                if self.use_memory_optimization and torch.cuda.is_available():
                    model_kwargs["torch_dtype"] = torch.float16
                
                self.model = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_name,
                    **model_kwargs
                )
            else:
                self.model = whisper.load_model(self.model_name).to(self.device)
                
                # Enable memory optimization for Whisper
                if self.use_memory_optimization and hasattr(self.model, 'half'):
                    self.model = self.model.half()
            
            print("done")
        except Exception as e:
            print(f"failed: {e}")
            raise
        
        # Load punctuation model
        self.sbertpunc = None
        if PUNC_AVAILABLE:
            print("Loading punctuation model...", end=" ")
            try:
                self.sbertpunc = SbertPuncCase().to("cpu")
                print("done")
            except Exception as e:
                print(f"failed: {e}")
    
    @staticmethod
    def extract_audio_ffmpeg(video_path: str) -> str:
        """Optimized audio extraction using FFmpeg with minimal processing."""
        try:
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio.close()
            
            # Optimized FFmpeg command - let Whisper handle resampling
            cmd = [
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1', '-y', temp_audio.name,
                '-hide_banner', '-loglevel', 'error'  # Reduce output noise
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            return temp_audio.name
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg extraction failed: {e.stderr.decode()}")
    
    def fix_punctuation_batch(self, texts: List[str]) -> List[str]:
        """Batch punctuation restoration for better efficiency."""
        if not PUNC_AVAILABLE or self.sbertpunc is None:
            return texts
        
        results = []
        for text in texts:
            if len(text) > 80 and not any(p in text for p in PUNCTUATION):
                try:
                    punctuated = self.sbertpunc.punctuate(text.strip(PUNCTUATION))
                    punctuated = re.sub(",([,\.\!\?])", "\\1", punctuated)
                    punctuated = re.sub("([\.\!\?]) ", "\\1\n", punctuated)
                    results.append(punctuated)
                except:
                    results.append(text)
            else:
                results.append(text)
        
        return results
    
    def transcribe_single_file(self, file_info: Dict) -> Tuple[str, bool, Dict]:
        """Optimized single file transcription with minimal preprocessing."""
        audio_path = file_info['path']
        file_ext = audio_path.split('.')[-1].lower()
        temp_file = None
        
        try:
            # Handle video files
            if file_ext in VIDEO_EXTS:
                temp_file = self.extract_audio_ffmpeg(audio_path)
                actual_path = temp_file
            else:
                actual_path = audio_path
            
            if not os.path.exists(actual_path):
                return "", False, {"error": "File not found"}
            
            start_time = time.time()
            
            # Transcribe with optimized settings
            if self.model_type == 'huggingface':
                # Use chunk_length_s for better memory management
                result = self.model(
                    actual_path,
                    return_timestamps=True,
                    chunk_length_s=30,  # Process in 30s chunks
                    stride_length_s=5   # 5s overlap
                )
                segments = [{"text": chunk["text"]} for chunk in result["chunks"]]
            else:
                # Whisper with optimized parameters
                options = {
                    "verbose": False,
                    "temperature": 0,  # Deterministic output
                    "compression_ratio_threshold": 2.4,
                    "logprob_threshold": -1.0,
                    "no_speech_threshold": 0.6
                }
                
                if file_info.get('language'):
                    options["language"] = file_info['language']
                
                result = self.model.transcribe(actual_path, **options)
                segments = result['segments']
            
            # Extract and clean text efficiently
            text_parts = [seg['text'].strip() for seg in segments if seg['text'].strip()]
            raw_text = ' '.join(text_parts)
            
            # Basic cleanup
            clean_text = re.sub(r'\s+', ' ', raw_text).strip()
            
            processing_time = time.time() - start_time
            
            return clean_text, True, {
                "processing_time": processing_time,
                "segments": len(segments)
            }
            
        except Exception as e:
            return "", False, {"error": str(e)}
        finally:
            # Cleanup
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    def process_batch_parallel(self, file_batch: List[Dict], language: str = None) -> List[Tuple[str, bool, Dict]]:
        """Process a batch of files with parallel execution."""
        # Add language to file info
        for file_info in file_batch:
            file_info['language'] = language
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=min(len(file_batch), self.num_workers)) as executor:
            results = list(executor.map(self.transcribe_single_file, file_batch))
        
        return results
    
    def generate_transcripts_optimized(self, input_csv: str, output_csv: str, 
                                     audio_root_dir: str, batch_size: int = 32,
                                     language: str = None, raw_output: bool = False,
                                     save_intermediate: bool = True) -> Dict:
        """
        Optimized transcript generation with parallel processing and memory management.
        """
        try:
            df = pd.read_csv(input_csv)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        
        total_files = len(df)
        print(f"Processing {total_files} files with optimized pipeline")
        print(f"Model: {self.model_name} ({self.model_type})")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")
        print(f"Workers: {self.num_workers}")
        
        # Prepare file information
        file_infos = [
            {"path": os.path.join(audio_root_dir, row['filepath']), "index": idx}
            for idx, row in df.iterrows()
        ]
        
        # Process in batches
        transcripts = [""] * total_files
        stats = {
            "successful": 0,
            "failed": 0,
            "empty": 0,
            "total_time": 0,
            "failed_files": []
        }
        
        start_total = time.time()
        
        # Calculate number of batches
        num_batches = (total_files + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_files)
            batch_files = file_infos[start_idx:end_idx]
            
            # Process batch
            batch_results = self.process_batch_parallel(batch_files, language)
            
            # Collect results
            for i, (transcript, success, metadata) in enumerate(batch_results):
                file_idx = start_idx + i
                
                if success and transcript.strip():
                    transcripts[file_idx] = transcript
                    stats["successful"] += 1
                elif success and not transcript.strip():
                    stats["empty"] += 1
                else:
                    stats["failed"] += 1
                    stats["failed_files"].append(batch_files[i]["path"])
                
                if "processing_time" in metadata:
                    stats["total_time"] += metadata["processing_time"]
            
            # Memory cleanup every few batches
            if batch_idx % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Save intermediate results
            if save_intermediate and batch_idx % 10 == 0:
                temp_df = df.copy()
                temp_df['Transcript'] = transcripts
                temp_df.to_csv(f"{output_csv}.tmp", index=False)
        
        # Apply punctuation restoration in batches if needed
        if not raw_output and self.sbertpunc:
            print("Applying punctuation restoration...")
            valid_transcripts = [t for t in transcripts if t.strip()]
            if valid_transcripts:
                corrected = self.fix_punctuation_batch(valid_transcripts)
                # Update transcripts with corrections
                corrected_idx = 0
                for i, transcript in enumerate(transcripts):
                    if transcript.strip():
                        transcripts[i] = corrected[corrected_idx]
                        corrected_idx += 1
        
        # Save final results
        df['Transcript'] = transcripts
        df.to_csv(output_csv, index=False)
        
        stats["total_processing_time"] = time.time() - start_total
        stats["success_rate"] = stats["successful"] / total_files * 100
        
        # Clean up temporary file
        temp_file = f"{output_csv}.tmp"
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        
        return stats


def generate_transcripts_optimized(input_csv: str, output_csv: str, audio_root_dir: str, 
                                 model_name: str = "large-v3-turbo", batch_size: int = 64,
                                 language: str = None, raw_output: bool = False, 
                                 device: str = 'cuda', num_workers: int = None) -> Dict:
    """
    Optimized wrapper function for transcript generation.
    
    Key optimizations:
    1. Parallel processing within batches
    2. Memory-efficient model loading
    3. Reduced audio preprocessing
    4. Batch punctuation restoration
    5. Intermediate saving for long runs
    """
    
    transcriber = OptimizedTranscriber(
        model_name=model_name,
        device=device,
        num_workers=num_workers
    )
    
    stats = transcriber.generate_transcripts_optimized(
        input_csv=input_csv,
        output_csv=output_csv,
        audio_root_dir=audio_root_dir,
        batch_size=batch_size,
        language=language,
        raw_output=raw_output
    )
    
    # Print results
    print(f"\n=== OPTIMIZATION RESULTS ===")
    print(f"Total files: {stats['successful'] + stats['failed'] + stats['empty']}")
    print(f"Successful: {stats['successful']}")
    print(f"Empty: {stats['empty']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    print(f"Total processing time: {stats['total_processing_time']:.1f}s")
    print(f"Average time per file: {stats['total_time']/(stats['successful']+stats['empty']):.2f}s")
    
    return stats


if __name__ == "__main__":
    load_dotenv()
    
    # Configuration
    dataset_root_dir = os.getenv("DATASET_ROOT_DIR")
    audio_root_dir = os.getenv("AUDIO_ROOT_DIR")
    real_audio_root_dir = os.path.join(dataset_root_dir, audio_root_dir)
    input_csv = os.getenv("PROCESSED_LABEL_FILE_NAME")
    output_csv = os.getenv("TRANSCRIBED_LABEL_FILE_NAME")
    
    # Optimized parameters
    model_name = "large-v3-turbo"  # or "openai/whisper-large-v3"
    language = "en"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64  # Larger batches for better throughput
    num_workers = 4  # Parallel workers
    
    generate_transcripts_optimized(
        input_csv=input_csv,
        output_csv=output_csv,
        audio_root_dir=real_audio_root_dir,
        model_name=model_name,
        batch_size=batch_size,
        language=language,
        raw_output=False,
        device=device,
        num_workers=num_workers
    )
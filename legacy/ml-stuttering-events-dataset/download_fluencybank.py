#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

"""
FluencyBank Audio Downloader with Authentication
For each FluencyBank episode:
* Authenticate with TalkBank
* Download the raw mp4 file
* Convert it to a 16k mono wav file
* Remove the original file
"""

import os
import pathlib
import subprocess
import requests
import numpy as np
import sys
import argparse

# FluencyBank configuration
EPISODES_FILE = "FluencyBank_episodes.csv"

def authenticate_talkbank(email, password):
    """
    Authenticate with TalkBank and return a session with valid cookies
    """
    session = requests.Session()
    
    # Set headers similar to browser request
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:141.0) Gecko/20100101 Firefox/141.0',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Origin': 'https://media.talkbank.org',
        'Referer': 'https://media.talkbank.org/',
        'DNT': '1',
        'Sec-GPC': '1',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site'
    })
    
    # First, make OPTIONS request (preflight)
    options_url = "https://sla2.talkbank.org/logInUser"
    try:
        options_response = session.options(options_url)
        print(f"OPTIONS request status: {options_response.status_code}")
    except requests.RequestException as e:
        print(f"OPTIONS request failed: {e}")
        return None
    
    # Then make the actual login POST request
    login_url = "https://sla2.talkbank.org/logInUser"
    login_data = {
        "email": email,
        "pswd": password
    }
    
    try:
        response = session.post(
            login_url,
            json=login_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success') and result.get('authStatus', {}).get('loggedIn'):
                print("Successfully authenticated with TalkBank")
                return session
            else:
                print(f"Authentication failed: {result}")
                return None
        else:
            print(f"Login request failed with status {response.status_code}")
            return None
            
    except requests.RequestException as e:
        print(f"Login request failed: {e}")
        return None

def save_cookies_to_file(session, cookie_file):
    """
    Save session cookies to a file in Netscape format for ffmpeg
    """
    try:
        with open(cookie_file, 'w') as f:
            f.write("# Netscape HTTP Cookie File\n")
            for cookie in session.cookies:
                # Format: domain, domain_specified, path, secure, expires, name, value
                domain = cookie.domain if cookie.domain else "media.talkbank.org"
                domain_specified = "TRUE" if cookie.domain_specified else "FALSE"
                path = cookie.path if cookie.path else "/"
                secure = "TRUE" if cookie.secure else "FALSE"
                expires = str(int(cookie.expires)) if cookie.expires else "0"
                
                f.write(f"{domain}\t{domain_specified}\t{path}\t{secure}\t{expires}\t{cookie.name}\t{cookie.value}\n")
        return True
    except Exception as e:
        print(f"Failed to save cookies: {e}")
        return False

def stream_with_requests_then_ffmpeg(session, url, output_path):
    """
    Alternative approach: Use requests to start the stream, then pipe to ffmpeg
    """
    try:
        # Set headers to mimic browser video request
        headers = {
            'Accept': 'video/webm,video/ogg,video/*;q=0.9,application/ogg;q=0.7,audio/*;q=0.6,*/*;q=0.5',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'identity',
            'DNT': '1',
            'Sec-GPC': '1',
            'Sec-Fetch-Dest': 'video',
            'Sec-Fetch-Mode': 'no-cors',
            'Sec-Fetch-Site': 'same-origin',
            'Referer': url,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:141.0) Gecko/20100101 Firefox/141.0'
        }
        
        # Make the request with streaming
        response = session.get(url, headers=headers, stream=True)
        
        print(f"    Response status: {response.status_code}")
        print(f"    Content-Type: {response.headers.get('content-type', 'Unknown')}")
        print(f"    Content-Length: {response.headers.get('content-length', 'Unknown')}")
        
        if response.status_code not in [200, 206]:
            print(f"    HTTP error: {response.status_code}")
            return False
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if 'video' not in content_type and 'octet-stream' not in content_type:
            print(f"    Unexpected content type: {content_type}")
            # Let's see what we actually got
            first_chunk = next(response.iter_content(chunk_size=100), b'')
            print(f"    First 100 bytes: {first_chunk}")
            return False
        
        # Use ffmpeg to process the stream from stdin
        cmd = [
            "ffmpeg", "-y",
            "-f", "mp4",  # Force input format
            "-i", "pipe:0",  # Read from stdin
            "-c", "copy",  # Copy without re-encoding
            str(output_path)
        ]
        
        print(f"    Piping stream to ffmpeg...")
        process = subprocess.Popen(cmd, 
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        
        # Stream the content to ffmpeg
        try:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    process.stdin.write(chunk)
        except Exception as e:
            print(f"    Error streaming to ffmpeg: {e}")
        finally:
            process.stdin.close()
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            return True
        else:
            stderr_text = stderr.decode('utf-8')
            print(f"    FFmpeg pipe processing failed:")
            
            # Extract error lines
            error_lines = []
            for line in stderr_text.split('\n'):
                if any(keyword in line.lower() for keyword in ['error', 'invalid', 'not found', 'failed']):
                    error_lines.append(line.strip())
            
            if error_lines:
                for error_line in error_lines[-2:]:
                    print(f"      {error_line}")
            
            return False
            
    except Exception as e:
        print(f"    Alternative streaming failed: {e}")
        return False
    """
    Stream video using ffmpeg with session cookies
    """
    cookie_file = "talkbank_cookies.txt"
    
    # Save session cookies to file
    if not save_cookies_to_file(session, cookie_file):
        return False
    
    try:
        # Use ffmpeg to stream and save the video
        cmd = [
            "ffmpeg", "-y",
            "-cookies", cookie_file,
            "-user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:141.0) Gecko/20100101 Firefox/141.0",
            "-headers", "Accept: video/webm,video/ogg,video/*;q=0.9,application/ogg;q=0.7,audio/*;q=0.6,*/*;q=0.5",
            "-headers", f"Referer: {url}",
            "-headers", "DNT: 1",
            "-headers", "Sec-GPC: 1",
            "-headers", "Sec-Fetch-Dest: video",
            "-headers", "Sec-Fetch-Mode: no-cors",
            "-headers", "Sec-Fetch-Site: same-origin",
            "-i", url,
            "-c", "copy",  # Copy without re-encoding
            str(output_path)
        ]
        
        print(f"  Streaming with ffmpeg...")
        process = subprocess.Popen(cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Clean up cookie file
        try:
            os.remove(cookie_file)
        except:
            pass
        
        if process.returncode == 0:
            return True
        else:
            stderr_text = stderr.decode('utf-8')
            print(f"  FFmpeg streaming failed:")
            
            # Extract error lines
            error_lines = []
            for line in stderr_text.split('\n'):
                if any(keyword in line.lower() for keyword in ['error', 'invalid', 'not found', 'failed', 'forbidden', 'unauthorized']):
                    error_lines.append(line.strip())
            
            if error_lines:
                for error_line in error_lines[-3:]:
                    print(f"    {error_line}")
            
            return False
            
    except Exception as e:
        print(f"Stream capture failed: {e}")
        return False

def main():
    """
    Main function to download and process FluencyBank audio files
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download and convert FluencyBank audio files with authentication')
    parser.add_argument('--wavs', type=str, default="wavs",
                       help='Directory where audio files will be saved (default: wavs)')
    parser.add_argument('--email', type=str, required=True,
                       help='TalkBank account email address')
    parser.add_argument('--pass', type=str, required=True, dest='password',
                       help='TalkBank account password')
    
    args = parser.parse_args()
    
    print("FluencyBank Audio Downloader")
    print("=" * 40)
    
    # Check if episodes file exists
    if not os.path.exists(EPISODES_FILE):
        print(f"Error: Episodes file '{EPISODES_FILE}' not found!")
        print("Please ensure FluencyBank_episodes.csv is in the current directory.")
        sys.exit(1)
    
    # Authenticate with TalkBank
    print("Authenticating with TalkBank...")
    session = authenticate_talkbank(args.email, args.password)
    if not session:
        print("Authentication failed. Please check your credentials.")
        sys.exit(1)
    
    # Load FluencyBank episode data
    print(f"Loading episodes from {EPISODES_FILE}")
    table = np.loadtxt(EPISODES_FILE, dtype=str, delimiter=",")
    urls = table[:,2]
    n_items = len(urls)
    print(f"Found {n_items} episodes to process")
    
    # Process each episode
    for i in range(n_items):
        # Get show/episode IDs
        show_abrev = table[i,-2].strip()
        ep_idx = table[i,-1].strip()
        episode_url = table[i,2].strip()
        
        # FluencyBank uses mp4 files
        ext = '.mp4'
        
        # Ensure the base folder exists for this episode
        episode_dir = pathlib.Path(f"{args.wavs}/{show_abrev}/")
        os.makedirs(episode_dir, exist_ok=True)
        
        # Get file paths
        audio_path_orig = pathlib.Path(f"{episode_dir}/{ep_idx}{ext}")
        wav_path = pathlib.Path(f"{episode_dir}/{ep_idx}.wav")
        
        # Check if this file has already been processed
        if os.path.exists(wav_path):
            print(f"Skipping {show_abrev}/{ep_idx} - already processed")
            continue
        
        print(f"Processing {show_abrev}/{ep_idx} ({i+1}/{n_items})")
        
        # Stream raw video file using ffmpeg with session cookies
        if not os.path.exists(audio_path_orig):
            print(f"  Streaming from {episode_url}")
            success = stream_with_ffmpeg(session, episode_url, audio_path_orig)
            if not success:
                print(f"  Failed to stream, skipping...")
                continue
        else:
            print(f"  Using existing file {audio_path_orig}")
        
        # Check if streamed file is valid
        file_size = os.path.getsize(audio_path_orig)
        if file_size < 1000:  # Less than 1KB is suspicious
            print(f"  Streamed file is too small ({file_size} bytes), likely failed")
            
            # Let's see what we actually got
            with open(audio_path_orig, 'rb') as f:
                content = f.read()
                print(f"  Content (hex): {content.hex()}")
                print(f"  Content (text): {content}")
            
            os.remove(audio_path_orig)
            
            # Try alternative approach: use requests to get stream URL then ffmpeg
            print(f"  Trying alternative streaming approach...")
            success = stream_with_requests_then_ffmpeg(session, episode_url, audio_path_orig)
            if not success:
                print(f"  Alternative approach also failed, skipping...")
                continue
            
            # Check the alternative result
            file_size = os.path.getsize(audio_path_orig)
            if file_size < 1000:
                print(f"  Alternative approach also produced small file ({file_size} bytes), skipping...")
                os.remove(audio_path_orig)
                continue
        
        print(f"  Successfully streamed {file_size:,} bytes")
        
        # Convert to 16khz mono wav file
        print(f"  Converting to WAV format...")
        cmd = ["ffmpeg", "-y", "-i", str(audio_path_orig), "-ac", "1", "-ar", "16000", str(wav_path)]
        process = subprocess.Popen(cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Check if conversion was successful
        if process.returncode != 0:
            print(f"  FFmpeg conversion failed for {audio_path_orig}")
            stderr_text = stderr.decode('utf-8')
            
            # Extract only the actual error lines (skip version info)
            error_lines = []
            for line in stderr_text.split('\n'):
                # Look for actual error messages
                if any(keyword in line.lower() for keyword in ['error', 'invalid', 'not found', 'failed']):
                    error_lines.append(line.strip())
            
            if error_lines:
                print(f"  Actual errors:")
                for error_line in error_lines[-3:]:  # Show last 3 error lines
                    print(f"    {error_line}")
            else:
                # If no specific errors found, show the last few lines
                lines = [line.strip() for line in stderr_text.split('\n') if line.strip()]
                print(f"  Last output lines:")
                for line in lines[-3:]:
                    print(f"    {line}")
            
            # Check if it's the "moov atom not found" error
            if 'moov atom not found' in stderr_text:
                print(f"  File appears to be corrupted or incomplete")
            
            continue
        
        # Remove the original mp4 file only if conversion succeeded
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            os.remove(audio_path_orig)
            print(f"  Successfully converted and cleaned up")
        else:
            print(f"  Warning: WAV file not created properly")
    
    print("\nFluencyBank download and conversion complete!")

def stream_with_ffmpeg(session, url, output_path):
    """
    Stream video using ffmpeg with session cookies
    """
    cookie_file = "talkbank_cookies.txt"
    
    # Save session cookies to file
    if not save_cookies_to_file(session, cookie_file):
        return False
    
    try:
        # Use ffmpeg to stream and save the video
        cmd = [
            "ffmpeg", "-y",
            "-cookies", cookie_file,
            "-user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:141.0) Gecko/20100101 Firefox/141.0",
            "-headers", "Accept: video/webm,video/ogg,video/*;q=0.9,application/ogg;q=0.7,audio/*;q=0.6,*/*;q=0.5",
            "-headers", f"Referer: {url}",
            "-headers", "DNT: 1",
            "-headers", "Sec-GPC: 1",
            "-headers", "Sec-Fetch-Dest: video",
            "-headers", "Sec-Fetch-Mode: no-cors",
            "-headers", "Sec-Fetch-Site: same-origin",
            "-i", url,
            "-c", "copy",  # Copy without re-encoding
            str(output_path)
        ]
        
        print(f"  Streaming with ffmpeg...")
        process = subprocess.Popen(cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Clean up cookie file
        try:
            os.remove(cookie_file)
        except:
            pass
        
        if process.returncode == 0:
            return True
        else:
            stderr_text = stderr.decode('utf-8')
            print(f"  FFmpeg streaming failed:")
            
            # Extract error lines
            error_lines = []
            for line in stderr_text.split('\n'):
                if any(keyword in line.lower() for keyword in ['error', 'invalid', 'not found', 'failed', 'forbidden', 'unauthorized']):
                    error_lines.append(line.strip())
            
            if error_lines:
                for error_line in error_lines[-3:]:
                    print(f"    {error_line}")
            
            return False
            
    except Exception as e:
        print(f"Stream capture failed: {e}")
        return False

if __name__ == "__main__":
    main()
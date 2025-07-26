# Automatic Dysfluency Detector and Corrector for Stuttered Speech

## Instructions

### Pre-training

#### Acquire Dataset

Apply patch:

```bash
python dataset_patches.py
```

Download raw audio files:

```bash
cd ml-stuttering-events-dataset
python download_audio.py --episodes SEP-28k_episodes_patched.csv --wavs [WAV_DIR]
```

Extract clips:

```bash
cd ml-stuttering-events-dataset
python extract_clips.py --labels SEP-28k_labels.csv --wavs [DATA_DIR] --clips [CLIP_DIR]
```

#### Pre-processing

Data Prep:

```bash
python data-prep.py
```

Transcribe clips:

```bash
python transcribe.py
```

#### Training

```bash
python detector.py
```
# Automatic Dysfluency Detector ~~and Corrector~~ for Stuttered Speech

## Instructions

#### Pre-processing

```bash
python data-prep.py
```

This script will download 2 datasets: SEP-28k for Training, and UCLASS for Testing.

#### Training

```bash
python train.py --csv_file all_labels.csv --epochs 20 [--batch_size 32] [--lr 1e-4]
```

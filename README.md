# Scale OCR - Video Weight Tool

A machine learning system for automatically reading weight values from seven-segment scale displays in video footage. Uses a CNN+LSTM+CTC architecture trained on labeled frame data.

## Overview

This project provides:
- **Model Training**: Train a CRNN (CNN + Bidirectional LSTM) model with CTC loss to recognize weight values like "7.535" from scale display images
- **Video Inference**: Process videos to extract frame-by-frame weight readings with temporal smoothing and confidence scoring
- **Labeling Tools**: GUI applications to efficiently label training data from video frames
- **Data Pipeline**: Scripts to extract frames, split datasets, and validate models

## Requirements

- Python 3.8–3.12 (recommended: 3.11 or 3.12)
- CUDA-capable GPU (recommended for training)
- ffmpeg (system installation required)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/harrly14/VideoWeightTool
cd VideoWeightTool
```

2. Create a virtual environment:
```bash
# Linux/macOS
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install ffmpeg:
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: `choco install ffmpeg` or download from https://ffmpeg.org/

## Project Structure

```
VideoWeightTool/
├── train.py                 # Model training script
├── model.py                 # CNN+LSTM+CTC architecture
├── dataset.py               # Dataset class and data loading
├── process_video.py         # Video inference script
├── requirements.txt
│
├── data/
│   ├── images/              # Training images (created by extract_frames.py)
│   ├── labels/              # Train/val/test CSVs (created by split_data.py)
│   └── all_data.csv         # Master labels file
│
├── models/                  # Saved model checkpoints (created during training)
│
├── training_scripts/
│   ├── extract_frames.py    # Extract and crop frames from videos
│   ├── split_data.py        # Split labels into train/val/test
│   └── validate.py          # Evaluate model on test set
│
├── labelling_workflow/      # GUI tool for batch frame labeling
│   ├── main.py
│   └── USAGE_GUIDE.md
│
├── manual_workflow/         # GUI tool for video editing and manual labeling
│   └── main.py
│
└── mini_scripts/            # Utility scripts (ad-hoc, not polished)
```

## Workflow

### 1. Label Training Data

Use the labeling tool to create ground-truth labels:

```bash
cd labelling_workflow
python main.py
```

See `labelling_workflow/USAGE_GUIDE.md` for detailed instructions.

### 2. Extract Frames

Extract cropped frames from labeled videos:

```bash
python training_scripts/extract_frames.py --video_dir /path/to/videos
```

This creates images in `data/images/`.

### 3. Split Dataset

Split labels into train/validation/test sets:

```bash
python training_scripts/split_data.py
```

This creates CSV files in `data/labels/`.

### 4. Train Model

Train the Scale OCR model:

```bash
python train.py
```

You can also configure training via command-line arguments:

```bash
python train.py --batch-size 16 --epochs 100 --data-dir /path/to/data --save-dir /path/to/save/models
```

Key arguments:
- `--batch-size`: Default 32 (reduce if memory limited)
- `--epochs`: Default 400
- `--data-dir`: Directory containing `images/` and `labels/` (default: `data`)
- `--save-dir`: Directory to save checkpoints (default: `models`)
- `--no-amp`: Disable mixed precision training (useful for debugging)

Models are saved to `models/` (or your specified `--save-dir`):
- `best_model.pth` - Lowest validation loss
- `best_accuracy_model.pth` - Highest sequence accuracy
- `latest_model.pth` - Most recent checkpoint

### 5. Run Inference

Process a video to extract weight readings:

```bash
python training_scripts/process_video.py --video path/to/video.mp4 --output weights.csv
```

Options:
```bash
# Specify ROI (region of interest)
python process_video.py --video video.mp4 --roi 100,50,400,150

# Use conservative mode (flags uncertain predictions)
python process_video.py --video video.mp4 --conservative

# Compare against ground-truth
python process_video.py --video video.mp4 --ground-truth gt.csv

# Save annotated video
python process_video.py --video video.mp4 --save-video
```

## Model Architecture

The model uses a CRNN (Convolutional Recurrent Neural Network) architecture:

1. **CNN Backbone**: 6 convolutional layers extract visual features from 64×256 input images
2. **Bidirectional LSTM**: 2-layer LSTM reads features as a sequence
3. **CTC Decoder**: Connectionist Temporal Classification decodes the output to weight strings

Output format: `"X.XXX"` (e.g., "7.535", "12.450")

## Labeling Tools

### Labelling Workflow (Recommended)
Batch labeling tool with smart frame sampling, zoom/pan, and keyboard shortcuts. Use if you want to label frames for use in training a model.
```bash
cd labelling_workflow && python main.py
```

### Manual Workflow
Video editing tool with crop, trim, brightness/contrast adjustments. Use if you want to manually label frames and save edits made to videos.
```bash
cd manual_workflow && python main.py
```

## Troubleshooting

### NumPy installation fails
```bash
pip install --only-binary=numpy -r requirements.txt
```

### CUDA out of memory
Reduce `batch_size` in `train.py` (try 8 or 16).

### Python 3.13+
Some packages may not have pre-built wheels. Use Python 3.11 or 3.12 instead.

## License

[MIT](https://choosealicense.com/licenses/mit/)

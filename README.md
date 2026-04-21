# Scale OCR - Video Weight Tool

A machine learning system for reading weight values from seven-segment scale displays in video footage. The current pipeline trains a per-digit CNN classifier on extracted frame crops, then reconstructs full weight strings like `7.535` from four digit predictions.

## Overview

This project provides:
- **Model Training**: Train a digit-classification CNN that predicts each of the four scale digits independently
- **Video Inference**: Process videos to extract frame-by-frame weight readings with temporal smoothing, confidence scoring, and optional flagged-frame review signals
- **Labelling Tools**: PyQt applications for ROI setup, batch labelling, and manual video editing
- **Data Pipeline**: Scripts to extract frames, split datasets, evaluate checkpoints, and maintain datasets

## Requirements

- Python 3.8–3.12 (recommended: 3.11 or 3.12)
- CUDA-capable GPU (recommended for training)
- ffmpeg (system installation required for the manual video editing workflow)

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
├── video_weight_tool.py     # Master terminal menu entry point
├── requirements.txt
│
├── core/                    # Shared Library Code
│   ├── model.py             # Digit CNN classifier
│   ├── dataset.py           # Digit dataset + transforms
│   └── config.py            # Unified Configuration
│
├── pipelines/               # Automated Processing Scripts
│   ├── train.py             # Model training script
│   ├── inference.py         # Run inference on video
│   ├── evaluate.py          # Validation/test evaluation
│   └── preparation/         # Extract frames, Split data
│
├── workflows/               # GUI Applications (Human-in-the-Loop)
│   ├── labelling/            # Main labelling tool
│   └── manual/              # Manual crop/edit tool
│
├── data/                    # Data Storage
│   ├── all_data.csv         # Raw frame-level labels from labelling workflow
│   ├── metadata.json        # ROI sections and divider metadata
│   ├── images/              # Extracted training images
│   ├── labels/              # Train/val/test split CSVs
│   ├── raw_videos/          # Place your MP4s here
│   └── models/              # Saved Checkpoints
```

## Workflow

**Recommendation:** Run `python video_weight_tool.py` for an interactive menu that launches the main workflows.

### 1. Label Training Data

Launch the main labelling tool via the menu (`Labelling & Data Entry`) or directly:

```bash
python workflows/labelling/main.py
```

This updates:
- `data/all_data.csv`
- `data/metadata.json`

### 2. Prepare Dataset

After labelling, extract frames and then split the dataset.

**Via menu:** `Data Processing Pipelines` -> `Extract Frames from Videos`, then `Split Dataset`

**Via CLI:**
```bash
python pipelines/preparation/extract.py
python pipelines/preparation/split.py
```

This produces:
- `data/images/`
- `data/labels/train_labels.csv`
- `data/labels/val_labels.csv`
- `data/labels/test_labels.csv`

### 3. Train Model

Train the digit classifier:

**Via menu:** `Training & Inference` -> `Train New Model`

**Via CLI:**
```bash
python pipelines/train.py
```

**Monitoring:**
You can monitor training progress (Loss, Accuracy, Learning Rate) in real-time using TensorBoard:
```bash
tensorboard --logdir runs
```
Open the URL (http://localhost:6006) in your browser.

Key outputs:
- `data/models/latest_model.pth`
- `data/models/best_model.pth`
- `data/models/best_accuracy_model.pth`
- `runs/` TensorBoard logs

Arguments match standard PyTorch patterns. See:
```bash
python pipelines/train.py --help
```

You can resume from a checkpoint:
```bash
python pipelines/train.py --resume data/models/latest_model.pth
```

You can disable AMP if needed:
```bash
python pipelines/train.py --no-amp
```

### 4. Run Inference

Process new videos to extract weight readings:

**Via menu:** `Training & Inference` -> `Run Inference on Video`

**Via CLI:**
```bash
python pipelines/inference.py --video path/to/video.mp4 --output weights.csv
```


Options:
```bash
# Manually specify ROI + digit dividers
# ROI: 8 comma-separated quad points: x1,y1,x2,y2,x3,y3,x4,y4
# Dividers: 4 comma-separated x-coordinates in warped canvas space
python pipelines/inference.py --video video.mp4 --roi 100,50,400,50,400,150,100,150 --dividers 38,77,115,154

# Use conservative mode (flag uncertain predictions more aggressively)
python pipelines/inference.py --video video.mp4 --conservative

# Compare against ground-truth
python pipelines/inference.py --video video.mp4 --ground-truth gt.csv

# Save annotated video
python pipelines/inference.py --video video.mp4 --save-video
```

If `--roi` is omitted, inference attempts to load ROI sections from `data/metadata.json` for the input video.

If `--roi` is provided, also provide `--dividers` so the ROI can be split into the four digit regions used by the model.

### 5. Evaluate a Checkpoint

Run evaluation on the validation split:

```bash
python pipelines/evaluate.py
```

Run the final test audit:

```bash
python pipelines/evaluate.py --test
```

## Model Architecture

The current model is a digit classifier, not a CRNN/CTC sequence model.

1. **ROI warping**: ROI metadata defines a quadrilateral section of the scale display per video segment.
2. **Digit slicing**: Each ROI is split into four digit crops using divider coordinates.
3. **CNN Backbone**: A compact convolutional network processes each grayscale digit crop.
4. **Classifier head**: A linear head predicts one of 10 digit classes for each crop.
5. **Weight reconstruction**: Four digit predictions are assembled into the final `X.XXX` string.

Output format: `X.XXX` (for example `7.535`)

## Labelling Tools

### Labelling Workflow (Recommended)
Batch labelling tool with smart frame sampling, zoom/pan, ROI setup, and keyboard shortcuts. Use this to create `data/all_data.csv` and `data/metadata.json`.

```bash
cd workflows/labelling && python main.py
```

### Manual Workflow
Video editing tool with crop, trim, brightness/contrast adjustments, temporal frame averaging preview (pixel-level, odd windows), and ffmpeg/OpenCV export.

Temporal averaging notes:
- Preview uses centered temporal averaging across neighboring frames, clipped to active trim bounds.
- Export uses ffmpeg `tmix` with clone-padding and trim alignment for non-warp exports.
- Warp-enabled exports use the OpenCV rendering path and also apply temporal averaging when enabled.

```bash
cd workflows/manual && python main.py
```

## Maintenance Scripts

Additional utilities live under `pipelines/maintenance/`:
- `cleanup.py`: Remove CSV entries that reference missing files
- `stats.py`: Inspect dataset/image statistics
- `create_digit_heatmap.py`: Generate label distribution heatmaps

## Troubleshooting

### NumPy installation fails
```bash
pip install --only-binary=numpy -r requirements.txt
```

### CUDA out of memory
Reduce `--batch-size` in `pipelines/train.py` (try 8 or 16).

### Python 3.13+
Some packages may not have pre-built wheels. Use Python 3.11 or 3.12 instead.

## Notes

- `ffmpeg-python` is a Python wrapper, but the `ffmpeg` binary must also be installed and available on `PATH` for the manual editing workflow.
- The GUI workflows require a desktop environment because they use PyQt5.

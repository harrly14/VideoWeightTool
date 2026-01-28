# Scale OCR - Video Weight Tool

A machine learning system for automatically reading weight values from seven-segment scale displays in video footage. Uses a CNN+LSTM+CTC architecture trained on labeled frame data.

## Overview

This project provides:
- **Model Training**: Train a CRNN (CNN + Bidirectional LSTM) model with CTC loss to recognize weight values like "7.535" from scale display images
- **Video Inference**: Process videos to extract frame-by-frame weight readings with temporal smoothing and confidence scoring
- **labelling Tools**: GUI applications to efficiently label training data from video frames
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
├── video_weight_tool.py     # Master TUI Entry Point 
├── requirements.txt
│
├── core/                    # Shared Library Code
│   ├── model.py             # Neural Network Architecture
│   ├── dataset.py           # Data Loading Logic
│   └── config.py            # Unified Configuration
│
├── pipelines/               # Automated Processing Scripts
│   ├── train.py             # Model training script
│   ├── inference.py         # Run model on meaningful video
│   └── preparation/         # Extract frames, Split data
│
├── workflows/               # GUI Applications (Human-in-the-Loop)
│   ├── labelling/            # Main labelling tool
│   └── manual/              # Manual crop/edit tool
│
├── data/                    # Data Storage
│   ├── images/              # Extracted training images
│   ├── labels/              # CSV labels
│   ├── raw_videos/          # Place your MP4s here
│   └── models/              # Saved Checkpoints
```

## Workflow

**Recommendation:** Run `./video_weight_tool.py` for an interactive menu guiding you through all steps.

### 1. Label Training Data

Launch the key labelling tool via the TUI (Option 1) or directly:

```bash
python workflows/labelling/main.py
```

This updates:
- `data/labels/all_data.csv`
- `data/valid_video_sections.json`

### 2. Prepare Dataset

After labelling, you must extract frames and split the dataset.
**Via TUI:** Option 2 -> 1 (Extract), then Option 2 -> 2 (Split).

**Via CLI:**
```bash
python pipelines/preparation/extract.py
python pipelines/preparation/split.py
```

### 3. Train Model

Train the Scale OCR model:
**Via TUI:** Option 3 -> 1 (Extract).

**Via CLI:**
```bash
python train.py
```

**Monitoring:**
You can monitor training progress (Loss, Accuracy, Learning Rate) in real-time using TensorBoard:
```bash
tensorboard --logdir runs
```
Open the URL (http://localhost:6006) in your browser.

Arguments match standard PyTorch patterns (see `python train.py --help`).

### 4. Run Inference

Process new videos to extract weight readings:

```bash
# Via TUI: Option 3, then 2
# Via CLI:
python pipelines/inference.py --video path/to/video.mp4 --output weights.csv
```


Options:
```bash
# Manually specify ROI (8 comma-separated quad points: x1,y1,x2,y2,x3,y3,x4,y4)
python pipelines/inference.py --video video.mp4 --roi 100,50,400,50,400,150,100,150

# Use conservative mode (flags uncertain predictions)
python pipelines/inference.py --video video.mp4 --conservative

# Compare against ground-truth
python pipelines/inference.py --video video.mp4 --ground-truth gt.csv

# Save annotated video
python pipelines/inference.py --video video.mp4 --save-video
```

## Model Architecture

The model uses a CRNN (Convolutional Recurrent Neural Network) architecture:

1. **CNN Backbone**: 6 convolutional layers extract visual features from 64×256 input images
2. **Bidirectional LSTM**: 2-layer LSTM reads features as a sequence
3. **CTC Decoder**: Connectionist Temporal Classification decodes the output to weight strings

Output format: `"X.XXX"` (e.g., "7.535", "12.450")

## labelling Tools

### Labelling Workflow (Recommended)
Batch labelling tool with smart frame sampling, zoom/pan, and keyboard shortcuts. Use if you want to label frames for use in training a model.
```bash
cd workflows/labelling && python main.py
```

### Manual Workflow
Video editing tool with crop, trim, brightness/contrast adjustments. Use if you want to manually label frames and save edits made to videos.
```bash
cd workflows/manual && python main.py
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

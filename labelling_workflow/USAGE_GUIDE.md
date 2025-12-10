# Quick Start Guide - Video Labelling Tool

## Installation

Ensure dependencies are installed:
```bash
pip install PyQt5 opencv-python numpy
```

## Running the Tool

### Option 1: Startup Dialog (Recommended)
```bash
cd labelling_workflow
python main.py
```

This will open a configuration dialog where you can:
- Set number of labels per video (n)
- Select the video folder to process
- Select or create the output CSV file

### Option 2: Command Line Arguments
```bash
python main.py --n 50 --video-dir /path/to/videos --csv /path/to/output.csv
```

This skips the dialog and goes straight to processing.

## Workflow

### 1. Video Selection
The tool auto-discovers all videos (.mp4, .mov, .avi, .mkv) in your selected folder. It will skip any videos already in your output CSV. Therefore, it is recommended that you prepare a local folder with all the .mp4 video files you want to process. Avoid using network share folders as the latency can cause problems with OpenCV.

### 2. Valid Frame Range (First Step)
Before labelling, select the valid frame range:
- **Start frame:** Drag the "Start frame" slider or use the spinbox
- **End frame:** Drag the "End frame" slider or use the spinbox
- **Default:** Entire video

The frame display updates in real-time. Choose a range that excludes "garbage" frames (e.g., artifacts, low quality, etc.).

### 3. Start Labelling
Click the **"Start Labelling"** button. The tool will:
- Calculate target frames using smart sampling
- Lock the valid range (can't change it)
- Show how many frames you'll label
- Display a warning if fewer frames available than target

### 4. Label Target Frames
For each target frame:
- **View:** Frame displays with adjustment sliders (brightness, saturation, contrast)
- **Enter weight:** Type a weight value (0-1 decimal or integer)
- **Submit:** Press Enter, Space, or click Submit
- **Navigate:** Use Left/Right arrows or Previous/Next buttons
- **Adjust view:** Use mouse wheel to zoom, click+drag to pan when zoomed

The tool **auto-advances** to the next frame after submission.

### 5. Summary & Continue
After labelling all target frames (or clicking Save and Close):
- Summary dialog shows how many frames were labelled
- Options: Continue to next video, or go back to review/edit

Results are saved to your CSV immediately.

## Display Adjustment Sliders

These sliders help you see the video better. They DON'T affect saved weights—only how you view the frame:

| Slider | Range | Default | Purpose |
|--------|-------|---------|---------|
| Brightness | -255 to 255 | 0 | Lighten/darken frame |
| Saturation | 0 to 300 | 100 | Reduce/boost colors |
| Contrast | 0 to 200 | 100 | Increase/decrease contrast |

**Tip:** Use these if the video is hard to see (too dark, washed out, etc.). Click "Reset" to go back to normal.

## Frame Navigation

### Labelling (Between Target Frames)
- **Previous/Next buttons:** Jump to previous/next target frame
- **Left/Right arrow keys:** Same as buttons
- **Scrub slider:** Skip directly to any frame for context

### Free Navigation (Any Frame)
- **Scrub slider:** Drag to view any frame
- **Spinbox:** Type frame number directly

**Note:** You can navigate freely, but you only *label* the target frames calculated by the tool.

## Zoom & Pan

To see fine details in a frame:
1. **Scroll wheel up:** Zoom in (up to 10x)
2. **Click and drag:** Pan across the zoomed frame
3. **Scroll wheel down:** Zoom out

The tool remembers your zoom level as you navigate frames.

## Output CSV Format

Your results are saved in a CSV with three columns:

```csv
filename,frame_number,weight
video1.mp4,0,0.85
video1.mp4,20,0.92
video1.mp4,40,0.78
video2.mp4,15,0.65
```

- **filename:** Video name (how it appears in folder)
- **frame_number:** Frame index (0-based)
- **weight:** Your label value

Only **non-zero weights** are written. Zero weights are treated as "not labelled".

## Sampling Strategy

The tool calculates which frames to label intelligently:

**Formula:** `interval = floor(valid_frames / n)`

**Example:**
- Valid range: frames 0-499 (500 frames)
- Target labels: n=50
- Interval: 500/50 = 10
- Result: Label frames [0, 10, 20, 30, ..., 490]

This ensures labels are spread evenly across the entire video.

**Short Videos:** If a video has fewer frames than your target, you'll label all of them (with a warning).

## Resume From Backup

If the program crashes or you close it:
- Your current progress is **auto-saved** to a backup file
- The CSV is **always saved** after each weight entry
- Next run: Videos already in CSV are skipped (you don't relabel them)

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Left Arrow | Previous target frame |
| Right Arrow | Next target frame |
| Ctrl+Left | Jump back 10 frames (free nav) |
| Ctrl+Right | Jump forward 10 frames (free nav) |
| W | Focus weight entry field |
| Enter | Submit weight |
| Space | Submit weight (when in weight field) |

## Troubleshooting

**"No videos found in [folder]"**
- Make sure folder contains `.mp4`, `.mov`, `.avi`, or `.mkv` files
- Check file extensions are lowercase (case-sensitive on some systems)

**"Frame took too long to read"**
- If video is on network drive, copy to local disk first
- File might be corrupted

**"All videos already processed"**
- All videos in that folder are in your CSV
- To relabel a video, remove its rows from CSV first

**"Fewer labels than target"**
- Valid frame range is shorter than n
- You'll label all available frames (warning shown)

**Weights not saving**
- Make sure you click Submit or press Enter after typing
- Check CSV file has write permission
- Look for `.backup_*.csv` files if you think data was lost

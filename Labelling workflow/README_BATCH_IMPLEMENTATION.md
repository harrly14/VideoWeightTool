# Labelling Workflow - Batch Processing Implementation

## Overview

The labelling workflow has been completely refactored to support batch processing of multiple videos with intelligent frame sampling. The tool is now labelling-only (no video editing/export), with simplified UI and dynamic touchstone sampling.

## Key Changes

### 1. **Removed Features**
- ❌ Video editing (trim, crop, ffmpeg operations)
- ❌ Video export functionality
- ❌ `CropLabel` with drag-to-crop interaction
- ❌ Trim slider for video processing

### 2. **New Features**
- ✅ Batch processing of multiple videos
- ✅ Dynamic touchstone sampling (intelligent frame distribution)
- ✅ Valid frame range selection (required before labelling)
- ✅ Zoom and pan for frame viewing (no crop)
- ✅ Startup configuration dialog with CLI arg override
- ✅ Auto-discovery of video files in directory
- ✅ Resume from existing CSV (skip already-processed videos)
- ✅ Per-video completion summaries with continue/review options
- ✅ Unified CSV output: `filename, frame_number, weight`
- ✅ Auto-save backups during labelling

### 3. **New/Modified Files**

#### **New Files Created**

1. **`ZoomPanLabel.py`** - Lightweight label widget with zoom/pan
   - Mouse wheel zoom (1x to 10x)
   - Mouse drag pan when zoomed
   - No crop functionality (pure viewing aid)

2. **`FrameSampler.py`** - Frame selection logic
   - Calculates intelligent frame distribution
   - Formula: `m = floor((valid_end - valid_start) / n)`
   - Labels every `m`-th frame within valid range, up to `n` labels
   - Warns if video too short (fewer labels available)

3. **`StartupDialog.py`** - Configuration UI
   - Input for `n` (labels per video)
   - Folder picker for video directory
   - File picker for output CSV (create or existing)
   - Explains sampling strategy

4. **`BatchManager.py`** - Batch orchestration
   - Auto-discovers video files (.mp4, .mov, .avi, .mkv)
   - Tracks processed videos (skips duplicates)
   - Launches labelling window per video
   - Handles per-video summaries
   - Aggregates results into unified CSV
   - Auto-saves partial results on interruption

#### **Modified Files**

1. **`VideoParams.py`**
   - Removed: `trim_start`, `trim_end`, `crop_coords`
   - Kept: `brightness`, `saturation`, `contrast` sliders only
   - Simplified validation

2. **`Window.py`** - Completely refactored
   - Removed: Trim UI, crop UI, all crop-related methods
   - Replaced: `CropImageLabel` → `ZoomPanLabel`
   - Added: Valid range sliders (locked during labelling)
   - Added: "Start Labelling" button (calculates target frames)
   - Added: Progress display (X of Y target frames)
   - Added: Frame navigation spinbox and slider
   - Modified: Navigation only between target frames (during labelling)
   - Modified: Auto-advance on weight entry
   - Modified: Summary after completing all target frames

3. **`main.py`** - Complete rewrite
   - Old: Single-video workflow with editing
   - New: Batch processing orchestration
   - Supports CLI args: `--n`, `--video-dir`, `--csv`
   - Falls back to startup dialog if no CLI args
   - Delegates to `BatchManager` for batch logic

#### **Archived Files**

- `Window_old.py` - Original implementation
- `CropLabel.py` - No longer needed (→ archived)

## Workflow

### Startup

```bash
# Option 1: Use startup dialog (default)
python main.py

# Option 2: Use CLI args (skip dialogs)
python main.py --n 50 --video-dir /path/to/videos --csv /path/to/output.csv
```

### Per-Video Flow

1. **Valid Range Selection** (required first step)
   - User adjusts `Start frame` and `End frame` sliders
   - Default: entire video (0 to total_frames-1)
   - Sliders update frame display in real-time
   - Can also use text spinboxes or jump with prev/next buttons

2. **Start Labelling**
   - User clicks "Start Labelling"
   - System calculates target frames using `FrameSampler`
   - Valid range sliders lock
   - Warning shown if fewer labels than target (short video)

3. **Label Target Frames**
   - Navigate between target frames (prev/next or scrub slider)
   - Enter weight for each frame
   - Submit with Enter/Space or click Submit button
   - Auto-advances to next target frame
   - Progress display: "X of Y target frames"
   - Free navigation with scrub slider (for context)

4. **Summary & Continue**
   - After all target frames labelled or manual save:
     - Summary dialog shows count
     - Options: Continue to next video, or Review/edit current video
   - Results saved to unified CSV (append if exists)

### Output CSV

**Format:** `filename, frame_number, weight` (one row per labelled frame)

```csv
filename,frame_number,weight
video1.mp4,0,0.85
video1.mp4,20,0.92
video1.mp4,40,0.78
video2.mp4,15,0.65
...
```

**Notes:**
- Only non-zero weights are written
- New CSV includes header; existing CSV appends without header
- Auto-saved during labelling in backup files (`.backup_filename.csv`)

### Sampling Strategy

**Dynamic Touchstone Sampling:**

Given:
- Total frames in valid range: `V = valid_end - valid_start + 1`
- Target labels: `n`

Calculate:
- Interval: `m = floor(V / n)`
- Target frames: `[valid_start, valid_start+m, valid_start+2m, ..., valid_end]`
- Stops at `n` labels or `valid_end`, whichever comes first

**Example:**
- Valid range: frames 0-499 (500 frames)
- Target: n=50 labels
- Interval: m = floor(500/50) = 10
- Target frames: [0, 10, 20, 30, ..., 490] → exactly 50 frames

**Short Video Handling:**
- If `m < 1` (fewer frames than target): warn user, label all available frames
- E.g., 15-frame video with n=50 → label all 15 frames + warning

## Configuration Options

### Startup Dialog

- **Labels per video (n):** Number of frames to label per video
  - Default: 50
  - Frames distributed evenly throughout valid range
  
- **Video folder:** Directory containing video files to process
  - Auto-discovers `.mp4, .mov, .avi, .mkv` files (case-insensitive)
  
- **Output CSV:** Path to unified output CSV
  - Create new: specify desired path
  - Append to existing: select existing CSV file

### CLI Arguments

```bash
--n N              # Labels per video (required for CLI mode)
--video-dir PATH   # Video directory (required for CLI mode)
--csv PATH         # Output CSV path (required for CLI mode)
```

All three must be provided to skip startup dialog.

## Display Adjustments

Three sliders available during labelling (don't affect saved output):

- **Brightness:** -255 to 255 (default 0)
- **Saturation:** 0 to 300 (default 100 = normal)
- **Contrast:** 0 to 200 (default 100 = normal)

Each slider has:
- Visual feedback (current value displayed)
- Reset button (restores default)
- Real-time preview on frame display

## Zoom & Pan

When zoomed > 1x:
- **Mouse wheel:** Zoom in/out (tracks cursor position)
- **Left click + drag:** Pan across frame
- **Cursor:** Changes to indicate pan-able area when zoomed

Note: Zoom is for preview only; doesn't affect labelled weights.

## Resume & Re-labelling

**Processed Video Tracking:**
- Checked against output CSV by filename
- Videos with existing entries are skipped

**Re-labelling:**
- Not allowed (no overwrite mode)
- Prevents data loss
- If needed: manually edit/remove from CSV first

**Interruption Recovery:**
- Partial results auto-saved to backup file
- On next run with same CSV: skips completed videos
- Previously labelled frames restored from CSV

## Edge Cases

1. **Very Short Video (fewer frames than target n)**
   - Warning shown during start labelling
   - All frames in valid range are labelled
   - E.g., 10-frame video with n=50 → label all 10

2. **Video Already Processed**
   - Filename checked against CSV
   - Automatically skipped
   - Prevents duplicate labelling

3. **Network File Latency**
   - Timeout warning if frame read takes > 5 seconds
   - Recommends local copy
   - Exits on timeout (data safe in CSV backup)

4. **No Videos in Directory**
   - Error shown during queue discovery
   - User redirected to select valid directory

## Testing

### Test FrameSampler Logic

```bash
python3 << 'EOF'
from FrameSampler import FrameSampler

# 100 frames, label 10
sampler = FrameSampler(100, 0, 99, 10)
frames, m, warning = sampler.get_target_frames()
# Result: [0, 10, 20, ..., 90] with m=10, no warning

# 15 frames, label 50
sampler = FrameSampler(15, 0, 14, 50)
frames, m, warning = sampler.get_target_frames()
# Result: [0, 1, ..., 14] with m=1, warning=True
EOF
```

## File Structure

```
Labelling workflow/
├── main.py              # Entry point, batch orchestrator
├── Window.py            # Labelling UI (refactored)
├── BatchManager.py      # Batch processing logic
├── StartupDialog.py     # Configuration dialog
├── FrameSampler.py      # Frame selection algorithm
├── ZoomPanLabel.py      # Zoom/pan label widget
├── VideoParams.py       # Display parameter validation
├── _archive/            # Old files (archived)
│   ├── Window_old.py
│   └── CropLabel.py
└── README.md            # This file
```

## Future Enhancements

Possible improvements (not implemented):

1. **Keyboard shortcuts** for label entry (number keys)
2. **Hotkeys for sliders** (e.g., arrow keys adjust brightness)
3. **Progress persistence** across program crashes
4. **Video preview thumbnails** in queue
5. **Batch CSV filtering** (e.g., export non-zero weights only)
6. **Undo/redo** for weight entries
7. **Custom sampling strategies** (e.g., random, stratified)
8. **Multi-threaded video loading** for smoother navigation

## Troubleshooting

**Issue:** "No video files found in [directory]"
- **Solution:** Ensure directory contains `.mp4, .mov, .avi,` or `.mkv` files

**Issue:** "Frame took too long to read"
- **Solution:** Copy video to local disk if on network; or increase threshold

**Issue:** "CSV already contains data for this video"
- **Solution:** Video was already processed; remove from CSV if re-labelling needed

**Issue:** Fewer labels than target displayed
- **Solution:** Valid frame range is shorter than target n; warning shown during start

## Dependencies

- PyQt5 (GUI)
- OpenCV (cv2) for video processing
- NumPy (for image processing)
- Standard library: csv, os, sys, argparse, pathlib

---

**Last Updated:** December 4, 2025

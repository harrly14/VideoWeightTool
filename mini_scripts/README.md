# Mini Scripts

Utility scripts used during development. These are ad-hoc tools and may require modification for your specific use case.

## Scripts

| Script | Description | Status |
|--------|-------------|--------|
| `add_touchstones.py` | PyQt5 GUI for creating ground-truth labels for every Nth frame in a video | Working |
| `apply_uniform_edits.py` | Applies uniform video edits (brightness, contrast, saturation, crop) via FFmpeg | Working |
| `bulk_download_vids.py` | Downloads MP4 files from network share directories | Working |
| `check_split.py` | Checks validation dataset split statistics | Working |
| `edit_video_for_cnn.py` | PyQt5 GUI for cropping/preprocessing videos (256×64, CLAHE) | Working |
| `get_unique_weights.py` | Prints all distinct weight values across train/test/val datasets | Working |
| `getImageSize.py` | Reports unique image dimensions across dataset splits | Working |
| `resplit_data.py` | Re-splits label data into 70/15/15 train/val/test using sklearn | Working |
| `revise_frame_names.py` | Renames frame files from `frameNNN.jpg` to `<video>_<frame>.jpg` format | Working |
| `test_clahe_pipeline.py` | Tests CLAHE preprocessing consistency between dataset and process_video | Working |
| `vid_to_test_images.py` | Extracts all frames from a video via Qt dialog | Working |
| `videos_to_frames.py` | Batch extracts frames from videos matching a naming pattern | Working |

## Deprecated/Broken

| Script | Issue |
|--------|-------|
| `make_labels_file.py` | Incomplete stub with syntax errors |
| `remake_label_csvs.py` | Empty file |

## Usage

Most scripts are standalone and can be run directly:

```bash
cd mini_scripts
python <script_name>.py
```

Some scripts have hardcoded paths or patterns that may need adjustment for your setup.

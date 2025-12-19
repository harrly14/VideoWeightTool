"""
Scale OCR Video Processing Script

Processes video files to extract weight readings from seven-segment displays.
Outputs CSV with frame-by-frame weights and applies temporal smoothing.

Usage:
    python process_video.py --video path/to/video.mp4 --output results.csv
    python process_video.py --video video.mp4 --roi 100,50,400,150 --smoothing-window 5
    python process_video.py --video video.mp4 --batch-size 16 --checkpoint-every 1000 --save-video
    python process_video.py --video video.mp4 --ground-truth gt.csv --gt-tolerance 0.0
    python process_video.py --video video.mp4 --resume  # Resume from checkpoint
    python process_video.py --video video.mp4 --no-strict  # Disable aggressive flagging

Recommendation:
    For new or unprocessed videos (no ground-truth available), run with `--conservative`
    to maximize recall (the script will flag any frame that isn't extremely certain).
    Example:
        python process_video.py --video new_video.mp4 --output new_weights.csv --conservative
"""
import argparse, cv2, torch, pandas as pd, numpy as np
import os
import sys
import time
# Fix for GoPro/high-res videos with multiple streams
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(2**18)

import json
from pathlib import Path
from scipy.signal import medfilt
from tqdm import tqdm
from model import create_model
from dataset import get_transforms

def load_model(model_path, device):
    """Load trained model from checkpoint"""
    model = create_model(device=device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Assume the checkpoint is just the state_dict
            state_dict = checkpoint
        
        # Handle compiled model state_dict with _orig_mod prefix
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Model loaded from: {model_path}")
        return model
    except Exception as e:
        raise Exception(f"Failed to load model from {model_path}: {e}")

# ============================================================
# CONFIGURATION

def cleanup_checkpoint(checkpoint_path):
    """Remove checkpoint file after successful completion"""
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("  Checkpoint file cleaned up")

# ============================================================
# VIDEO PROCESSING
# ============================================================
def get_video_metadata(video_path):
    """Get video metadata without loading frames"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    metadata = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    
    cap.release()
    return metadata

def get_roi(frame, roi_coords=None):
    if roi_coords:
        try:
            # Expect a quad: list of 4 [x,y] points
            if isinstance(roi_coords, (list, tuple)) and len(roi_coords) == 4 and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in roi_coords):
                # Warp to CNN size (256x64)
                src_pts = np.float32(roi_coords)
                dst_pts = np.float32([[0, 0], [256, 0], [256, 64], [0, 64]])
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(frame, M, (256, 64), flags=cv2.INTER_LINEAR)
                return warped
            else:
                print(f"Warning: Expected quad ROI [[x,y],...], got {roi_coords!r}. Using full frame.")
                return frame
        except Exception as e:
            print(f"Warning: Error processing quad ROI: {e}. Using full frame.")
            return frame
    return frame

def apply_clahe(frame):
    """Apply CLAHE preprocessing to match training data pipeline."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    bgr_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return bgr_enhanced

def preprocess_frame(frame, transform):
    """Preprocess frame for model input with CLAHE enhancement."""
    enhanced = apply_clahe(frame)
    image = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    image_tensor = augmented['image']
    return image_tensor

# ============================================================
# INFERENCE
# ============================================================
def predict_weight_batch(model, frame_tensors, device):
    """
    Run model inference on a batch of frames.
    Args:
        model: trained model
        frame_tensors: list of tensors (C, H, W)
        device: torch device
    Returns:
        list of (predicted_weight, confidence) tuples
    """
    if len(frame_tensors) == 0:
        return []
    
    # Stack into batch: list of (C,H,W) -> (B, C, H, W)
    batch_tensor = torch.stack(frame_tensors).to(device)
    
    with torch.no_grad():
        # Forward pass
        log_probs, _ = model(batch_tensor)

        # Decode predictions using model's built-in decoder
        # log_probs shape: (seq_len, batch, num_classes)
        decoded_list = model.decode_predictions(log_probs, enforce_format=True)

        # Calculate confidence scores
        probs = torch.exp(log_probs)
        max_probs, _ = torch.max(probs, dim=2)  # (seq_len, batch)
        confidences = torch.mean(max_probs, dim=0)  # (batch,)

        # Calculate per-sample entropy across time steps
        eps = 1e-9
        entropy_per_timestep = - (probs * torch.log(probs + eps)).sum(dim=2)  # (seq_len, batch)
        entropies = torch.mean(entropy_per_timestep, dim=0)  # (batch,)

        # Pair predictions with confidences and entropies
        results = [(pred, conf.item(), ent.item()) for pred, conf, ent in zip(decoded_list, confidences, entropies)]

    return results

def process_video_streaming_batched(video_path, model, transform, roi_coords, device, 
                                    batch_size=8, checkpoint_every=1000, resume=False,
                                    frame_range=None):
    """
    Process video frame-by-frame with batch inference and checkpoint saving.
    Memory usage: O(batch_size) regardless of video length.
    
    Args:
        video_path: path to video file
        model: trained model
        transform: preprocessing transform
        roi_coords: ROI coordinates or None
        device: torch device
        batch_size: number of frames to process in one batch
        checkpoint_every: save checkpoint every N frames (0 = no checkpoints)
        resume: whether to resume from checkpoint
        frame_range: optional (start_frame, end_frame) tuple to limit processing
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Apply frame range limits
    range_start = 0
    range_end = total_frames - 1
    if frame_range is not None:
        range_start, range_end = frame_range
        range_start = max(0, range_start)
        range_end = min(total_frames - 1, range_end)
        print(f"Processing frame range: {range_start} - {range_end} ({range_end - range_start + 1} frames)")
    
    # Checkpoint management
    checkpoint_path = get_checkpoint_path(video_path) if checkpoint_every > 0 else None
    results = []
    start_frame = range_start
    
    # Seek to start of frame range
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    if resume and checkpoint_path:
        results, start_frame = load_checkpoint(checkpoint_path)
        if start_frame > 0:
            print(f"Skipping to frame {start_frame}/{total_frames}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_batch = []
    frame_nums = []
    frame_num = start_frame
    frames_to_process = range_end - start_frame + 1
    
    print(f"\n{'='*60}")
    print(f"Processing {frames_to_process} frames (batch_size={batch_size})...")
    print(f"{'='*60}")
    
    with tqdm(total=frames_to_process, desc="Processing frames", unit="frame", 
              initial=0, dynamic_ncols=True) as pbar:
        while frame_num <= range_end:
            ret, frame = cap.read()
            if not ret:
                break
            
            cropped = get_roi(frame, roi_coords)
            
            tensor = preprocess_frame(cropped, transform)
            
            frame_batch.append(tensor)
            frame_nums.append(frame_num)
            
            if len(frame_batch) >= batch_size:
                predictions = predict_weight_batch(model, frame_batch, device)
                
                for fn, (weight, confidence, entropy) in zip(frame_nums, predictions):
                    timestamp = fn / fps if fps > 0 else fn
                    results.append({
                        'frame_num': fn,
                        'timestamp': timestamp,
                        'raw_weight': weight,
                        'confidence': confidence,
                        'entropy': entropy
                    })
                
                pbar.update(len(frame_batch))
                
                frame_batch = []
                frame_nums = []
                
                if checkpoint_path and checkpoint_every > 0 and frame_num % checkpoint_every == 0:
                    save_checkpoint(checkpoint_path, results, frame_num)
            
            frame_num += 1
    
    if len(frame_batch) > 0:
        predictions = predict_weight_batch(model, frame_batch, device)
        for fn, (weight, confidence, entropy) in zip(frame_nums, predictions):
            timestamp = fn / fps if fps > 0 else fn
            results.append({
                'frame_num': fn,
                'timestamp': timestamp,
                'raw_weight': weight,
                'confidence': confidence,
                'entropy': entropy
            })
        pbar.update(len(frame_batch))
    
    cap.release()
    
    if checkpoint_path and checkpoint_every > 0:
        save_checkpoint(checkpoint_path, results, frame_num)
    
    return results, checkpoint_path

# ============================================================
# POST-PROCESSING
# ============================================================
def apply_temporal_smoothing(weights, window_size=5):
    """Apply median filter to smooth weight predictions"""
    float_weights = []
    for w in weights:
        try:
            float_weights.append(float(w))
        except (ValueError, TypeError):
            float_weights.append(0.0)
    
    if len(float_weights) == 0:
        return []
            
    if window_size % 2 == 0:
        window_size += 1
    
    window_size = min(window_size, len(float_weights))
    if window_size < 3:
        return float_weights
        
    smoothed = medfilt(float_weights, kernel_size=window_size)
    
    return smoothed

def detect_outliers(raw_weights, smoothed_weights, threshold=0.1):
    """
    Flag predictions that differ significantly from smoothed values.
    
    Args:
        raw_weights: list of raw weight predictions
        smoothed_weights: list of smoothed weights
        threshold: difference threshold in g (default: 0.1 = 100 grams)
    
    Returns:
        list of boolean flags
    """
    flags = []
    for raw, smooth in zip(raw_weights, smoothed_weights):
        try:
            r_val = float(raw)
            diff = abs(r_val - smooth)
            flags.append(diff > threshold)
        except (ValueError, TypeError):
            flags.append(True)  # Flag unparseable values
    return flags

def detect_sudden_changes(raw_weights, prev_smoothed, threshold=0.05):
    """Flag frames where the raw prediction suddenly deviates from previous smoothed value.

    Args:
        raw_weights: list of raw weight predictions (strings or float)
        prev_smoothed: list of smoothed weights (floats)
        threshold: g threshold to call a sudden change

    Returns:
        list of boolean flags (True = sudden change)
    """
    flags = []
    for i, raw in enumerate(raw_weights):
        try:
            r = float(raw)
            prev = prev_smoothed[i-1] if i > 0 else prev_smoothed[i]
            flags.append(abs(r - prev) > threshold)
        except (ValueError, TypeError):
            flags.append(True)
    return flags

import re
def check_format_flags(weights, pattern=r'^\d+\.\d{3}$'):
    """Flag predictions that don't match the expected numeric format 'X.XXX'.

    Returns a list of booleans where True = format is invalid.
    """
    flags = []
    prog = re.compile(pattern)
    for w in weights:
        try:
            s = str(w)
            flags.append(not bool(prog.match(s)))
        except Exception:
            flags.append(True)
    return flags


def load_ground_truth(gt_path):
    """Load ground-truth CSV and return a normalized mapping.

    Accepts CSVs that contain one of the frame columns: 'frame_num', 'frame', 'frameNumber'
    or a timestamp column 'timestamp'/'time'. The weight column may be named 'actual_weight',
    'actual', or 'weight'. Returns a pandas.DataFrame for flexible matching.
    """
    if gt_path is None:
        return None
    if not os.path.exists(gt_path):
        print(f"  Warning: Ground-truth file not found: {gt_path}")
        return None

    try:
        df = pd.read_csv(gt_path)
    except Exception as e:
        print(f"  Warning: Could not read ground-truth CSV: {e}")
        return None

    df_cols = {c.lower(): c for c in df.columns}
    frame_col = None
    for candidate in ('frame_num', 'frame', 'framenumber'):
        if candidate in df_cols:
            frame_col = df_cols[candidate]
            break
    time_col = None
    for candidate in ('timestamp', 'time'):
        if candidate in df_cols:
            time_col = df_cols[candidate]
            break

    weight_col = None
    for candidate in ('actual_weight', 'actual', 'weight'):
        if candidate in df_cols:
            weight_col = df_cols[candidate]
            break

    if weight_col is None:
        print('  Warning: Ground-truth CSV missing weight column (expecting actual_weight/actual/weight)')
        return None

    return {
        'df': df,
        'frame_col': frame_col,
        'time_col': time_col,
        'weight_col': weight_col
    }

def calculate_confidence_flags(confidences, threshold=0.3):
    """
    Flag low-confidence predictions.
    
    Args:
        confidences: list of confidence scores
        threshold: minimum acceptable confidence (default: 0.3)
    
    Returns:
        list of boolean flags (True = needs review)
    """
    return [c < threshold for c in confidences]

def calculate_entropy_flags(entropies, threshold=0.5):
    """Flag high-entropy (uncertain) predictions."""
    flags = []
    for e in entropies:
        try:
            flags.append(e > threshold)
        except Exception:
            flags.append(True)
    return flags

# ============================================================
# OUTPUT
# ============================================================
def save_results_csv(results, output_path, metadata):
    """Save predictions to CSV"""
    df = pd.DataFrame(results)

    # Include actual_weight column if present in any result
    cols = ['frame_num', 'timestamp', 'raw_weight', 'smoothed_weight', 'confidence', 'entropy']
    if any('actual_weight' in r for r in results):
        cols.append('actual_weight')
    cols.append('needs_review')
    # Ensure only existing columns are selected
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total frames: {len(df)}")
    print(f"Video duration: {df['timestamp'].max():.1f} seconds ({df['timestamp'].max()/60:.1f} minutes)")
    print(f"Flagged for review: {df['needs_review'].sum()} ({df['needs_review'].sum()/len(df)*100:.1f}%)")
    print(f"\nWeight Statistics (g):")
    print(df['smoothed_weight'].describe())
    print(f"\nConfidence Statistics:")
    print(df['confidence'].describe())

def create_annotated_video(video_path, results, output_path, roi_coords=None, frame_range=None):
    """Create video with predicted weights overlaid"""
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine frame range
    start_frame = 0
    end_frame = total_frames - 1
    if frame_range is not None:
        start_frame, end_frame = frame_range
        start_frame = max(0, start_frame)
        end_frame = min(total_frames - 1, end_frame)
        print(f"  Annotating frame range: {start_frame} - {end_frame}")
    
    # Try multiple codecs with fallback
    fourcc_options = [
        ('mp4v', 'MP4V'),
        ('avc1', 'H.264/AVC1'),
        ('X264', 'X264'),
        ('MJPG', 'Motion JPEG')
    ]
    
    out = None
    for codec, name in fourcc_options:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec) # type: ignore
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                print(f"  Using {name} codec")
                break
            else:
                out = None
        except Exception as e:
            print(f"  Warning: {name} codec failed: {e}")
            out = None
    
    if out is None or not out.isOpened():
        print("  Error: Could not initialize video writer with any codec")
        cap.release()
        return
    
    print(f"\nCreating annotated video...")
    
    # Seek to start frame if needed
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_idx = start_frame
    
    # Convert results to dict for faster lookup
    results_dict = {r['frame_num']: r for r in results}
    frames_to_process = end_frame - start_frame + 1
    
    with tqdm(total=frames_to_process, desc="Annotating frames", unit="frame", dynamic_ncols=True) as pbar:
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx in results_dict:
                res = results_dict[frame_idx]
                text = f"Weight: {res['smoothed_weight']:.3f} g (Conf: {res['confidence']:.2f})"
                color = (0, 255, 0) if not res['needs_review'] else (0, 0, 255)
                
                # Draw ROI box if applicable
                if roi_coords:
                    try:
                        # Quad ROI -> draw polygon
                        if isinstance(roi_coords, (list, tuple)) and len(roi_coords) == 4 and isinstance(roi_coords[0], (list, tuple)):
                            pts = np.array([[int(p[0]), int(p[1])] for p in roi_coords], dtype=np.int32)
                            cv2.polylines(frame, [pts.reshape((-1,1,2))], isClosed=True, color=(255,0,0), thickness=2)
                        else:
                            print(f"Warning: Expected quad ROI for annotation, got {roi_coords!r}. Skipping ROI drawing.")
                        cv2.putText(frame, text, (30, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    except Exception as e:
                        print(f"Warning: Error drawing ROI: {e}")
                        cv2.putText(frame, text, (30, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                else:
                    cv2.putText(frame, text, (30, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    
            out.write(frame)
            frame_idx += 1
            pbar.update(1)
        
    cap.release()
    out.release()
    print(f"Annotated video saved to: {output_path}")

# ============================================================
# CHECKPOINTING
# ============================================================
def get_checkpoint_path(video_path):
    """Generate checkpoint file path for a video"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    return f"{base_name}_checkpoint.json"

def save_checkpoint(checkpoint_path, results, last_frame):
    """Save processing progress to checkpoint file"""
    data = {
        'results': results,
        'last_frame': last_frame,
        'timestamp': time.time()
    }
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Checkpoint saved: {len(results)} results up to frame {last_frame}")
    except Exception as e:
        print(f"  Warning: Failed to save checkpoint: {e}")

def load_checkpoint(checkpoint_path):
    """Load processing progress from checkpoint file"""
    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        print(f"  Found checkpoint from {data.get('timestamp', 'unknown time')}")
        print(f"  Resuming from frame {data['last_frame']}")
        return data['results'], data['last_frame']
    except Exception as e:
        print(f"  Warning: Could not load checkpoint: {e}")
        return [], 0

# ============================================================
# MAIN WORKFLOW
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Scale OCR Video Processing Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python process_video.py --video path/to/video.mp4 --output results.csv
    python process_video.py --video video.mp4 --roi 883,407,1260,364,1325,505,868,548 --smoothing-window 5
    python process_video.py --video video.mp4 --batch-size 16 --checkpoint-every 1000 --save-video
    python process_video.py --video video.mp4 --ground-truth gt.csv --gt-tolerance 0.0
    python process_video.py --video video.mp4 --resume  # Resume from checkpoint
    python process_video.py --video video.mp4 --no-strict  # Disable aggressive flagging
        """
    )
    
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path to output CSV file (default: video_name_weights.csv)')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='Path to trained model')
    parser.add_argument('--roi', type=str, help='ROI as quad points: x1,y1,x2,y2,x3,y3,x4,y4 (auto-loaded from data/all_data.json if not provided)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--checkpoint-every', type=int, default=0, help='Save checkpoint every N frames (0 to disable)')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--save-video', action='store_true', help='Create annotated output video')
    parser.add_argument('--smoothing-window', type=int, help='Temporal smoothing window size (default: 3, or 1 in strict mode)')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, help='Confidence threshold for flagging')
    parser.add_argument('--strict', action='store_true', help='Enable strict flagging mode')
    parser.add_argument('--conservative', action='store_true', help='Conservative mode: only unflag extremely certain frames')
    parser.add_argument('--pass-confidence', type=float, default=0.95, help='Min confidence for conservative unflag')
    parser.add_argument('--pass-entropy', type=float, default=0.2, help='Max entropy for conservative unflag')
    parser.add_argument('--pass-delta', type=float, default=0.01, help='Max raw-smooth delta for conservative unflag')
    parser.add_argument('--ground-truth', type=str, help='Path to ground-truth CSV for validation')
    parser.add_argument('--gt-tolerance', type=float, default=0.05, help='Ground-truth tolerance in g')
    parser.add_argument('--gt-match-by', choices=['frame', 'time'], default='time', help='How to match GT rows')
    
    return parser.parse_args()

def main():
    """Main execution pipeline"""
    
    args = parse_args()
    
    # Require video argument
    if args.video is None:
        print("Error: --video argument is required.")
        return 1

    print("\n" + "="*60)
    print("SCALE OCR VIDEO PROCESSOR")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Checkpoint interval: {args.checkpoint_every} frames" if args.checkpoint_every > 0 else "Checkpoints: disabled")
    
    try:
        model = load_model(args.model, device)
    except Exception as e:
        print(f"\nError loading model: {e}")
        return 1
    
    print(f"\nLoading video: {args.video}")
    try:
        metadata = get_video_metadata(args.video)
        print(f"   Resolution: {metadata['width']}x{metadata['height']}")
        print(f"   FPS: {metadata['fps']:.2f}")
        print(f"   Total frames: {metadata['frame_count']}")
        print(f"   Duration: {metadata['frame_count']/metadata['fps']:.1f} seconds ({metadata['frame_count']/metadata['fps']/60:.1f} minutes)")
    except Exception as e:
        print(f"\nError loading video: {e}")
        return 1

    if not args.conservative and args.ground_truth is None:
        print("\nTIP: For new/unlabeled videos, consider running with --conservative to maximize recall and avoid unflagged errors.")
    
    # Load ROI from JSON if --roi not provided
    roi_coords = None
    if args.roi:
        # Parse --roi as quad: x1,y1,x2,y2,x3,y3,x4,y4
        try:
            coords = [int(x) for x in args.roi.split(',')]
            if len(coords) == 8:
                roi_coords = [[coords[i], coords[i+1]] for i in range(0, 8, 2)]
            else:
                print(f"Error: --roi must be 8 comma-separated integers for quad points, got {len(coords)}")
                return 1
        except ValueError as e:
            print(f"Error parsing --roi: {e}")
            return 1
    else:
        print("\nNo ROI specified. Loading from data/all_data.json...")
        video_name = os.path.basename(args.video)
        try:
            with open('data/all_data.json', 'r') as f:
                data = json.load(f)
            if 'rois' in data and video_name in data['rois']:
                roi_coords = data['rois'][video_name]['quad']
                print(f"   Loaded ROI for {video_name}: {roi_coords}")
            else:
                print(f"   No ROI found for {video_name} in data/all_data.json")
                print("   Please provide --roi argument or add ROI to the JSON file.")
                return 1
        except FileNotFoundError:
            print("   data/all_data.json not found.")
            print("   Please provide --roi argument or create the JSON file with ROIs.")
            return 1
        except json.JSONDecodeError as e:
            print(f"   Error parsing data/all_data.json: {e}")
            return 1
    
    transform = get_transforms(image_size=(256, 64), is_train=False)
    
    try:
        results, checkpoint_path = process_video_streaming_batched(
            args.video, model, transform, 
            roi_coords, device,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume
        )
    except KeyboardInterrupt:
        print("\n\nWarning: Processing interrupted by user (Ctrl+C)")
        print("Tip: Use --resume flag to continue from last checkpoint.")
        return 1
    except Exception as e:
        print(f"\nError during processing: {e}")
        print("Tip: Checkpoint saved. Use --resume flag to continue.")
        import traceback
        traceback.print_exc()
        return 1
    
    # Decide smoothing window: explicit --smoothing-window overrides other logic
    if args.smoothing_window is not None:
        smoothing_window = args.smoothing_window
    else:
        # strict mode disables smoothing by default (window=1)
        smoothing_window = 1 if args.strict else 3
    print(f"\nApplying temporal smoothing (window={smoothing_window})...")
    raw_weights = [r['raw_weight'] for r in results]
    smoothed_weights = apply_temporal_smoothing(raw_weights, smoothing_window)
    
    # Update results with smoothed weights
    for i, result in enumerate(results):
        result['smoothed_weight'] = smoothed_weights[i]
    
    print(f"Flagging problematic predictions...")

    # Determine thresholds (strict mode makes flagging aggressive)
    if args.strict:
        outlier_thresh = 0.05  # 50 g
        conf_thresh = max(args.confidence_threshold, 0.85)
        sudden_thresh = 0.05  # 50 g sudden jump
        format_check = True
        print(f"   STRICT mode: outlier={outlier_thresh} g, conf>={conf_thresh}, sudden={sudden_thresh} g, format_check=ON")
    else:
        outlier_thresh = 0.1  # 100 g
        conf_thresh = args.confidence_threshold
        sudden_thresh = 0.2
        format_check = False
        print(f"   outlier={outlier_thresh} g, confidence>={conf_thresh}, sudden={sudden_thresh} g")

    outlier_flags = detect_outliers(raw_weights, smoothed_weights, threshold=outlier_thresh)
    confidence_flags = calculate_confidence_flags(
        [r['confidence'] for r in results],
        conf_thresh
    )

    sudden_flags = detect_sudden_changes(raw_weights, smoothed_weights, threshold=sudden_thresh)
    format_flags = check_format_flags(raw_weights) if format_check else [False] * len(results)

    entropies = [r.get('entropy', 0.0) for r in results]
    entropy_thresh = 0.3 if args.strict else 0.6
    entropy_flags = calculate_entropy_flags(entropies, threshold=entropy_thresh)

    # Median disagreement flag: compare raw to median smoothed value
    try:
        median_sm = float(np.median(smoothed_weights))
    except Exception:
        median_sm = None
    median_flags = []
    if median_sm is not None:
        med_thresh = outlier_thresh * 0.7
        for rw in raw_weights:
            try:
                median_flags.append(abs(float(rw) - median_sm) > med_thresh)
            except Exception:
                median_flags.append(True)
    else:
        median_flags = [False] * len(results)

    for i, result in enumerate(results):
        # needs_review if any aggressive condition is met
        result['needs_review'] = bool(
            outlier_flags[i] or confidence_flags[i] or sudden_flags[i] or format_flags[i] or entropy_flags[i] or median_flags[i]
        )

    # Conservative mode: invert logic and only mark NOT needs_review when model is extremely certain
    if args.conservative:
        print(f"   CONSERVATIVE mode: only frames meeting high-certainty thresholds will be unflagged (pass_conf={args.pass_confidence}, pass_entropy={args.pass_entropy}, pass_delta={args.pass_delta})")
        for i, result in enumerate(results):
            try:
                conf_ok = float(result.get('confidence', 0.0)) >= args.pass_confidence
            except Exception:
                conf_ok = False
            try:
                ent_ok = float(result.get('entropy', 1.0)) <= args.pass_entropy
            except Exception:
                ent_ok = False
            try:
                raw = float(result.get('raw_weight'))
                smooth = float(result.get('smoothed_weight'))
                delta_ok = abs(raw - smooth) <= args.pass_delta
            except Exception:
                delta_ok = False
            fmt_ok = not format_flags[i]

            certain = conf_ok and ent_ok and delta_ok and fmt_ok
            if not certain:
                result['needs_review'] = True
    gt_info = None
    if args.ground_truth:
        gt_info = load_ground_truth(args.ground_truth)

    if gt_info is not None:
        df_gt = gt_info['df']
        frame_col = gt_info['frame_col']
        time_col = gt_info['time_col']
        weight_col = gt_info['weight_col']

        total_gt_found = 0
        total_mismatches = 0
        mismatches_flagged_before = 0
        mismatches_forced = 0

        frame_lookup = None
        if frame_col is not None:
            try:
                frame_lookup = {int(r[frame_col]): float(r[weight_col]) for _, r in df_gt.iterrows() if pd.notna(r[frame_col])}
            except Exception:
                frame_lookup = None

        for i, res in enumerate(results):
            actual = None
            if args.gt_match_by == 'frame' and frame_lookup is not None:
                actual = frame_lookup.get(int(res['frame_num']), None)
            if actual is None and time_col is not None:
                try:
                    diffs = (df_gt[time_col].astype(float) - float(res['timestamp'])).abs()
                    idx = int(diffs.idxmin())
                    actual = float(df_gt.loc[idx, weight_col])
                except Exception:
                    actual = None

            if actual is not None:
                total_gt_found += 1
                res['actual_weight'] = actual
                try:
                    raw_val = float(res['raw_weight'])
                except Exception:
                    raw_val = None
                try:
                    smooth_val = float(res['smoothed_weight'])
                except Exception:
                    smooth_val = None

                mismatch = False
                if raw_val is None and smooth_val is None:
                    mismatch = True
                else:
                    if raw_val is not None and abs(raw_val - actual) > args.gt_tolerance:
                        mismatch = True
                    if smooth_val is not None and abs(smooth_val - actual) > args.gt_tolerance:
                        mismatch = True

                if mismatch:
                    total_mismatches += 1
                    if res.get('needs_review', False):
                        mismatches_flagged_before += 1
                    else:
                        mismatches_forced += 1
                        res['needs_review'] = True

        print(f"\nGround-truth rows matched: {total_gt_found}")
        print(f"Total mismatches found vs GT: {total_mismatches}")
        print(f"Mismatches already flagged before GT enforcement: {mismatches_flagged_before}")
        print(f"Mismatches forced to flagged by GT enforcement: {mismatches_forced}")
        if total_mismatches > 0 and mismatches_forced == 0:
            print("All mismatches were already flagged by heuristics.")
        elif total_mismatches > 0:
            print("GT enforcement ensured all mismatches are flagged. Set --gt-tolerance smaller to be stricter.")
    
    output_path = args.output
    if output_path is None:
        base_name = os.path.splitext(args.video)[0]
        output_path = f"{base_name}_weights.csv"
        
    save_results_csv(results, output_path, metadata)
    
    if args.save_video:
        video_output = os.path.splitext(args.video)[0] + '_annotated.mp4'
        print(f"\nCreating annotated video: {video_output}")
        create_annotated_video(args.video, results, video_output, roi_coords)
    
    if checkpoint_path:
        cleanup_checkpoint(checkpoint_path)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    return 0

if __name__ == "__main__":
    exit(main())
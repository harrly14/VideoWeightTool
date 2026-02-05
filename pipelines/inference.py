"""
Scale OCR Video Processing Script

Processes video files to extract weight readings from seven-segment displays.
Outputs CSV with frame-by-frame weights and applies temporal smoothing.

Flags:
    Input/Output:
        --video PATH           Path to input video file (required)
        --output PATH          Path to output CSV file (default: <video>_weights.csv)
        --model PATH           Path to trained model (default: data/models/best_accuracy_model.pth)
        --roi X1,Y1,...,X4,Y4  ROI as 8 comma-separated quad points (auto-loaded from data/valid_video_sections.json if not provided)
        --save-video           Create annotated output video with weight overlay

    Processing:
        --batch-size N         Batch size for inference (default: 8)
        --smoothing-window N   Temporal median filter window size (default: 3, or 1 in strict mode)
        --checkpoint-every N   Save checkpoint every N frames, 0 to disable (default: 0)
        --resume               Resume processing from last checkpoint

    Flagging Modes:
        --strict               Aggressive flagging: tighter thresholds, format checking enabled
        --conservative         Flag everything except extremely certain frames (recommended for new videos)
        --confidence-threshold Minimum confidence to avoid flagging (default: 0.5)

    Conservative Mode Thresholds:
        --pass-confidence F    Min confidence to unflag (default: 0.95)
        --pass-entropy F       Max entropy to unflag (default: 0.2)
        --pass-delta F         Max raw-smooth delta to unflag (default: 0.01)

    Ground-Truth Validation:
        --ground-truth PATH    Path to ground-truth CSV for validation
        --gt-tolerance F       Tolerance in kg for GT mismatch (default: 0.05)
        --gt-match-by MODE     Match GT by 'frame' or 'time' (default: time)
"""
import argparse, cv2, torch, pandas as pd, numpy as np
import os
import sys
import time
import re
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(2**18)

import json
from pathlib import Path
from scipy.signal import medfilt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from core.model import create_model
from core.dataset import get_transforms

def get_roi_for_frame(frame_num, roi_sections):
    """
    Find the correct ROI for a given frame number from a list of sections.
    
    Args:
        frame_num: The frame number to look up
        roi_sections: List of section dicts with 'quad', 'start_frame', 'end_frame'
    
    Returns:
        ROI quad coordinates list, or None if frame not covered
    """
    if not roi_sections:
        return None
    
    for section in roi_sections:
        start = section.get('start_frame', 0)
        end = section.get('end_frame', float('inf'))
        if start <= frame_num <= end:
            return section.get('quad')
    
    return None  # Frame not covered by any section


def load_model(model_path, device):
    """Load trained model from checkpoint, including char_map if present."""
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
        
        # Load char_map from checkpoint if present, otherwise use default from config
        char_map = checkpoint.get('char_map', None) if isinstance(checkpoint, dict) else None
        
        model = create_model(device=device, char_map=char_map)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Model loaded from: {model_path}")
        if char_map is not None:
            print(f"  char_map loaded from checkpoint")
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
    if not roi_coords:
        return cv2.resize(frame, (256,64)) # fall back
    try:
        pts = np.array(roi_coords, dtype="float32")

        width = max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3]))
        height = max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1]))
        aspect_ratio = width / height

        target_height = 64
        target_width = 256
        new_height = target_height
        new_width = int(aspect_ratio * new_height)

        dst_pts = np.float32([[0,0], [new_width - 1, 0], [new_width - 1, new_height - 1], [0, new_height - 1]]) # -1 because its 0 indexed
        M = cv2.getPerspectiveTransform(pts, dst_pts)
        warped = cv2.warpPerspective(frame, M, (new_width, new_height), flags=cv2.INTER_LINEAR)

        #center on black canvas
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        if new_width > target_width: # unlikely case where its super wide
            warped = cv2.resize(warped, (target_width, target_height))
            new_width = target_width
        x_offset = (target_width - new_width) // 2
        canvas[:, x_offset:x_offset + new_width] = warped

        return canvas
    except Exception as e:
        print(f"ROI error: {e}")
        return cv2.resize(frame, (256,64))



def preprocess_frame(frame, transform):
    """Preprocess frame for model input with CLAHE enhancement."""
    # Just convert to RGB, as Albumentations expects RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Pass to the pipeline (which does CLAHE -> Crop -> Normalize -> Tensor)
    augmented = transform(image=image)
    image_tensor = augmented['image']
    
    return image_tensor

# ============================================================
# INFERENCE
# ============================================================
from core.config import MIN_NONBLANK_COUNT

def predict_weight_batch(model, frame_tensors, device):
    """
    Run model inference on a batch of frames.
    
    Returns raw model output (no format enforcement) with confidence and entropy
    calculated only over non-blank token predictions.
    
    Args:
        model: trained model
        frame_tensors: list of tensors (C, H, W)
        device: torch device
    Returns:
        list of (predicted_weight, confidence, entropy) tuples
        - predicted_weight: raw decoded string (may not match X.XXX format)
        - confidence: mean confidence over non-blank predictions (0.0 if too few non-blank tokens)
        - entropy: mean entropy over non-blank predictions
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
        # Return RAW output without format enforcement for honest validation
        decoded_list = model.decode_predictions(log_probs, enforce_format=False)

        # Calculate confidence and entropy over NON-BLANK tokens only
        probs = torch.exp(log_probs)  # (seq_len, batch, num_classes)
        max_probs, preds = torch.max(probs, dim=2)  # (seq_len, batch)
        
        # Blank label is the last class
        blank_label = probs.size(2) - 1
        non_blank_mask = (preds != blank_label).float()  # (seq_len, batch)
        
        # Count non-blank predictions per sample
        non_blank_counts = non_blank_mask.sum(dim=0)  # (batch,)
        
        # Calculate confidence: mean of max_probs over non-blank timesteps only
        masked_max_probs = max_probs * non_blank_mask  # Zero out blank timesteps
        # Avoid division by zero; use count or 1
        safe_counts = non_blank_counts.clamp(min=1)
        confidences = masked_max_probs.sum(dim=0) / safe_counts  # (batch,)
        
        # If too few non-blank tokens, confidence is unreliable -> set to 0
        confidences = torch.where(
            non_blank_counts >= MIN_NONBLANK_COUNT,
            confidences,
            torch.zeros_like(confidences)
        )

        # Calculate per-sample entropy across non-blank time steps only
        eps = 1e-9
        entropy_per_timestep = - (probs * torch.log(probs + eps)).sum(dim=2)  # (seq_len, batch)
        masked_entropy = entropy_per_timestep * non_blank_mask
        entropies = masked_entropy.sum(dim=0) / safe_counts  # (batch,)

        # Pair predictions with confidences and entropies
        results = [(pred, conf.item(), ent.item()) for pred, conf, ent in zip(decoded_list, confidences, entropies)]

    return results

def process_video_streaming_batched(video_path, model, transform, roi_sections, device, 
                                    batch_size=8, checkpoint_every=1000, resume=False,
                                    frame_range=None):
    """
    Process video frame-by-frame with batch inference and checkpoint saving.
    Memory usage: O(batch_size) regardless of video length.
    
    Args:
        video_path: path to video file
        model: trained model
        transform: preprocessing transform
        roi_sections: ROI sections list (each with 'quad', 'start_frame', 'end_frame')
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
            
            # Get the correct ROI for this frame
            roi_coords = get_roi_for_frame(frame_num, roi_sections)
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
from core.config import FILTER_CONF_THRESH, FILTER_ENT_THRESH, FILTER_JUMP_THRESH

# Flag reason codes for review UI
FLAG_REASONS = {
    'low_conf': 'Low confidence',
    'high_ent': 'High entropy',
    'bad_format': 'Invalid format',
    'jump': 'Physical jump',
    'parse_error': 'Parse error',
    'low_nonblank': 'Too few non-blank tokens',
}

def robust_filter_pipeline(results, conf_thresh=FILTER_CONF_THRESH, ent_thresh=FILTER_ENT_THRESH, jump_thresh=FILTER_JUMP_THRESH):
    # exactly 1 digit, dot, exactly 3 digits
    pattern = re.compile(r'^\d\.\d{3}$')
    
    filtered_weights = []
    flag_reasons = []
    last_valid = None
    
    for r in results:
        pred = r.get('raw_weight', '')
        conf = r.get('confidence', 0.0)
        ent = r.get('entropy', float('inf'))
        
        reason = None
        is_valid = False
        parsed_val = None
        str_pred = str(pred)
        
        if len(str_pred) != 5:
            reason = 'bad_format'
        elif not pattern.match(str_pred):
            reason = 'bad_format'
        else:
            try:
                parsed_val = float(pred)
            except (ValueError, TypeError):
                reason = 'parse_error'
        
        if reason is None:
            if conf == 0.0:
                reason = 'low_nonblank'
            elif conf < conf_thresh:
                reason = 'low_conf'
            elif ent > ent_thresh:
                reason = 'high_ent'
            elif last_valid is not None and abs(parsed_val - last_valid) > jump_thresh:
                reason = 'jump'
            else:
                is_valid = True
        
        if is_valid:
            last_valid = parsed_val
        
        filtered_weights.append(last_valid)
        flag_reasons.append(reason)
    
    return filtered_weights, flag_reasons


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

# ============================================================
# OUTPUT
# ============================================================
def save_results_csv(results, output_path, metadata, model=None):
    """
    Save predictions to CSV with format enforcement applied.
    
    Args:
        results: list of result dicts with raw_weight, smoothed_weight, etc.
        output_path: path to output CSV file
        metadata: video metadata dict
        model: optional model instance to use for enforce_weight_format
    """
    # Apply enforce_weight_format to smoothed_weight before export
    # This ensures CSV has clean X.XXX format for spreadsheet processing
    for r in results:
        sw = r.get('smoothed_weight')
        if sw is not None:
            if model is not None:
                r['smoothed_weight'] = model.enforce_weight_format(str(sw))
            else:
                # Fallback: simple formatting if model not provided
                try:
                    r['smoothed_weight'] = f"{float(sw):.3f}"
                except (ValueError, TypeError):
                    pass
    
    df = pd.DataFrame(results)

    # Include actual_weight and flag_reason columns if present
    cols = ['frame_num', 'timestamp', 'raw_weight', 'smoothed_weight', 'confidence', 'entropy']
    if any('actual_weight' in r for r in results):
        cols.append('actual_weight')
    cols.append('needs_review')
    if any('flag_reason' in r for r in results):
        cols.append('flag_reason')
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
    
    # Show breakdown by flag reason if available
    if 'flag_reason' in df.columns:
        print(f"\nFlag Reason Breakdown:")
        reason_counts = df['flag_reason'].value_counts(dropna=False)
        for reason, count in reason_counts.items():
            reason_str = reason if reason else 'Valid'
            print(f"  {reason_str}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\nWeight Statistics (kg):")
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
                val = res['smoothed_weight']

                weight_str = f"{val:.3f}" if val is not None else "N/A"
                text = f"Weight: {weight_str} kg (Conf: {res['confidence']:.2f})"
                color = (0, 255, 0) if not res['needs_review'] else (0, 0, 255)
                
                # Draw ROI box if applicable
                if roi_coords:
                    try:
                        if isinstance(roi_coords, (list, tuple)) and len(roi_coords) == 4 and isinstance(roi_coords[0], (list, tuple)):
                            pts = np.array([[int(p[0]), int(p[1])] for p in roi_coords], dtype=np.int32)
                            cv2.polylines(frame, [pts.reshape((-1,1,2))], isClosed=True, color=(255,0,0), thickness=2)
                            
                            # Place text at the ROI's top-left corner
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.7
                            thickness = 2
                            # bounding rect gives a stable "top-left" even for rotated quads
                            bx, by, bw, bh = cv2.boundingRect(pts)
                            offset_x = 5
                            offset_y = 5
                            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                            
                            candidate_y = by - offset_y
                            
                            text_x = bx + offset_x
                            
                            cv2.putText(frame, text, (int(text_x), int(candidate_y)),
                                      font, font_scale, color, thickness, cv2.LINE_AA)
                        else:
                            print(f"Warning: Expected quad ROI for annotation, got {roi_coords!r}. Skipping ROI drawing.")
                            cv2.putText(frame, text, (30, 50), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    except Exception as e:
                        print(f"Warning: Error drawing ROI: {e}")
                        # Fallback: slightly inset top-left of frame
                        cv2.putText(frame, text, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
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
    python process_video.py --video video.mp4 --roi 883,407,1260,364,1325,505,868,548 --confidence-threshold 0.7
    python process_video.py --video video.mp4 --batch-size 16 --checkpoint-every 1000 --save-video
    python process_video.py --video video.mp4 --ground-truth gt.csv --gt-tolerance 0.0
    python process_video.py --video video.mp4 --resume  # Resume from checkpoint
    python process_video.py --video video.mp4 --no-strict  # Disable aggressive flagging
        """
    )
    
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path to output CSV file (default: video_name_weights.csv)')
    parser.add_argument('--model', type=str, default='data/models/best_accuracy_model.pth', help='Path to trained model')
    parser.add_argument('--roi', type=str, help='ROI as quad points: x1,y1,x2,y2,x3,y3,x4,y4 (auto-loaded from data/valid_video_sections.json if not provided)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--checkpoint-every', type=int, default=0, help='Save checkpoint every N frames (0 to disable)')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--save-video', action='store_true', help='Create annotated output video')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, help='Confidence threshold for flagging')
    parser.add_argument('--strict', action='store_true', help='Enable strict flagging mode')
    parser.add_argument('--conservative', action='store_true', help='Conservative mode: only unflag extremely certain frames')
    parser.add_argument('--pass-confidence', type=float, default=0.95, help='Min confidence for conservative unflag')
    parser.add_argument('--pass-entropy', type=float, default=0.2, help='Max entropy for conservative unflag')
    parser.add_argument('--pass-delta', type=float, default=0.01, help='Max raw-smooth delta for conservative unflag')
    parser.add_argument('--ground-truth', type=str, help='Path to ground-truth CSV for validation')
    parser.add_argument('--gt-tolerance', type=float, default=0.05, help='Ground-truth tolerance in kg')
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
    roi_sections = None
    if args.roi:
        # Parse --roi as quad: x1,y1,x2,y2,x3,y3,x4,y4
        # This becomes a single "section" covering all frames
        try:
            coords = [int(x) for x in args.roi.split(',')]
            if len(coords) == 8:
                roi_quad = [[coords[i], coords[i+1]] for i in range(0, 8, 2)]
                # Wrap in sections format for compatibility
                roi_sections = [{'quad': roi_quad, 'start_frame': 0, 'end_frame': float('inf')}]
            else:
                print(f"Error: --roi must be 8 comma-separated integers for quad points, got {len(coords)}")
                return 1
        except ValueError as e:
            print(f"Error parsing --roi: {e}")
            return 1
    else:
        print("\nNo ROI specified. Loading from data/valid_video_sections.json...")
        video_name = os.path.basename(args.video)
        try:
            with open('data/valid_video_sections.json', 'r') as f:
                data = json.load(f)
            if 'rois' in data and video_name in data['rois']:
                video_data = data['rois'][video_name]
                if 'sections' in video_data:
                    roi_sections = video_data['sections']
                    print(f"   Loaded {len(roi_sections)} ROI section(s) for {video_name}")
                    for i, sec in enumerate(roi_sections):
                        print(f"      Section {i+1}: frames {sec['start_frame']}-{sec['end_frame']}")
                else:
                    print(f"   No 'sections' data found for {video_name}")
                    return 1
            else:
                print(f"   No ROI found for {video_name} in data/valid_video_sections.json")
                print("   Please provide --roi argument or add ROI to the JSON file.")
                return 1
        except FileNotFoundError:
            print("   data/valid_video_sections.json not found.")
            print("   Run `python workflows/labelling/main.py` to create it, or provide --roi argument.")
            return 1
        except json.JSONDecodeError as e:
            print(f"   Error parsing data/valid_video_sections.json: {e}")
            return 1
    
    transform = get_transforms(image_size=(256, 64), is_train=False)
    
    try:
        results, checkpoint_path = process_video_streaming_batched(
            args.video, model, transform, 
            roi_sections, device,
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
    
    print(f"Flagging problematic predictions...")

    # Determine thresholds (strict mode uses tighter thresholds)
    from core.config import (
        FILTER_CONF_THRESH, FILTER_ENT_THRESH, FILTER_JUMP_THRESH,
        STRICT_CONF_THRESH, STRICT_ENT_THRESH, STRICT_JUMP_THRESH
    )
    
    if args.strict:
        conf_thresh = max(args.confidence_threshold, STRICT_CONF_THRESH)
        ent_thresh = STRICT_ENT_THRESH
        jump_thresh = STRICT_JUMP_THRESH
        print(f"   STRICT mode: conf>={conf_thresh}, entropy<={ent_thresh}, jump<={jump_thresh} kg")
    else:
        conf_thresh = args.confidence_threshold
        ent_thresh = FILTER_ENT_THRESH
        jump_thresh = FILTER_JUMP_THRESH
        print(f"   Normal mode: conf>={conf_thresh}, entropy<={ent_thresh}, jump<={jump_thresh} kg")

    print(f"\nApplying robust filter pipeline...")
    smoothed_weights, flag_reasons = robust_filter_pipeline(
        results, 
        conf_thresh=conf_thresh,
        ent_thresh=ent_thresh,
        jump_thresh=jump_thresh
    )
    
    # Update results with filtered weights and flags
    for i, result in enumerate(results):
        result['smoothed_weight'] = smoothed_weights[i]
        result['flag_reason'] = flag_reasons[i]
        result['needs_review'] = flag_reasons[i] is not None

    # Conservative mode: apply stricter thresholds on top
    if args.conservative:
        print(f"   CONSERVATIVE mode: re-checking with pass_conf={args.pass_confidence}, pass_entropy={args.pass_entropy}, pass_delta={args.pass_delta}")
        pattern = re.compile(r'^\d+\.\d{3}$')
        for i, result in enumerate(results):
            # Skip if already flagged
            if result['needs_review']:
                continue
            
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
            fmt_ok = pattern.match(str(result.get('raw_weight', ''))) is not None

            certain = conf_ok and ent_ok and delta_ok and fmt_ok
            if not certain:
                result['needs_review'] = True
                result['flag_reason'] = 'conservative'

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
        
    save_results_csv(results, output_path, metadata, model=model)
    
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
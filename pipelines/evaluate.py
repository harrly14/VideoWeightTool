"""
Evaluation Script for Scale OCR Model

Runs inference on a validation dataset and outputs a detailed CSV report comparing
predictions against ground truth labels. Also provides options to inspect the worst errors.
"""

import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
import editdistance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.model import create_model
from core.dataset import ScaleOCRDataset, get_transforms
from core.config import FILTER_CONF_THRESH, FILTER_ENT_THRESH
from pipelines.inference import predict_weight_batch



def should_flag(prediction, confidence, entropy, conf_thresh, ent_thresh):
    """
    Check if a prediction should be flagged based on format and quality metrics.
    Does NOT check for sequential jumps as this is single-image evaluation.
    """
    pattern = re.compile(r'^\d+\.\d{3}$')
    
    if not pattern.match(prediction):
        return True, 'bad_format'
        
    try:
        float(prediction)
    except ValueError:
        return True, 'parse_error'
        
    if confidence == 0.0:
        return True, 'low_nonblank'
        
    if confidence < conf_thresh:
        return True, 'low_conf'
        
    if entropy > ent_thresh:
        return True, 'high_ent'
        
    return False, None

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from: {args.model}")
    try:
        checkpoint = torch.load(args.model, map_location=device)
        
        state_dict = checkpoint['model_state_dict']
        char_map = checkpoint.get('char_map', None)

        # Handle compiled model prefix
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
            
        model = create_model(device=device, char_map=char_map)
        model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return


    transforms = get_transforms(image_size=(256, 64), is_train=False)
    
    try:
        dataset = ScaleOCRDataset(
            labels_csv=args.labels,
            images_dir=args.images,
            transform=transforms,
            validate=True,
            verbose=False
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=None # default collate works for (img, label) tuple
    )
    
    print(f"Evaluating on {len(dataset)} images from {args.labels}")
    print(f"Results will be saved to: {args.output}")


    results_data = []
    
    # Inference and eval loop
    with torch.no_grad():
        for images, gt_weights_batch, filenames_batch in tqdm(dataloader, desc="Running Inference"):
            
            # Unpack batch tensor back to list of tensors for predict_weight_batch
            frame_tensors = [t for t in images]
            
            predictions = predict_weight_batch(model, frame_tensors, device)
            
            for i, (pred_str, conf, ent) in enumerate(predictions):
                gt_weight_str = gt_weights_batch[i]
                img_name = filenames_batch[i]
                img_path = str(Path(args.images) / img_name)
                
                is_correct = False
                try:
                    # Numeric comparison
                    val_pred = float(pred_str)
                    val_gt = float(gt_weight_str)
                    is_correct = abs(val_pred - val_gt) < 1e-5 # Tolerance for float equality
                except:
                    is_correct = False
                
                flagged, flag_reason = should_flag(
                    pred_str, conf, ent, 
                    args.confidence_threshold, 
                    args.entropy_threshold
                )
                
                lev_dist = editdistance.eval(pred_str, gt_weight_str)
                
                results_data.append({
                    'image_path': img_path,
                    'ground_truth': gt_weight_str,
                    'prediction': pred_str,
                    'is_correct': is_correct,
                    'confidence': conf,
                    'entropy': ent,
                    'was_flagged': flagged,
                    'flag_reason': flag_reason,
                    'levenshtein_distance': lev_dist
                })


    df_results = pd.DataFrame(results_data)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_results.to_csv(args.output, index=False)
    
    total = len(df_results)
    correct = df_results['is_correct'].sum()
    flagged = df_results['was_flagged'].sum()
    accuracy = (correct / total * 100) if total > 0 else 0
    flag_rate = (flagged / total * 100) if total > 0 else 0
    
    print("\n" + "="*40)
    print("EVALUATION SUMMARY")
    print("="*40)
    print(f"Total Images:     {total}")
    print(f"Accuracy (Num):   {accuracy:.2f}% ({correct}/{total})")
    print(f"Flagged Rate:     {flag_rate:.2f}% ({flagged}/{total})")
    print("-" * 40)
    
    if args.debug_n_images > 0:
        print(f"\nTOP {args.debug_n_images} WORST PREDICTIONS (by Levenshtein Distance)")
        print(f"{'Images (Incorrect Only)':<60} | {'GT':<10} | {'Pred':<10} | {'Dist':<5} | {'Conf':<6}")
        print("-" * 110)
        
        # Filter for incorrect ones, sort by distance descending
        incorrect_df = df_results[~df_results['is_correct']].copy()
        incorrect_df = incorrect_df.sort_values(by='levenshtein_distance', ascending=False)
        
        top_n = incorrect_df.head(args.debug_n_images)
        
        for _, row in top_n.iterrows():
            # Truncate path for cleaner display
            path_display = row['image_path'][-58:] if len(row['image_path']) > 58 else row['image_path']
            print(f"{path_display:<60} | {row['ground_truth']:<10} | {row['prediction']:<10} | {row['levenshtein_distance']:<5} | {row['confidence']:.4f}")
    
    print(f"\nFull report saved to: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Scale OCR Model")
    parser.add_argument('--model', type=str, default='data/models/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--labels', type=str, default='data/labels/val_labels.csv', help='Path to labels CSV')
    parser.add_argument('--images', type=str, default='data/images', help='Path to images directory')
    
    default_out = f"data/outputs/val_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    parser.add_argument('--output', type=str, default=default_out, help='Output CSV path')
    
    parser.add_argument('--batch-size', type=int, default=32, help='Inference batch size')
    parser.add_argument('--debug-n-images', type=int, default=5, help='Log N worst incorrect predictions')
    
    parser.add_argument('--confidence-threshold', type=float, default=FILTER_CONF_THRESH, help='Flagging confidence threshold')
    parser.add_argument('--entropy-threshold', type=float, default=FILTER_ENT_THRESH, help='Flagging entropy threshold')

    args = parser.parse_args()
    
    evaluate(args)

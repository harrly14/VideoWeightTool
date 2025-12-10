import torch
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import create_model
from dataset import ScaleOCRDataset, get_transforms
from train import CTCLabelEncoder

def validate_model(model_path, test_csv, images_dir, batch_size=32, device='cuda'):
    print(f"Loading model from {model_path}...")
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # CTCLabelEncoder has 12 classes (0-9, ., blank)
    encoder = CTCLabelEncoder()
    model = create_model(num_chars=len(encoder.char_to_idx), device=device)
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Strict load failed, trying non-strict: {e}")
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    model.eval()
    
    print(f"Loading test data from {test_csv}...")
    transform = get_transforms(image_size=(256, 64), is_train=False)
    dataset = ScaleOCRDataset(
        labels_csv=test_csv,
        images_dir=images_dir,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    total_cer = 0
    total_wer = 0 
    total_samples = 0
    correct_sequences = 0
    
    results = []
    
    with torch.no_grad():
        for images, weights, filenames in tqdm(loader):
            images = images.to(device)
            
            log_probs, _ = model(images)
            
            decoded_preds = model.decode_predictions(log_probs)
            
            for pred, target, fname in zip(decoded_preds, weights, filenames):
                is_correct = (pred == target)
                if is_correct:
                    correct_sequences += 1
                
                def levenshtein_distance(s1, s2):
                    if len(s1) < len(s2):
                        return levenshtein_distance(s2, s1)
                    if len(s2) == 0:
                        return len(s1)
                    previous_row = range(len(s2) + 1)
                    for i, c1 in enumerate(s1):
                        current_row = [i + 1]
                        for j, c2 in enumerate(s2):
                            insertions = previous_row[j + 1] + 1
                            deletions = current_row[j] + 1
                            substitutions = previous_row[j] + (c1 != c2)
                            current_row.append(min(insertions, deletions, substitutions))
                        previous_row = current_row
                    return previous_row[-1]

                dist = levenshtein_distance(pred, target)
                cer = dist / max(len(target), 1)
                
                total_cer += cer
                total_samples += 1
                
                results.append({
                    'filename': fname,
                    'target': target,
                    'prediction': pred,
                    'correct': is_correct,
                    'cer': cer
                })
                
    avg_cer = total_cer / total_samples
    seq_acc = (correct_sequences / total_samples) * 100
    
    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    print(f"Total Samples: {total_samples}")
    print(f"Sequence Accuracy: {seq_acc:.2f}%")
    print(f"Average CER: {avg_cer:.4f}")
    print("="*50)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_path = Path(test_csv).parent / 'validation_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Detailed results saved to {results_path}")
    
    # Show some errors
    errors = results_df[~results_df['correct']]
    if not errors.empty:
        print("\nSample Errors:")
        print(errors[['target', 'prediction', 'cer']].head(10))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--test_csv', default='data/labels/test_labels.csv', help='Path to test labels CSV')
    parser.add_argument('--images_dir', default='data/images', help='Path to images directory')
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)
        
    if not os.path.exists(args.test_csv):
        print(f"Test CSV not found: {args.test_csv}")
        sys.exit(1)
        
    validate_model(args.model, args.test_csv, args.images_dir)

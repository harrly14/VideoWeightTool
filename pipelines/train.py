import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import time
import json
import signal
import sys
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from core.dataset import create_dataloaders
from core.model import create_model
from core.config import IMAGE_SIZE, NUM_DIGIT_CLASSES

"""
Scale OCR Training Script

Trains a CNN digit classification model for reading scale weights from extracted 
frame crops. This file exposes the main training entrypoint `train_model(...)` 
and a CLI wrapper at the bottom of the module.

Quick usage (CLI):
    python pipelines/train.py            # run with defaults
    python pipelines/train.py --epochs 50 --batch-size 32 --resume path/to.ckpt
    python pipelines/train.py --no-amp   # disable mixed precision

Common defaults / locations:
    - Training data:         data/ (images/ and labels/ expected)
    - Model checkpoints:     data/models/
        - latest_model.pth
        - best_model.pth
        - best_accuracy_model.pth
    - TensorBoard logs:      runs/ (per-run subfolders like runs/scale_ocr_YYYYmmdd-HHMMSS)
      Monitor with:
        tensorboard --logdir runs

Tips:
    - You can also launch training from the TUI: run [video_weight_tool.py](video_weight_tool.py) -> "Training & Inference" -> "Train New Model".
    - Use --resume to continue from a saved checkpoint.
    - Use --no-amp to disable FP16/GradScaler if not desired or if troubleshooting instability.
    - Recommended Python: 3.8–3.12. CUDA-enabled GPU recommended for faster training.

See the `train_model` function for the full programmatic API and arguments.
"""

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SignalHandler:
    """
    Handles SIGINT (Ctrl+C) to allow for graceful shutdown.
    A second signal will force an immediate exit.
    """
    def __init__(self):
        self.received_signal = False
        self.force_exit = False
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signal, frame):
        if self.received_signal:
            print("\n" + "="*60)
            print("SECOND INTERRUPT RECEIVED. FORCING EXIT...")
            print("="*60 + "\n")
            self.force_exit = True
            sys.exit(1) # Force exit
        
        print("\n" + "="*60)
        print("INTERRUPT RECEIVED (Ctrl+C)")
        print("Attempting graceful shutdown. Finishing current epoch...")
        print("Press Ctrl+C again to force immediate exit.")
        print("="*60 + "\n")
        self.received_signal = True

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None):
    model.train()
    
    num_batches = len(train_loader)
    last_log_time = time.time()

    total_loss = 0.0
    total_acc = 0.0

    for batch_idx, (images, labels, frame_ids, slot_idxs) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        batch_acc = (logits.argmax(1) == labels).float().mean()
        total_loss += loss.item()
        total_acc += batch_acc.item()

        if (batch_idx + 1) % 10 == 0:
            now = time.time()
            interval = now - last_log_time
            speed = 10 / interval
            print(f"  Epoch [{epoch}] Batch [{batch_idx+1}/{num_batches}] "
                  f"Loss: {loss.item():.4f} Acc: {batch_acc.item():.2%} Speed: {speed:.1f} it/s")
            last_log_time = now
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_acc = total_acc / num_batches if num_batches > 0 else 0
    print(f"  Train Summary -> Loss: {avg_loss:.4f} | Acc: {avg_acc:.2%}")
    return avg_loss, avg_acc
    return avg_loss, avg_acc


def log_visual_validation(writer, epoch, images, targets, predictions):
    """
    Makes a grid of images with their predicted vs actual labels and logs to TensorBoard.
    """
    if writer is None or images is None or len(images) == 0:
        return

    # vutils.make_grid expects BxCxHxW
    max_imgs = min(len(images), 4)
    img_grid = vutils.make_grid(images[:max_imgs], normalize=True)
    writer.add_image('Validation/Images', img_grid, epoch)
    
    # Create a text string for the first few samples
    text_log = "### Validation Samples\n\n| Index | Ground Truth | Prediction | Status |\n|:---:|:---:|:---:|:---:|\n"
    for i in range(max_imgs):
        status = "Match" if targets[i] == predictions[i] else "Mismatch"
        text_log += f"| `{targets[i]}` | `{predictions[i]}` | {status} |\n"
    
    writer.add_text('Validation/Predictions', text_log, epoch)

def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    
    total_loss = 0.0
    total_correct_digits = 0
    total_digits = 0
    
    vis_images = None
    vis_targets = []
    vis_preds = []
    
    # for sequence accuracy
    sequence_data = {}  # dict of frame_id: {slot_idx: (predicted, target)}

    # for class-level accuracy
    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}

    with torch.no_grad():
        for batch_idx, (images, labels, frame_ids, slot_idxs) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            predictions = logits.argmax(1)
            
            # Digit accuracy logic
            correct = (predictions == labels)
            total_correct_digits += correct.sum().item()
            total_digits += labels.size(0)
            
            # Sequence accuracy logic & per-class logic
            for i in range(len(labels)):
                p = predictions[i].item()
                t = labels[i].item()
                f_id = frame_ids[i]
                s_idx = slot_idxs[i].item()

                if f_id not in sequence_data:
                    sequence_data[f_id] = {}
                sequence_data[f_id][s_idx] = (p, t)
                
                class_correct[t] += (p == t)
                class_total[t] += 1
            
            # Capture first batch for visualization
            if batch_idx == 0:
                vis_images = images.cpu()
                vis_targets = labels.cpu().tolist()
                vis_preds = predictions.cpu().tolist()
                
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    digit_acc = (total_correct_digits / total_digits * 100) if total_digits > 0 else 0.0
    
    # Calculate sequence accuracy
    correct_seqs = 0
    total_valid_seqs = 0
    
    for f_id, slots in sequence_data.items():
        if len(slots) == 4:
            total_valid_seqs += 1
            is_correct = all(slots[s][0] == slots[s][1] for s in range(4))
            if is_correct:
                correct_seqs += 1
                
    seq_acc = (correct_seqs / total_valid_seqs * 100) if total_valid_seqs > 0 else 0.0

    print(f"  Valid Summary -> Loss: {avg_loss:.4f} | "
          f"Digit Acc: {digit_acc:.2f}% | "
          f"Seq Acc: {seq_acc:.2f}% ({correct_seqs}/{total_valid_seqs})")
    
    # Optionally print per-class breakdown for worst classes to track imbalance issues
    # print("  Class Accuracies:")
    # for c in range(10):
    #     if class_total[c] > 0:
    #         c_acc = class_correct[c] / class_total[c] * 100
    #         if c_acc < 80.0 or class_total[c] < 5:  # highlight weak/rare classes
    #             print(f"    Class {c}: {c_acc:.1f}% ({class_correct[c]}/{class_total[c]})")
    
    return avg_loss, digit_acc, seq_acc, vis_images, vis_targets, vis_preds


def train_model(
    batch_size=16,
    num_epochs=50,
    learning_rate=0.001,
    image_size=IMAGE_SIZE,
    save_dir='data/models',
    data_dir='data',
    resume_from=None,
    use_amp=True,
    seed=42,
    run_name=None
):
    """
    Main training function
    
    Args:
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        image_size: (width, height) for image resizing
        save_dir: Directory to save models
        data_dir: Directory containing data
        resume_from: Path to checkpoint to resume from (optional)
        use_amp: Whether to use automatic mixed precision
        seed: Random seed for reproducibility
        run_name: Name for the run (tensorboard logging)
    """
    seed_everything(seed)
    signal_handler = SignalHandler()
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    # initialize tensorboard writer 
    # creates the folder 'runs/scale_ocr_{timestamp}'
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    if run_name:
        run_name_clean = ""
        for c in run_name:
            if c.isalnum() or c in "-_":
                run_name_clean += c
        log_dir = f"runs/{run_name_clean}_{timestamp}"
    else:
        log_dir = f"runs/scale_ocr_{timestamp}"

    writer = SummaryWriter(log_dir=log_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"TRAINING SCALE OCR MODEL")
    print(f"{'-'*50}")
    print(f"  Device:       {device}")
    print(f"  Data Dir:     {data_dir}")
    print(f"  Batch:        {batch_size}")
    print(f"  Epochs:       {num_epochs}")
    print(f"  Learning Rate:{learning_rate}")
    print(f"  FP16 Mode:    {use_amp}")
    print(f"  Logging to:   {log_dir}")
    print(f"{'='*50}\n")
    
    print("Loading data and model...")
    # Use number of CPUs for workers, but cap at 8 to avoid overhead
    num_workers = min(os.cpu_count() or 2, 8)
    
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        verbose=False
    )
    
    model = create_model(
        num_classes=NUM_DIGIT_CLASSES,
        device=device.type
    )

    # Log model graph to Tensorboard
    try:
        dummy_input = torch.zeros(1, 1, image_size[1], image_size[0]).to(device)
        writer.add_graph(model, dummy_input)
    except Exception as e:
        print(f"Warning: Failed to add graph to TensorBoard: {e}")

    if hasattr(torch, 'compile'):
        try:
            print("Attempting to compile model with torch.compile()...")
            model = torch.compile(model)
        except:
            print(f"torch.compile() skipped.")
    
    # Calculate class weights for imbalance from train dataset
    train_digits = train_dataset.labels_df['digit_label'].astype(int)
    class_counts = train_digits.value_counts().to_dict()
    print("\nClass Distribution in Training Set:")
    for c in range(10):
        c_count = class_counts.get(c, 0)
        print(f"  Digit {c}: {c_count:5d} samples")
        
    weights = torch.ones(NUM_DIGIT_CLASSES, dtype=torch.float32)
    max_count = max(class_counts.values()) if class_counts else 1
    for c in range(NUM_DIGIT_CLASSES):
        c_count = class_counts.get(c, 0)
        if c_count > 0:
            # Floor out the weight so rare classes aren't over-weighted to > 10x
            W = min(max_count / c_count, 10.0) 
            weights[c] = W
    weights = weights / weights.mean() # normalize
    print(f"\nClass Weights for Loss: \n{weights.numpy()}\n")
    
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Only use GradScaler if on CUDA
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8
    )
    
    best_val_loss = float('inf')
    best_seq_accuracy = 0.0
    start_epoch = 0
    final_metrics_hparams = {}
    
    if resume_from and Path(resume_from).exists():
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        # Handle torch.compile prefix based on model's compiled state
        fixed_state_dict = {}
        model_has_orig = hasattr(model, '_orig_mod')  # Check if model is compiled
        
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                if model_has_orig:
                    fixed_state_dict[k] = v  # Keep prefix if model is compiled
                else:
                    fixed_state_dict[k[len('_orig_mod.'):]] = v  # Strip prefix if model is not compiled
            else:
                if model_has_orig:
                    fixed_state_dict['_orig_mod.' + k] = v  # Add prefix if model is compiled
                else:
                    fixed_state_dict[k] = v  # Keep as-is if model is not compiled
        
        model.load_state_dict(fixed_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
             scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_seq_accuracy = checkpoint.get('best_seq_acc', 0.0)
        best_digit_accuracy = checkpoint.get('best_digit_acc', 0.0)
        
    print(f"\nStarting training loop for {num_epochs - start_epoch} epochs...")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'digit_acc': [], 'seq_acc': []}
    
    try:
        for epoch in range(start_epoch, num_epochs):
            if signal_handler.received_signal:
                break

            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            

            train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, device, epoch+1,
                    scaler=scaler
            )
            
            if signal_handler.received_signal:
                break

            val_loss, digit_acc, seq_acc, v_imgs, v_targs, v_preds = validate(model, val_loader, criterion, device, epoch+1)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['digit_acc'].append(digit_acc)
            history['seq_acc'].append(seq_acc)

            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Metrics/Train_Acc', train_acc, epoch)
            writer.add_scalar('Metrics/Digit_Acc', digit_acc, epoch)
            writer.add_scalar('Metrics/Seq_Acc', seq_acc, epoch)
            
            # Visual validation
            if v_imgs is not None:
                log_visual_validation(writer, epoch, v_imgs, v_targs, v_preds)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('HyperParams/Learning_Rate', current_lr, epoch)
            writer.flush()
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'best_digit_acc': getattr(model, 'best_digit_accuracy', digit_acc),
                'best_seq_acc': getattr(model, 'best_seq_accuracy', seq_acc),
            }
            
            torch.save(checkpoint, save_path / 'latest_model.pth')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, save_path / 'best_model.pth')
                print(f"\nSaved new best model (val_loss: {val_loss:.4f})")
            
            if seq_acc > best_seq_accuracy:
                best_seq_accuracy = seq_acc
                torch.save(checkpoint, save_path / 'best_accuracy_model.pth')
                print(f"Saved best accuracy model (seq_acc: {seq_acc:.2f}%)")
    finally:
        writer.close()
    print(f"\nTraining complete. Best sequence accuracy: {best_seq_accuracy:.2f}%")
    
    return model, history, criterion, device, test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Scale OCR Model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.00025, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing data')
    parser.add_argument('--save-dir', type=str, default='data/models', help='Directory to save models')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--run-name', type=str, default=None, help='Name for TensorBoard run')
    
    args = parser.parse_args()

    model, history, criterion, device, test_loader = train_model(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        data_dir=args.data_dir,
        resume_from=args.resume,
        use_amp=not args.no_amp,
        seed=args.seed,
        run_name=args.run_name
    )
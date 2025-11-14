import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import json

from dataset import create_dataloaders
from model import create_model

# note: this file is largely vibe coded. I wish I had the time to understand it, but oh well
"""
requires the dir structure to be like: 
CNN_stuff/
├── dataset.py
├── model.py
├── train.py
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels.csv
│   ├── val/
│   │   ├── images/
│   │   └── labels.csv
│   └── test/
│       ├── images/
│       └── labels.csv
└── models/             (will be created automatically)
"""

class CTCLabelEncoder:
    """
    Encode text labels to indices for CTC loss
    Handles conversion between strings like "7.535" and tensor indices
    """
    def __init__(self):
        # Character to index mapping
        self.char_to_idx = {
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
            '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            '.': 10
        }
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.blank_label = 11
    
    def encode(self, text):
        # encode a string to list of indices
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]
    
    def encode_batch(self, texts):
        # Encode a batch of strings

        encoded = [self.encode(text) for text in texts]
        target_lengths = torch.LongTensor([len(enc) for enc in encoded])
        
        # Concatenate all targets into single tensor (required by CTC loss)
        targets = torch.cat([torch.LongTensor(enc) for enc in encoded])
        
        return targets, target_lengths


def train_one_epoch(model, train_loader, criterion, optimizer, encoder, device, epoch):
    """
    Train for one epoch
    
    Args:
        model: The neural network
        train_loader: DataLoader for training data
        criterion: Loss function (CTCLoss)
        optimizer: Optimizer (Adam)
        encoder: Label encoder
        device: 'cpu' or 'cuda'
        epoch: Current epoch number
    
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()  # Set to training mode
    
    total_loss = 0
    num_batches = len(train_loader)
    
    print(f"\nEpoch {epoch} - Training")
    print("-" * 60)
    
    start_time = time.time()
    
    for batch_idx, (images, weights, filenames) in enumerate(train_loader):
        # Move images to device
        images = images.to(device)
        
        # Encode labels
        targets, target_lengths = encoder.encode_batch(weights)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        
        # Forward pass
        log_probs, output_lengths = model(images)
        output_lengths = output_lengths.to(device)
        
        # Calculate CTC loss
        loss = criterion(log_probs, targets, output_lengths, target_lengths)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"Warning: NaN loss detected at batch {batch_idx}")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            avg_loss_so_far = total_loss / (batch_idx + 1)
            elapsed = time.time() - start_time
            batches_per_sec = (batch_idx + 1) / elapsed
            
            print(f"  Batch [{batch_idx+1}/{num_batches}] | "
                  f"Loss: {loss.item():.4f} | "
                  f"Avg Loss: {avg_loss_so_far:.4f} | "
                  f"Speed: {batches_per_sec:.2f} batch/s")
    
    avg_loss = total_loss / num_batches
    epoch_time = time.time() - start_time
    
    print(f"\nEpoch {epoch} Training Complete:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Time: {epoch_time:.2f}s")
    
    return avg_loss


def validate(model, val_loader, criterion, encoder, device, epoch):
    """
    Validate the model
    
    Args:
        model: The neural network
        val_loader: DataLoader for validation data
        criterion: Loss function (CTCLoss)
        encoder: Label encoder
        device: 'cpu' or 'cuda'
        epoch: Current epoch number
    
    Returns:
        avg_loss: Average validation loss
        accuracy: Character-level accuracy
        sequence_accuracy: Full sequence accuracy (exact match)
    """
    model.eval()  # Set to evaluation mode
    
    total_loss = 0
    total_chars = 0
    correct_chars = 0
    total_sequences = 0
    correct_sequences = 0
    
    print(f"\nEpoch {epoch} - Validation")
    print("-" * 60)
    
    with torch.no_grad():  # No gradient computation during validation
        for batch_idx, (images, weights, filenames) in enumerate(val_loader):
            # Move to device
            images = images.to(device)
            
            # Encode labels
            targets, target_lengths = encoder.encode_batch(weights)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            # Forward pass
            log_probs, output_lengths = model(images)
            output_lengths = output_lengths.to(device)
            
            # Calculate loss
            loss = criterion(log_probs, targets, output_lengths, target_lengths)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
            
            # Decode predictions
            # use a slightly larger max_length (or compute from batch) to avoid truncation
            max_len = max(len(w) for w in weights) if len(weights) > 0 else 10
            predictions = model.decode_predictions(log_probs, max_length=max_len + 2)
            
            # Calculate accuracy
            for pred, target in zip(predictions, weights):
                total_sequences += 1
                
                # Exact match (sequence accuracy)
                if pred == target:
                    correct_sequences += 1
                
                # count missing chars as incorrect
                max_l = max(len(pred), len(target))
                for i in range(max_l):
                    total_chars += 1
                    p_char = pred[i] if i < len(pred) else None
                    t_char = target[i] if i < len(target) else None
                    if p_char == t_char:
                        correct_chars += 1
    
    avg_loss = total_loss / len(val_loader)
    char_accuracy = (correct_chars / total_chars * 100) if total_chars > 0 else 0
    seq_accuracy = (correct_sequences / total_sequences * 100) if total_sequences > 0 else 0
    
    print(f"\nValidation Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Character Accuracy: {char_accuracy:.2f}%")
    print(f"  Sequence Accuracy: {seq_accuracy:.2f}%")
    
    # Show some example predictions
    print(f"\nExample Predictions:")
    num_examples = min(5, len(predictions))
    for i in range(num_examples):
        print(f"  Pred: '{predictions[i]}' | Target: '{weights[i]}'")
    
    return avg_loss, char_accuracy, seq_accuracy


def train_model(
    train_dir='data/train',
    val_dir='data/val',
    test_dir='data/test',
    batch_size=16,
    num_epochs=50,
    learning_rate=0.001,
    hidden_size=256,
    num_lstm_layers=2,
    image_size=(256, 64),
    save_dir='models',
    resume_from=None
):
    """
    Main training function
    
    Args:
        train_dir: Path to training data
        val_dir: Path to validation data
        test_dir: Path to test data (optional)
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        hidden_size: LSTM hidden size
        num_lstm_layers: Number of LSTM layers
        image_size: (width, height) for image resizing
        save_dir: Directory to save models
        resume_from: Path to checkpoint to resume from (optional)
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"TRAINING SCALE OCR MODEL")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Image size: {image_size}")
    print(f"{'='*60}\n")
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=1
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    if test_dataset:
        print(f"  Test: {len(test_dataset)} samples")

    # Create label encoder
    encoder = CTCLabelEncoder()
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        num_chars=len(encoder.char_to_idx),
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        device=device.type
    )
    
    # Loss function (CTC Loss)
    criterion = nn.CTCLoss(blank=encoder.blank_label, zero_infinity=True)
    
    # Optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler (reduce LR when validation loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Tracking best model
    best_val_loss = float('inf')
    best_seq_accuracy = 0.0
    start_epoch = 0
    
    # Resume from checkpoint if provided
    if resume_from and Path(resume_from).exists():
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {start_epoch}")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'char_accuracy': [],
        'seq_accuracy': [],
        'learning_rate': []
    }
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, encoder, device, epoch+1
        )
        
        # Validate
        val_loss, char_acc, seq_acc = validate(
            model, val_loader, criterion, encoder, device, epoch+1
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['char_accuracy'].append(char_acc)
        history['seq_accuracy'].append(seq_acc)
        history['learning_rate'].append(current_lr)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'char_accuracy': char_acc,
            'seq_accuracy': seq_acc,
            'best_val_loss': best_val_loss,
            'history': history
        }
        
        # Save latest model
        torch.save(checkpoint, save_path / 'latest_model.pth')
        
        # Save best model (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, save_path / 'best_model.pth')
            print(f"\n✓ Saved new best model (val_loss: {val_loss:.4f})")
        
        # Also save if best sequence accuracy
        if seq_acc > best_seq_accuracy:
            best_seq_accuracy = seq_acc
            torch.save(checkpoint, save_path / 'best_accuracy_model.pth')
            print(f"✓ Saved best accuracy model (seq_acc: {seq_acc:.2f}%)")
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, save_path / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"✓ Saved checkpoint at epoch {epoch+1}")
    
    # Save training history
    with open(save_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best sequence accuracy: {best_seq_accuracy:.2f}%")
    print(f"Models saved to: {save_path}")
    print(f"{'='*60}\n")
    
    return model, history


if __name__ == "__main__":
    # Train the model
    model, history = train_model(
        train_dir='data/train',
        val_dir='data/val',
        test_dir='data/test',
        batch_size=8, # Reduce to 8 or 4 if running out of memory
        num_epochs=50,
        learning_rate=0.001,
        hidden_size=256,
        num_lstm_layers=2,
        image_size=(256, 64),
        save_dir='models'
    )
    
    print("\nTraining finished! Next steps:")
    print("1. Check models folder for saved models")
    print("2. Best model is saved as 'best_model.pth'")
    print("3. Use this model for inference on new videos")
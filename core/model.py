"""
model.py - Neural Network Architecture for Scale OCR

This file defines the CNN+LSTM+CTC model that learns to read
seven-segment displays from scale images.

Architecture:
    Input (1, CNN_HEIGHT, CNN_WIDTH) grayscale image
    CNN layers (extract visual features)
    LSTM layers (read features as sequence)
    Linear layer (predict characters)
    CTC decoder
    Output: "X.XXX" string
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import CHAR_MAP, CNN_WIDTH, CNN_HEIGHT, NUM_CHARS, HIDDEN_SIZE, NUM_LSTM_LAYERS


class ScaleOCRModel(nn.Module):
    # Derived CNN output dimensions — computed once from config constants.
    # Pool1 (2,2) and Pool2 (2,2) halve both dimensionss
    # Pool4 and Pool6 are (2,1)so they halve height only.  
    # Net effect:
    #   seq_len  = CNN_WIDTH  // 4          (two (2,2) width halvings)
    #   feat_h   = CNN_HEIGHT // 16         (two (2,2) + two (2,1) height halvings)
    SEQ_LEN = CNN_WIDTH // 4
    FEAT_H  = CNN_HEIGHT // 16

    def __init__(self, num_chars=NUM_CHARS, hidden_size=HIDDEN_SIZE, num_lstm_layers=NUM_LSTM_LAYERS, char_map=None):
        super(ScaleOCRModel, self).__init__()
        
        self.num_chars = num_chars
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        
        # Use provided char_map or fall back to config default
        # char_map: index -> character (e.g., {0: '0', 1: '1', ..., 10: '.'})
        self.char_map = char_map if char_map is not None else CHAR_MAP.copy()
        # Also store inverse map for encoding: character -> index
        self.char_to_idx = {v: k for k, v in self.char_map.items()}
        
        # CTC needs an extra "blank" character, so output is num_chars + 1
        self.num_classes = num_chars + 1  
        
        # Part 1: CNN Feature Extractor
        # Input: (batch, 1, CNN_HEIGHT, CNN_WIDTH)  e.g. (batch, 1, 128, 192)
        # Goal: Extract visual features (edges, segments, shapes)
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # (64, H/2, W/2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # (128, H/4, W/4)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d((2, 1))  # Height only: (256, H/8, W/4)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool6 = nn.MaxPool2d((2, 1))  # Height only: (512, H/16, W/4)
        
        # After all conv layers: (batch, 512, FEAT_H, SEQ_LEN)
        # Reshape to (batch, SEQ_LEN, 512 * FEAT_H)
        # SEQ_LEN becomes the time-step axis for the LSTM.

        lstm_input_size = 512 * self.FEAT_H

        # Part 2: LSTM Sequence Reader
        # Input: (batch, SEQ_LEN, lstm_input_size)
        # Goal: Read the sequence and understand digit order
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,  # Read both forward and backward
            batch_first=True,
            dropout=0.4 if num_lstm_layers > 1 else 0
        )
        
        # Bidirectional LSTM
        lstm_output_size = hidden_size * 2
        
        # Part 3: Classification Head
        # Input: (batch, SEQ_LEN, lstm_output_size)
        # Goal: Predict character probabilities at each time step
        
        self.fc = nn.Linear(lstm_output_size, self.num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input images (batch, 1, CNN_HEIGHT, CNN_WIDTH)
        
        Returns:
            log_probs: Log probabilities (SEQ_LEN, batch, num_classes)
            output_lengths: Length of each sequence (all same: SEQ_LEN)
        """
        batch_size = x.size(0)
        
        # CNN Feature Extraction
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # Block 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        
        # Block 6
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.pool6(x)
        
        # Current shape: (batch, 512, FEAT_H, SEQ_LEN)
        
        # Reshape for LSTM
        # We want (batch, SEQ_LEN, features)
        # Treat width (SEQ_LEN) as sequence length
        # Flatten height and channels into features: 512 * FEAT_H
        
        x = x.permute(0, 3, 2, 1)       # (batch, SEQ_LEN, FEAT_H, 512)
        x = x.reshape(batch_size, self.SEQ_LEN, -1)  # (batch, SEQ_LEN, 512*FEAT_H)
        
        # LSTM Sequence Processing
        x, _ = self.lstm(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification
        x = self.fc(x)
        
        # Prepare for CTC Loss
        # CTC expects: (seq_len, batch, num_classes)
        x = x.permute(1, 0, 2)  # (SEQ_LEN, batch, num_classes)
        
        # CTC expects log probabilities
        log_probs = F.log_softmax(x, dim=2)
        
        # Output lengths (all sequences have same length after CNN)
        output_lengths = torch.full(
            size=(batch_size,),
            fill_value=self.SEQ_LEN,
            dtype=torch.long,
            device=x.device
        )
        
        return log_probs, output_lengths
    
    def indices_to_string(self, indices):
        """Convert list of indices back to string."""
        return ''.join(self.char_map.get(i, '') for i in indices)
    
    def decode_predictions(self, log_probs, blank_label=11, max_length=5, enforce_format=False):
        """
        Decode CTC output to text strings (greedy decoding)
        
        Args:
            log_probs: (seq_len, batch, num_classes)
            blank_label: Index of blank character (default: 11)
            max_length: Maximum output length (default: 5 for X.XXX format)
            enforce_format: If True, force X.XXX format (default: False)
        
        Returns:
            List of decoded strings
        """
        # default blank is last class
        blank_label = self.num_classes - 1

        # Get most likely character at each time step
        _, preds = torch.max(log_probs, dim=2)  # (seq_len, batch)
        preds = preds.transpose(0, 1)  # (batch, seq_len)
        
        decoded_strings = []
        
        for pred in preds:
            pred = pred.tolist()
            
            # CTC collapse: remove repeated characters and blanks
            decoded = []
            prev_char = None
            
            for char_idx in pred:
                if char_idx == blank_label:
                    prev_char = None
                    continue
                
                if char_idx != prev_char:
                    decoded.append(char_idx)
                    prev_char = char_idx
                
                if len(decoded) >= max_length:
                    break
            
            decoded_str = self.indices_to_string(decoded)
            
            if enforce_format:
                decoded_str = self.enforce_weight_format(decoded_str)
            
            decoded_strings.append(decoded_str)
        
        return decoded_strings
    
    def enforce_weight_format(self, text):
        """
        Enforce X.XXX format on decoded text.
        
        This handles cases where CTC collapses the dot or places it wrongly.
        Since the format is strictly X.XXX, we extract all digits and force
        the dot at index 1.
        """
        digits_only = ''.join(c for c in text if c.isdigit())
        
        if len(digits_only) >= 4:
            # take first 4 digits and force into X.XXX form
            return f"{digits_only[0]}.{digits_only[1:4]}"
        elif len(digits_only) == 3:
            # Assume leading zero was missed: 0.XXX
            return f"0.{digits_only}"
        elif len(digits_only) == 2:
            # Pad with trailing zeros: 0.XX0
            # could also be X.X00 but that isn't handled here (or at all)
            return f"0.{digits_only}0"
        elif len(digits_only) == 1:
            # Single digit: X.000
            return f"{digits_only}.000"
        else:
            return "0.000"


def create_model(num_chars=NUM_CHARS, hidden_size=HIDDEN_SIZE, num_lstm_layers=NUM_LSTM_LAYERS, device='cpu', char_map=None):
    """
    Factory function to create and initialize the model
    
    Args:
        num_chars: Number of unique characters (default: NUM_CHARS from config)
        hidden_size: LSTM hidden size (default: HIDDEN_SIZE from config)
        num_lstm_layers: Number of LSTM layers (default: NUM_LSTM_LAYERS from config)
        device: Device to put model on ('cpu' or 'cuda')
        char_map: Optional character map (index -> char). If None, uses CHAR_MAP from config.
    
    Returns:
        Initialized model on specified device
    """
    model = ScaleOCRModel(
        num_chars=num_chars,
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        char_map=char_map
    )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")
    
    return model


if __name__ == "__main__":
    print("="*60)
    print("TESTING MODEL ARCHITECTURE")
    print("="*60 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    model = create_model(device=device)
    
    print("\nTesting forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, CNN_HEIGHT, CNN_WIDTH).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    log_probs, output_lengths = model(dummy_input)
    
    seq_len = ScaleOCRModel.SEQ_LEN
    print(f"Output log_probs shape: {log_probs.shape}")  # Should be (SEQ_LEN, 4, 12)
    print(f"Output lengths: {output_lengths}")  # Should be [SEQ_LEN]*4
    
    # Test decoding without limits
    print("\nTesting decoding (no limits)...")
    decoded_unlimited = model.decode_predictions(log_probs, max_length=100)
    print(f"Unlimited predictions (garbage, untrained):")
    for i, pred in enumerate(decoded_unlimited[:2]):  # Just show 2
        print(f"  Sample {i}: '{pred}'")

    # Test decoding with max_length
    print("\nTesting decoding (max_length=5)...")
    decoded = model.decode_predictions(log_probs, max_length=5)
    print(f"Limited predictions:")
    for i, pred in enumerate(decoded):
        print(f"  Sample {i}: '{pred}'")

    # Test with format enforcement
    print("\nTesting with format enforcement...")
    decoded_enforced = model.decode_predictions(log_probs, max_length=5, enforce_format=True)
    print(f"Enforced format predictions:")
    for i, pred in enumerate(decoded_enforced):
        print(f"  Sample {i}: '{pred}'")
        
    print(f"Decoded predictions (random, untrained):")
    for i, pred in enumerate(decoded):
        print(f"  Sample {i}: '{pred}'")
    
    print("\n" + "="*60)
    print("MODEL READY FOR TRAINING")
    print("="*60)
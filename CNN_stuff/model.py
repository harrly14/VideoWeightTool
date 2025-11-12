# note: this file is largely vibe coded. I wish I had the time to understand it, but oh well

"""
model.py - Neural Network Architecture for Scale OCR

This file defines the CNN+LSTM+CTC model that learns to read
seven-segment displays from scale images.

Architecture:
    Input (3, 64, 256) image
        ↓
    CNN layers (extract visual features)
        ↓
    LSTM layers (read features as sequence)
        ↓
    Linear layer (predict characters)
        ↓
    CTC decoder
        ↓
    Output: "X.XXX" string
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleOCRModel(nn.Module):
    def __init__(self, num_chars=11, hidden_size=256, num_lstm_layers=2):
        super(ScaleOCRModel, self).__init__()
        
        self.num_chars = num_chars
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        
        # Character set: 0-9 and '.'
        # CTC needs an extra "blank" character, so output is num_chars + 1
        self.num_classes = num_chars + 1  # 12 classes total
        
        # Part 1: CNN Feature Extractor
        # Input: (batch, 3, 64, 256)
        # Goal: Extract visual features (edges, segments, shapes)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # Reduces to (64, 32, 128)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # Reduces to (128, 16, 64)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d((2, 1))  # Reduces height only: (256, 8, 64)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool6 = nn.MaxPool2d((2, 1))  # Reduces height only: (512, 4, 64)
        
        # After all conv layers: (batch, 512, 4, 64)
        # We'll reshape this to (batch, 64, 512*4) = (batch, 64, 2048)
        # The 64 becomes our sequence length (time steps)
        

        # Part 2: LSTM Sequence Reader
        # Input: (batch, seq_len=64, features=2048)
        # Goal: Read the sequence and understand digit order
        
        self.lstm = nn.LSTM(
            input_size=512 * 4,  # 2048 features per time step
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,  # Read both forward and backward
            batch_first=True,
            dropout=0.3 if num_lstm_layers > 1 else 0
        )
        
        # Bidirectional LSTM
        lstm_output_size = hidden_size * 2
        

        # Part 3: Classification Head
        # Input: (batch, seq_len=64, lstm_output_size)
        # Goal: Predict character probabilities at each time step
        
        self.fc = nn.Linear(lstm_output_size, self.num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input images (batch, 3, 64, 256)
        
        Returns:
            log_probs: Log probabilities (seq_len, batch, num_classes)
            output_lengths: Length of each sequence (all same: seq_len)
        """
        batch_size = x.size(0)
        
        # CNN Feature Extraction
        
        # Block 1: (batch, 3, 64, 256) → (batch, 64, 32, 128)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2: (batch, 64, 32, 128) → (batch, 128, 16, 64)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3: (batch, 128, 16, 64) → (batch, 256, 16, 64)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Block 4: (batch, 256, 16, 64) → (batch, 256, 8, 64)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # Block 5: (batch, 256, 8, 64) → (batch, 512, 8, 64)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        
        # Block 6: (batch, 512, 8, 64) → (batch, 512, 4, 64)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.pool6(x)
        
        # Current shape: (batch, 512, 4, 64)
        
        # Reshape for LSTM
        # We want: (batch, seq_len, features)
        # Treat width (64) as sequence length
        # Flatten height and channels into features: 512 * 4 = 2048
        
        # Permute to (batch, 64, 4, 512)
        x = x.permute(0, 3, 2, 1)
        
        # Reshape to (batch, 64, 2048)
        x = x.reshape(batch_size, 64, -1)
        
        # LSTM Sequence Processing
        # Input: (batch, seq_len=64, features=2048)
        # Output: (batch, seq_len=64, hidden_size*2)
        
        x, _ = self.lstm(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification
        # Input: (batch, seq_len=64, hidden_size*2)
        # Output: (batch, seq_len=64, num_classes=12)
        
        x = self.fc(x)
        
        # Prepare for CTC Loss
        # CTC expects: (seq_len, batch, num_classes)
        # So we need to permute
        
        x = x.permute(1, 0, 2)  # (seq_len=64, batch, num_classes=12)
        
        # Apply log softmax (CTC expects log probabilities)
        log_probs = F.log_softmax(x, dim=2)
        
        # Output lengths (all sequences have same length after CNN)
        output_lengths = torch.full(
            size=(batch_size,),
            fill_value=64,  # seq_len
            dtype=torch.long,
            device=x.device
        )
        
        return log_probs, output_lengths
    
    def indices_to_string(self, indices):
        """
        Convert list of indices back to string.
        
        Args:
            indices: List of integer indices
        
        Returns:
            String representation
        """
        # Assuming char_map is defined in __init__ (add it if not)
        if not hasattr(self, 'char_map'):
            self.char_map = {str(i): i for i in range(10)}
            self.char_map['.'] = 10
        inv_map = {v: k for k, v in self.char_map.items()}
        return ''.join(inv_map.get(i, '') for i in indices)
    
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
        # Get most likely character at each time step
        _, preds = torch.max(log_probs, dim=2)  # (seq_len, batch)
        preds = preds.transpose(0, 1)  # (batch, seq_len)
        
        decoded_strings = []
        
        for pred in preds:
            # Convert to list
            pred = pred.tolist()
            
            # CTC collapse: remove repeated characters and blanks
            decoded = []
            prev_char = None
            
            for char_idx in pred:
                # Skip blanks
                if char_idx == blank_label:
                    prev_char = None
                    continue
                
                # Skip repeated characters
                if char_idx != prev_char:
                    decoded.append(char_idx)
                    prev_char = char_idx
                
                # Stop if we've reached max length
                if len(decoded) >= max_length:
                    break
            
            # Convert indices to string
            decoded_str = self.indices_to_string(decoded)
            
            # Optionally enforce X.XXX format
            if enforce_format:
                decoded_str = self.enforce_weight_format(decoded_str)
            
            decoded_strings.append(decoded_str)
        
        return decoded_strings
    
    def enforce_weight_format(self, text):
        """
        Enforce X.XXX format on decoded text
        
        Args:
            text: Decoded string (may be malformed)
        
        Returns:
            String in X.XXX format, or original if can't fix
        """
        # Remove any characters that aren't digits or decimal
        cleaned = ''.join(c for c in text if c.isdigit() or c == '.')
        
        # If we have at least 4 digits, try to construct X.XXX
        digits_only = ''.join(c for c in cleaned if c.isdigit())
        
        if len(digits_only) >= 4:
            # Take first 4 digits and format as X.XXX
            return f"{digits_only[0]}.{digits_only[1:4]}"
        elif len(digits_only) == 3:
            # If only 3 digits, assume 0.XXX
            return f"0.{digits_only}"
        else:
            # Can't fix, return cleaned version
            return cleaned if cleaned else "0.000"


def create_model(num_chars=11, hidden_size=256, num_lstm_layers=2, device='cpu'):
    """
    Factory function to create and initialize the model
    
    Args:
        num_chars: Number of unique characters (default: 11)
        hidden_size: LSTM hidden size (default: 256)
        num_lstm_layers: Number of LSTM layers (default: 2)
        device: Device to put model on ('cpu' or 'cuda')
    
    Returns:
        Initialized model on specified device
    """
    model = ScaleOCRModel(
        num_chars=num_chars,
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers
    )
    
    # Move to device
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")
    
    return model


# for testing: 
if __name__ == "__main__":
    print("="*60)
    print("TESTING MODEL ARCHITECTURE")
    print("="*60 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    model = create_model(device=device)
    
    print("\nTesting forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 64, 256).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    log_probs, output_lengths = model(dummy_input)
    
    print(f"Output log_probs shape: {log_probs.shape}")  # Should be (64, 4, 12)
    print(f"Output lengths: {output_lengths}")  # Should be [64, 64, 64, 64]
    
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
"""
model.py - Neural Network Architecture for Scale OCR

This file defines the CNN classifier model for recognizing 
scale display digits from grayscale images.

Architecture:
    Input (1, CNN_HEIGHT, CNN_WIDTH) grayscale image
    CNN layers (extract visual features)
    Linear classifier (predict class logits)
    Output: class index
"""

import torch
import torch.nn as nn

from core.config import CNN_WIDTH, CNN_HEIGHT, NUM_DIGIT_CLASSES

class DigitCNN(nn.Module):
    def __init__(self, num_classes=NUM_DIGIT_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64,kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # allow for different digit widths by fixing the spatial output to 4x4 pixels
            nn.AdaptiveAvgPool2d((4,4))  
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 128), # 128*4*4 = 2048 features
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128,num_classes)
            #outputs raw logits
        )

    def forward(self,x):
        """
        passes input through features, then flattens
        returns raw logits
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def create_model(num_classes=NUM_DIGIT_CLASSES, device='cpu'):
    model = DigitCNN(num_classes).to(device)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"DigitCNN: {total_params:,} params ({trainable_params:,} trainable)")
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
    
    logits = model(dummy_input)
    probs = torch.softmax(logits, dim=1)
    pred_classes = torch.argmax(probs, dim=1)

    print(f"Output logits shape: {logits.shape}")  # Should be (batch_size, NUM_DIGIT_CLASSES)
    print(f"Predicted classes: {pred_classes.tolist()}")

    print("\nSample class probabilities (first 2 samples):")
    for i in range(min(2, batch_size)):
        print(f"  Sample {i}: {probs[i].tolist()}")
    
    print("\n" + "="*60)
    print("MODEL READY FOR TRAINING")
    print("="*60)
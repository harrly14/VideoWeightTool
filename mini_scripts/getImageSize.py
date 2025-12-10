import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import ScaleOCRDataset
train_dataset = ScaleOCRDataset('data/train/images', 'data/train/labels.csv', validate=False, verbose=False)
test_dataset = ScaleOCRDataset('data/test/images', 'data/test/labels.csv', validate=False, verbose=False)
val_dataset = ScaleOCRDataset('data/val/images', 'data/val/labels.csv', validate=False, verbose=False)

image_sizes = set()
for dataset in [train_dataset, test_dataset, val_dataset]:
    for img, _, _ in dataset:
        image_sizes.add((img.shape))
print(image_sizes)
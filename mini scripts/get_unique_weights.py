import sys
sys.path.append('/home/sully/Desktop/projects/VideoWeightTool')

from dataset import ScaleOCRDataset
train_dataset = ScaleOCRDataset('data/train/images', 'data/train/labels.csv', validate=False, verbose=False)
test_dataset = ScaleOCRDataset('data/test/images', 'data/test/labels.csv', validate=False, verbose=False)
val_dataset = ScaleOCRDataset('data/val/images', 'data/val/labels.csv', validate=False, verbose=False)

distinct_weights = set()
for dataset in [train_dataset, test_dataset, val_dataset]:
    for _, weight, _ in dataset:
        distinct_weights.add(weight)
print(distinct_weights)
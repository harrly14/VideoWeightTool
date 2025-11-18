import sys
sys.path.append('/home/sully/Desktop/projects/VideoWeightTool')

from CNN_stuff.dataset import ScaleOCRDataset
train_dataset = ScaleOCRDataset('CNN_stuff/data/train/images', 'CNN_stuff/data/train/labels.csv', validate=False, verbose=False)
test_dataset = ScaleOCRDataset('CNN_stuff/data/test/images', 'CNN_stuff/data/test/labels.csv', validate=False, verbose=False)
val_dataset = ScaleOCRDataset('CNN_stuff/data/val/images', 'CNN_stuff/data/val/labels.csv', validate=False, verbose=False)

distinct_weights = set()
for dataset in [train_dataset, test_dataset, val_dataset]:
    for _, weight, _ in dataset:
        distinct_weights.add(weight)
print(distinct_weights)
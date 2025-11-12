import sys
sys.path.append('/home/sully/Desktop/projects/VideoWeightTool')

from CNN_stuff.dataset import ScaleOCRDataset
train_dataset = ScaleOCRDataset('CNN_stuff/data/train/images', 'CNN_stuff/data/train/labels.csv', validate=False, verbose=False)
test_dataset = ScaleOCRDataset('CNN_stuff/data/test/images', 'CNN_stuff/data/test/labels.csv', validate=False, verbose=False)
val_dataset = ScaleOCRDataset('CNN_stuff/data/val/images', 'CNN_stuff/data/val/labels.csv', validate=False, verbose=False)

image_sizes = set()
for dataset in [train_dataset, test_dataset, val_dataset]:
    for img, _, _ in dataset:
        image_sizes.add((img.shape))
print(image_sizes)

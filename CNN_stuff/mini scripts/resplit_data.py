from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Read CSV labels directly
train_labels = pd.read_csv('CNN_stuff/data/old_labels/train_labels.csv')
val_labels = pd.read_csv('CNN_stuff/data/old_labels/labels.csv')
test_labels = pd.read_csv('CNN_stuff/data/old_labels/test_labels.csv')

all_labels = pd.concat([train_labels, val_labels, test_labels], ignore_index=True)
all_labels = all_labels[all_labels['weight'] != 0]

train, temp = train_test_split(all_labels, test_size=0.30, random_state=42, shuffle=True)
val, test = train_test_split(temp, test_size=0.50, random_state=42, shuffle=True)

# Save new CSVs
out_dir = 'CNN_stuff/data/labels'
os.makedirs(out_dir, exist_ok=True)
train.to_csv(os.path.join(out_dir, 'train_labels.csv'), index=False)
val.to_csv(os.path.join(out_dir, 'val_labels.csv'), index=False)
test.to_csv(os.path.join(out_dir, 'test_labels.csv'), index=False)
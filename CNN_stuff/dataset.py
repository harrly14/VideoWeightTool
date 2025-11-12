import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from functools import partial

class ScaleOCRDataset(Dataset):
    def __init__(self, images_dir, labels_csv, file_extension='.jpg', 
                 transform=None, validate=True, verbose=True, weight_range=(0.0, 100.0)):
        self.images_dir = Path(images_dir)
        self.file_extension = file_extension
        self.transform = transform
        self.verbose = verbose
        self.weight_range = weight_range

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not Path(labels_csv).exists():
            raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

        df = pd.read_csv(labels_csv)

        original_count = len(df)
        df = df[df['weight'] != 0].reset_index(drop=True)

        filtered_count = original_count - len(df)
        if filtered_count > 0 and self.verbose:
            print(f"Filtered out {filtered_count} rows with weight=0")

        self.labels_df = df

        if self.verbose:
            print(f"Loaded dataset with {len(self.labels_df)} samples")
            print(f"Images directory: {self.images_dir}")
            print(f"Sample labels:\n{self.labels_df.head()}")

        if validate:
            self._validate_dataset()

    def _validate_dataset(self):
        if self.verbose: print("\nValidating dataset...")

        missing_files = []
        corrupt_files = []
        invalid_weights = []

        for i in range(len(self.labels_df)):
            row = self.labels_df.iloc[i]
            frame_num = row['frame_number'] 
            weight = row['weight']
            filename = f"{row['filename']}_{frame_num}{self.file_extension}"
            img_path = self.images_dir / filename

            if not img_path.exists():
                missing_files.append((i, filename))
            
            try:
                image = cv2.imread(str(img_path))
                if image is None: corrupt_files.append((i, filename))
            except Exception as e:
                corrupt_files.append((i, f"{filename} (error: {str(e)})"))

            try:
                float_weight = float(weight)
                if not self.weight_range[0] < float_weight < self.weight_range[1]:
                    corrupt_files.append((i, filename, weight))
            except (ValueError, TypeError):
                invalid_weights.append((i, filename, weight))

            if (i+1) % 100 == 0 and self.verbose:
                print(f"Validated {i+1}/{len(self.labels_df)} samples...")
        if self.verbose: print("\n" + "="*60 + "\nVALIDATION RESULTS\n" + "="*60)

        if not missing_files and not corrupt_files and not invalid_weights:
            if self.verbose:
                print("✓ All samples validated successfully!")
                print(f"✓ All {len(self.labels_df)} images can be loaded")
                print(f"✓ All weights are valid")
        else:
            if missing_files and self.verbose:
                print(f"\n✗ Found {len(missing_files)} missing image files:")
                for idx, filename in missing_files[:10]:
                    print(f"    Row {idx}: {filename}")
                if len(missing_files) > 10:
                    print(f"    ... and {len(missing_files) - 10} more")
            
            if corrupt_files and self.verbose:
                print(f"\n✗ Found {len(corrupt_files)} corrupt/unreadable images:")
                for idx, filename in corrupt_files[:10]:
                    print(f"    Row {idx}: {filename}")
                if len(corrupt_files) > 10:
                    print(f"    ... and {len(corrupt_files) - 10} more")
            
            if invalid_weights and self.verbose:
                print(f"\n✗ Found {len(invalid_weights)} invalid weights:")
                for idx, filename, weight in invalid_weights[:10]:
                    print(f"    Row {idx}: {filename} -> weight={weight}")
                if len(invalid_weights) > 10:
                    print(f"    ... and {len(invalid_weights) - 10} more")
            
            if self.verbose:
                print(f"\n⚠ Removing {len(missing_files) + len(corrupt_files) + len(invalid_weights)} problematic rows...")
            bad_indices = set([idx for idx, _ in missing_files] + 
                            [idx for idx, _ in corrupt_files] + 
                            [idx for idx, _, _ in invalid_weights])
            
            self.labels_df = self.labels_df.drop(index=list(bad_indices)).reset_index(drop=True)
            if self.verbose: print(f"✓ Dataset now has {len(self.labels_df)} valid samples")

        if self.verbose: print("="*60 + "\n")

        if len(self.labels_df) == 0:
            raise ValueError("No valid samples remain after validation")
        
        weights = self.labels_df['weight'].astype(float)
        if self.verbose:
            print(f"Weight statistics:")
            print(f"  Min: {weights.min():.3f}")
            print(f"  Max: {weights.max():.3f}")
            print(f"  Mean: {weights.mean():.3f}")
            print(f"  Median: {weights.median():.3f}")

    def __len__(self):
        return len(self.labels_df)
    def __getitem__(self, index):
        row = self.labels_df.iloc[index]
        frame_num = row['frame_number'] 
        weight = row['weight']
        filename = f"{row['filename']}_{frame_num}{self.file_extension}"

        img_path = self.images_dir / filename
        image = cv2.imread(str(img_path))

        if image is None:
            raise ValueError(f"Could not load images from {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, str(weight), filename
def get_transforms(image_size=(256, 64), is_train=False):
    target_width, target_height = image_size
    
    if is_train:
        # augmentation + resizing w padding
        return A.Compose([ # type: ignore
            # augmentation
            A.Rotate(limit=5, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.Sharpen(alpha=(0.2, 0.5), p=0.3),

            # resize
            A.LongestMaxSize(max_size=max(target_width, target_height)),

            A.PadIfNeeded(
                min_height=target_height,
                min_width=target_width,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(0, 0, 0),
                position='center'
            ),
            # crop
            A.CenterCrop(height=target_height, width=target_width),
            
            # normalize
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            
            ToTensorV2(),
        ])
    else: 
        return A.Compose([
            A.LongestMaxSize(max_size=max(target_width, target_height)),
            A.PadIfNeeded(
                min_height=target_height,
                min_width=target_width,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(0, 0, 0),
                position='center'
            ),
            A.CenterCrop(height=target_height, width=target_width),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
def create_dataloaders(train_dir, val_dir, test_dir=None, batch_size=16, image_size=(256,64), num_workers=2):
    print("Creating dataloaders...")

    train_transform = get_transforms(image_size, is_train=True)
    val_transform = get_transforms(image_size, is_train=False)

    train_dataset = ScaleOCRDataset(
        f"{train_dir}/images", 
        f"{train_dir}/labels.csv",
        transform=train_transform,
        validate=True
    )
    
    val_dataset = ScaleOCRDataset(
        f"{val_dir}/images", 
        f"{val_dir}/labels.csv",
        transform=val_transform,
        validate=True
    )
    
    test_dataset = None
    if test_dir:
        test_dataset = ScaleOCRDataset(
            f"{test_dir}/images", 
            f"{test_dir}/labels.csv",
            transform=val_transform,
            validate=True
        )
    
    # DataLoaders
    create_dataloader = partial(DataLoader, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    train_loader = create_dataloader(train_dataset, shuffle=True)
    val_loader = create_dataloader(val_dataset)
    test_loader = None
    if test_dataset:
        test_loader = create_dataloader(test_dataset)
    
    print(f"Dataloaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    if test_loader:
        print(f"  Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset



# for testing purposes:  
if __name__ == "__main__":
    print("="*60)
    print("TESTING DATASET WITH TRANSFORMS")
    print("="*60 + "\n")
    
    train_loader, val_loader, test_loader, _, _, _ = create_dataloaders(
        train_dir='data/train',
        val_dir='data/val',
        test_dir='data/test',
        batch_size=8,
        image_size=(256, 64)
    )
    
    images, weights, filenames = next(iter(train_loader))
    
    print(f"\nBatch shapes after transforms:")
    print(f"  Images: {images.shape}")  # should be [8, 3, 64, 256]
    print(f"  First 3 weights: {weights[:3]}")
    print(f"  First 3 filenames: {filenames[:3]}")
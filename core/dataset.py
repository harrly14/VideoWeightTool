import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
import json
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from typing import Optional
from pathlib import Path
from functools import partial
import torch.nn.functional as F

from core.config import IMAGE_SIZE
from core.roi_utils import get_roi_for_frame, slice_roi_into_digits


def apply_clahe_grayscale(image: np.ndarray, **kwargs) -> np.ndarray: # do i need this? roi_utils has apply_clahe
    """
    Apply CLAHE using grayscale conversion to match OpenCV inference pipeline.
    
    This ensures training uses the exact same CLAHE implementation as inference
    (process_video.py), avoiding discrepancies from Albumentations' LAB-based CLAHE.
    
    Pipeline: RGB -> Grayscale -> CLAHE -> single channel output
    Returns: (H, W, 1) grayscale image for single-channel model input
    """
    from core.config import CLAHE_CLIP_LIMIT, CLAHE_GRID_SIZE

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    enhanced = clahe.apply(grayscale_image)
    
    # Return as (H, W, 1) for albumentations compatibility
    return enhanced[:, :, np.newaxis]

def format_weight(weight_str):
    weight_float = float(weight_str)
    new_weight = f"{weight_float:.3f}"
    new_weight = new_weight.replace('.','')
    if len(new_weight) != 4 or not new_weight.isdigit():
        raise ValueError(f"Incorrect weight format. Got: {new_weight}")
    return new_weight

class ScaleDigitDataset(Dataset):
    def __init__(self, labels_csv, metadata_path='data/metadata.json', images_dir='data/images', file_extension='.jpg', 
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
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"ROI JSON not found: {metadata_path}")

        with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
            metadata = json.load(metadata_file)
        self.rois = metadata.get('rois', {})

        df = pd.read_csv(labels_csv)

        original_count = len(df)
        df = df[pd.to_numeric(df['weight'], errors='coerce').fillna(0) != 0].reset_index(drop=True)

        filtered_count = original_count - len(df)
        if filtered_count > 0 and self.verbose:
            print(f"Filtered out {filtered_count} rows with weight=0")

        self.labels_df = self._build_dataframe(df)

        if len(self.labels_df) == 0:
            raise ValueError("No digit samples available after expanding labels and metadata")

        if self.verbose:
            print(f"Loaded dataset with {len(self.labels_df)} samples")
            print(f"Images directory: {self.images_dir}")
            print(f"Sample labels:\n{self.labels_df.head()}")

        if validate:
            self._validate_dataset()

    def _build_dataframe(self, df):
        df = df.copy()
        df['frame_number'] = pd.to_numeric(df['frame_number'], errors='coerce')
        df = df.dropna(subset=['frame_number'])
        df['frame_number'] = df['frame_number'].astype(int)

        expanded_rows = []
        for row in df.itertuples(index=False):
            filename = row.filename
            frame = row.frame_number
            weight = row.weight
            roi_sections = self.rois.get(filename, {}).get('sections', [])
            roi_info = get_roi_for_frame(frame, roi_sections)

            if roi_info is None:
                continue

            _, dividers = roi_info

            if not isinstance(dividers, list) or len(dividers) != 3:
                continue

            digits = format_weight(weight)
            frame_id = f"{filename}_{frame}"

            for slot_idx, digit in enumerate(digits):
                expanded_rows.append({
                    'filename': filename,
                    'frame_number': frame,
                    'frame_id': frame_id,
                    'weight': weight,
                    'digit_label': digit,
                    'slot_index': slot_idx,
                    'dividers': dividers
                })
        return pd.DataFrame(expanded_rows)

    def _validate_dataset(self):
        if self.verbose:
            print("Validating dataset...")
        
        missing_files = []
        invalid_crops = []

        for row in self.labels_df.itertuples(index=True):
            i = row.index
            filename = f"{row.filename}_{row.frame_number}{self.file_extension}"
            img_path = self.images_dir / filename

            if not img_path.exists():
                missing_files.append((i, filename))
                continue
            
            image = cv2.imread(str(img_path))
            if image is None:
                invalid_crops.append((i, filename, 'image unreadable'))
                continue

            digit_crops = slice_roi_into_digits(image, row.dividers)
            if digit_crops is None or len(digit_crops) != 4:
                invalid_crops.append((i, filename, 'could not slice into four digits'))
                continue

            slot_idx = int(row.slot_index)
            if slot_idx >= len(digit_crops) or digit_crops[slot_idx] == 0:
                invalid_crops.append((i, filename, f'invalid slot {slot_idx}'))

        if missing_files or invalid_crops:
            bad_indices = {i for i, _ in missing_files}
            bad_indices.update(i for i,_,_ in invalid_crops)
            self.labels_dfdf = self.labels_df.drop(index=list(bad_indices)).reset_index(drop=True)

        if len(self.labels_df) <= 0:
            raise ValueError('No samples remain after validation')
        if self.verbose:
            if missing_files:
                print(f"Removed {len(missing_files)} samples with missing images")
                print(f"Removed {len(invalid_crops)} samples with invalid digit crops")
                print(f"Validation complete. {len(self.labels_df)} samples remain")
    def __len__(self):
        return len(self.labels_df)
    def __getitem__(self, index):
        row = self.labels_df.iloc[index]
        frame_num = row['frame_number']
        filename = f"{row['filename']}_{frame_num}{self.file_extension}"

        img_path = self.images_dir / filename
        image = cv2.imread(str(img_path))

        if image is None:
            raise ValueError(f"Could not load images from {img_path}")

        digit_crops = slice_roi_into_digits(image, row['dividers'])
        if digit_crops is None:
            raise ValueError(f"Could not split ROI image into digit crops for {img_path}")

        slot_index = int(row['slot_index'])
        image = digit_crops[slot_index]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, int(row['digit_label']), row['frame_id'], slot_index
    
def digit_collate_fn(batch):
    """
    Note: This function was written by Claude Opus 4.6
    Collate digit-level samples into a padded batch.

    Each sample is expected to be a tuple:
    (image, digit_label, frame_id, slot_index)

    Images may have different widths after digit cropping. This function converts
    all images to CHW tensors (if needed), right-pads each image to the widest
    image in the batch with zeros, and stacks them.

    Returns:
        tuple[
            torch.Tensor,      # (B, C, H, W_max)
            torch.LongTensor,  # (B,) digit labels in [0..9]
            list[str],         # (B,) frame ids, e.g. "video.mp4_123"
            torch.LongTensor,  # (B,) slot indices in [0..3]
        ]
    """
    images, labels, frame_ids, slot_idxs = zip(*batch)

    tensor_images = []
    for image in images:
        if isinstance(image, torch.Tensor):
            tensor_image = image
        else:
            image_array = np.asarray(image)
            if image_array.ndim == 2:
                tensor_image = torch.from_numpy(image_array).unsqueeze(0)
            else:
                tensor_image = torch.from_numpy(np.transpose(image_array, (2, 0, 1)))
        tensor_images.append(tensor_image)

    max_width = max(image.shape[-1] for image in tensor_images)
    padded_images = [
        F.pad(image, (0, max_width - image.shape[-1], 0, 0), value=0)
        for image in tensor_images
    ]

    return (
        torch.stack(padded_images),
        torch.tensor(labels, dtype=torch.long),
        list(frame_ids),
        torch.tensor(slot_idxs, dtype=torch.long),
    )

def get_transforms(image_size=None, is_train=False):
    if image_size is None:
        image_size = IMAGE_SIZE
    target_width, target_height = image_size
    
    if is_train:
        # augmentation + resizing w padding
        # CLAHE applied via custom function to match OpenCV inference pipeline
        return A.Compose([ # type: ignore
            # CLAHE using grayscale (matches inference preprocessing exactly)
            A.Lambda(image=apply_clahe_grayscale, p=1.0),
            
            # augmentation
            A.Affine(scale=(0.85, 1.15), 
                    translate_percent=0.05,
                    rotate=5,
                    fill=0,
                    p=0.3
            ),
                    
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.Sharpen(alpha=(0.2, 0.5), p=0.3),
            
            # Single channel grayscale normalization
            A.Normalize(mean=(0.5,), std=(0.5,)),
            
            ToTensorV2(),
        ])
    else: 
        return A.Compose([
            # CLAHE using grayscale (matches inference preprocessing exactly)
            A.Lambda(image=apply_clahe_grayscale, p=1.0),
            
            A.LongestMaxSize(max_size=max(target_width, target_height)),
            A.PadIfNeeded(
                min_height=target_height,
                min_width=target_width,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                position='center'
            ),
            A.CenterCrop(height=target_height, width=target_width),
            # Single channel grayscale normalization
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
def create_dataloaders(
    data_dir: str,
    metadata_json: str = 'data/metadata.json',
    batch_size: int = 16,
    image_size: Optional[tuple] = None,
    num_workers: int = 2,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    verbose: bool = True
):
    if image_size is None:
        from core.config import IMAGE_SIZE
        image_size = IMAGE_SIZE

    if verbose:
        print("Creating dataloaders...")

    train_transform = get_transforms(image_size, is_train=True)
    val_transform = get_transforms(image_size, is_train=False)

    images_dir = Path(data_dir) / 'images'
    labels_dir = Path(data_dir) / 'labels'
    train_csv = labels_dir / 'train_labels.csv'
    val_csv = labels_dir / 'val_labels.csv'
    test_csv = labels_dir / 'test_labels.csv'

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not Path(metadata_json).exists():
        raise FileNotFoundError(f"Metadata JSON not found: {metadata_json}")
    if not train_csv.exists():
        raise FileNotFoundError(f"Train labels CSV not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Val labels CSV not found: {val_csv}")

    train_dataset = ScaleDigitDataset(str(train_csv), metadata_path=str(metadata_json), images_dir=str(images_dir), transform=train_transform, validate=True, verbose=verbose)
    val_dataset = ScaleDigitDataset(str(val_csv), metadata_path=str(metadata_json), images_dir=str(images_dir), transform=val_transform, validate=True, verbose=verbose)

    test_dataset = None
    if test_csv.exists():
        test_dataset = ScaleDigitDataset(str(test_csv), metadata_path=str(metadata_json), images_dir=str(images_dir), transform=val_transform, validate=True, verbose=verbose)

    
    create_dataloader = partial(
        DataLoader, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn = digit_collate_fn
    )

    train_loader = create_dataloader(train_dataset, shuffle=True)
    val_loader = create_dataloader(val_dataset)
    test_loader = None
    if test_dataset:
        test_loader = create_dataloader(test_dataset)

    if verbose:
        print(f"Dataloaders created:")
        print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
        print(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
        if test_loader:
            test_samples = len(test_dataset) if test_dataset is not None else 0
            print(f"  Test: {len(test_loader)} batches ({test_samples} samples)")

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


# for testing purposes:  
if __name__ == "__main__":
    print("="*60)
    print("TESTING DATASET WITH TRANSFORMS")
    print("="*60 + "\n")
    
    train_loader, val_loader, test_loader, _, _, _ = create_dataloaders(
        data_dir="data",
        batch_size=8,
    )
    
    images, labels, frame_ids, slot_idxs = next(iter(train_loader))
    
    print(f"\nBatch shapes after transforms:")
    print(f"  Image dims: {images.shape}")
    print(f"  First 3 labels: {labels[:3].tolist()}")
    print(f"  First 3 frame ids: {frame_ids[:3]}")
    print(f"  First 3 slot indices: {slot_idxs[:3].tolist()}")
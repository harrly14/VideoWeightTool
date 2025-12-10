import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_dataset(csv_path, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split dataset into train, validation, and test sets.
    """
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train+n_val]
    test_df = df.iloc[n_train+n_val:]
    
    print(f"Total samples: {n}")
    print(f"Train: {len(train_df)} ({len(train_df)/n:.1%})")
    print(f"Val:   {len(val_df)} ({len(val_df)/n:.1%})")
    print(f"Test:  {len(test_df)} ({len(test_df)/n:.1%})")
    
    # Save
    train_df.to_csv(output_path / 'train_labels.csv', index=False)
    val_df.to_csv(output_path / 'val_labels.csv', index=False)
    test_df.to_csv(output_path / 'test_labels.csv', index=False)
    
    print(f"Saved splits to {output_path}")

if __name__ == "__main__":
    split_dataset(
        csv_path='data/all_data.csv',
        output_dir='data/labels'
    )

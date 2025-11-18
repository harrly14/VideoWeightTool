import pandas as pd

val_df = pd.read_csv('data/split_frame/val_labels.csv')
print(f"Val unique weights: {val_df['weight'].nunique()}")
print(f"Val weight range: {val_df['weight'].min()} - {val_df['weight'].max()}")
print(val_df['weight'].value_counts().head(10))
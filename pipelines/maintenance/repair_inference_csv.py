"""
Repair and post-process inference CSV output.

This script performs two optional passes on flagged frames from inference output:
  1. Hold-Last Repair (always performed):
     - Identifies contiguous flagged spans
     - For interior spans with agreeing boundaries: confirms hold-last behavior
     - For interior spans with disagreeing boundaries: linearly interpolates by row index
     - For trailing spans (no right boundary): marks as trailing without interpolation
  
  2. Rolling Median (optional, --no-median to skip):
     - Applies configurable rolling median over repaired smoothed values
     - Window default: 10 frames (representing 10 seconds at 1fps)

Output includes original columns plus:
  - repair_reason: describes the repair action taken on each flagged frame
  - repaired_smoothed_weight: repaired value (original smoothed_weight if unflagged)

Note: The --summary flag prints a diagnostic breakdown of flag_reason counts from input
before any repair operations.

Hard-coded context: Leading-digit correction (1.xxx -> 7.xxx) and domain range filtering
(6.0-8.0 kg) are temporary workarounds for this specific training/output dataset.
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import medfilt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def print_flag_reason_summary(df):
    print("\n" + "=" * 70)
    print("FLAG_REASON SUMMARY (Input CSV)")
    print("=" * 70)
    
    total_rows = len(df)
    flagged_rows = df['flag_reason'].notna().sum()
    unflagged_rows = total_rows - flagged_rows
    
    print(f"Total rows: {total_rows}")
    print(f"Unflagged (valid): {unflagged_rows} ({unflagged_rows/total_rows*100:.1f}%)")
    print(f"Flagged for review: {flagged_rows} ({flagged_rows/total_rows*100:.1f}%)")
    
    if flagged_rows > 0:
        print(f"\nBreakdown by flag_reason:")
        reason_counts = df[df['flag_reason'].notna()]['flag_reason'].value_counts()
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count} ({count/flagged_rows*100:.1f}% of flagged)")
    
    print()


def find_flagged_spans(df):
    """Find contiguous spans of flagged rows (excluding 'corrected' rows).
    
    Returns: list of (start_idx, end_idx) tuples (inclusive indices)
    """
    spans = []
    in_span = False
    span_start = None
    
    for i, row in df.iterrows():
        flag_reason = row.get('flag_reason')
        is_flagged = flag_reason is not None and flag_reason != 'corrected' and pd.notna(flag_reason)
        
        if is_flagged:
            if not in_span:
                in_span = True
                span_start = i
        else:
            if in_span:
                spans.append((span_start, i - 1))
                in_span = False
    
    # Handle trailing span
    if in_span:
        spans.append((span_start, len(df) - 1))
    
    return spans


def repair_with_span_analysis(df, tolerance=0.100):
    """Apply hold-last repair validation using span boundary analysis.
    
    For each contiguous flagged span:
      - Interior spans with agreeing boundaries: keep hold-last (repair_reason = 'hold_last_confirmed')
      - Interior spans with disagreeing boundaries: interpolate (repair_reason = 'interpolated')
      - Trailing spans: no interpolation (repair_reason = 'trailing_span')
    
    Unflagged and 'corrected' rows pass through unchanged.
    
    Args:
        df: DataFrame with columns ['frame_num', 'smoothed_weight', 'flag_reason', ...]
        tolerance: Absolute tolerance for boundary agreement
    
    Returns:
        DataFrame with added 'repair_reason' and 'repaired_smoothed_weight' columns
    """
    df = df.copy()
    
    # Initialize repair columns
    df['repair_reason'] = None
    df['repaired_smoothed_weight'] = df['smoothed_weight'].copy()
    
    spans = find_flagged_spans(df)
    
    for span_start, span_end in spans:
        has_left_boundary = span_start > 0
        has_right_boundary = span_end < len(df) - 1
        
        if has_left_boundary and has_right_boundary:
            left_boundary_value = df.loc[span_start - 1, 'smoothed_weight']
            right_boundary_value = df.loc[span_end + 1, 'smoothed_weight']
            
            if pd.notna(left_boundary_value) and pd.notna(right_boundary_value):
                boundary_diff = abs(left_boundary_value - right_boundary_value)
                
                if boundary_diff <= tolerance:
                    # Boundaries agree: keep hold-last
                    for i in range(span_start, span_end + 1):
                        df.loc[i, 'repair_reason'] = 'hold_last_confirmed'
                else:
                    # Boundaries disagree: interpolate
                    num_points = span_end - span_start + 1
                    interpolated_values = np.linspace(
                        left_boundary_value, 
                        right_boundary_value, 
                        num_points + 2
                    )[1:-1]  # Exclude boundaries, keep interior points
                    
                    for j, i in enumerate(range(span_start, span_end + 1)):
                        df.loc[i, 'repaired_smoothed_weight'] = interpolated_values[j]
                        df.loc[i, 'repair_reason'] = 'interpolated'
            else:
                # Can't compare boundaries (e.g., NaN values): mark as hold_last
                for i in range(span_start, span_end + 1):
                    df.loc[i, 'repair_reason'] = 'hold_last_confirmed'
        else:
            # Trailing span (no right boundary) or leading span: mark but don't interpolate
            for i in range(span_start, span_end + 1):
                df.loc[i, 'repair_reason'] = 'trailing_span'
    
    # For unflagged and 'corrected' rows, mark repair_reason as None (no repair applied)
    for i, row in df.iterrows():
        flag_reason = row.get('flag_reason')
        if flag_reason is None or flag_reason == 'corrected' or pd.isna(flag_reason):
            df.loc[i, 'repair_reason'] = None
    
    return df


def apply_rolling_median(df, window=10):
    """Apply rolling median over repaired smoothed weights.
    
    Args:
        df: DataFrame with 'repaired_smoothed_weight' column
        window: Window size in frames
    
    Returns:
        DataFrame with additional 'median_smoothed_weight' column
    """
    df = df.copy()
    
    values = df['repaired_smoothed_weight'].values.copy()
    
    valid_mask = ~np.isnan(values)
    
    if valid_mask.sum() == 0:
        df['median_smoothed_weight'] = np.nan
        return df
    
    # Apply median filter only to valid values
    # scipy.medfilt requires odd kernel size
    kernel_size = window if window % 2 == 1 else window + 1
    
    filtered = medfilt(values[valid_mask], kernel_size=kernel_size)
    
    # Reconstruct full array with NaN padding
    result = np.full_like(values, np.nan)
    result[valid_mask] = filtered
    
    df['median_smoothed_weight'] = result
    
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repair and post-process inference CSV output with span-aware hold/interpolation and optional rolling median.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--input', type=str, required=True, help='Path to input inference CSV')
    parser.add_argument('--output', type=str, required=True, help='Path to output repaired CSV')
    parser.add_argument('--tolerance', type=float, default=0.100, 
                        help='Absolute tolerance (kg) for boundary agreement when deciding hold vs interpolate (default: 0.100)')
    parser.add_argument('--window', type=int, default=10, 
                        help='Rolling median window size in frames (default: 10, ~10 seconds at 1fps)')
    parser.add_argument('--no-median', action='store_true', 
                        help='Skip rolling median pass, output only hold-last repair')
    parser.add_argument('--summary', action='store_true', 
                        help='Print flag_reason breakdown from input CSV before repair')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"\nLoading input CSV: {args.input}")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}")
        return 1
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return 1
    
    print(f"Loaded {len(df)} rows")
    
    if args.summary:
        print_flag_reason_summary(df)
    
    # Pass 1: Hold-Last Repair
    print(f"Pass 1: Hold-Last Repair Analysis")
    print(f"  Boundary tolerance: {args.tolerance} kg")
    df = repair_with_span_analysis(df, tolerance=args.tolerance)
    
    # Count repair outcomes
    repair_counts = df['repair_reason'].value_counts(dropna=False)
    print(f"\nRepair outcomes:")
    for reason, count in repair_counts.items():
        reason_str = reason if reason else 'Not repaired (unflagged or corrected)'
        print(f"  {reason_str}: {count}")
    
    # Pass 2: Rolling Median (optional)
    if not args.no_median:
        print(f"\nPass 2: Rolling Median Filter")
        print(f"  Window size: {args.window} frames")
        df = apply_rolling_median(df, window=args.window)
        print(f"  Applied rolling median to repaired values")
        
        # Use median-filtered values as final output
        df['repaired_smoothed_weight'] = df['median_smoothed_weight'].fillna(df['repaired_smoothed_weight'])
        df = df.drop(columns=['median_smoothed_weight'])
    else:
        print(f"\nPass 2: Skipped (--no-median)")
    

    print(f"\nWriting output CSV: {args.output}")
    try:
        # Select columns for output: original columns + repair_reason + repaired_smoothed_weight
        output_cols = [col for col in df.columns if col != 'repaired_smoothed_weight'] + ['repaired_smoothed_weight']
        df[output_cols].to_csv(args.output, index=False)
        print(f"Repaired CSV written: {args.output}")
    except Exception as e:
        print(f"Error writing output CSV: {e}")
        return 1
    
    print("\n" + "=" * 70)
    print("REPAIR COMPLETE")
    print("=" * 70)
    print(f"Output columns include:")
    print(f"  - Original columns from input")
    print(f"  - repair_reason: describes repair action (hold_last_confirmed, interpolated, trailing_span)")
    print(f"  - repaired_smoothed_weight: final smoothed value after repair (and optional median)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

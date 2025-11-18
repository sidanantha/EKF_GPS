import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def clean_duplicate_timestamps(csv_file_path, output_path=None):
    """
    Scan through CSV data, find duplicate data points at the same timestamp,
    and average them together into a single datapoint.
    
    Args:
        csv_file_path: Path to the CSV file to clean
        output_path: Path to save the cleaned CSV. If None, overwrites the original file.
    
    Returns:
        DataFrame with cleaned data
    """
    
    # Try reading with skiprows=1 first (for files with PuTTY log header)
    df = pd.read_csv(csv_file_path, skiprows=1)
    
    # If UTC column is missing, try reading without skipping rows
    if 'UTC' not in df.columns:
        df = pd.read_csv(csv_file_path)
    
    # If still no UTC column, raise an error
    if 'UTC' not in df.columns:
        raise ValueError(f"UTC column not found in {csv_file_path}. Available columns: {df.columns.tolist()}")
    
    # Get original row count
    original_count = len(df)
    print(f"Original data points: {original_count}")
    
    # Group by UTC timestamp
    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Group by UTC and aggregate
    # For numeric columns, take the mean; for non-numeric, take the first value
    agg_dict = {}
    for col in df.columns:
        if col in numeric_cols:
            agg_dict[col] = 'mean'
        else:
            agg_dict[col] = 'first'
    
    # Reset index so UTC becomes a regular column
    cleaned_df = df.groupby('UTC', as_index=False).agg(agg_dict)
    
    # Get cleaned row count
    cleaned_count = len(cleaned_df)
    duplicates_removed = original_count - cleaned_count
    
    print(f"Cleaned data points: {cleaned_count}")
    print(f"Duplicate entries removed: {duplicates_removed}")
    
    # Determine output path
    if output_path is None:
        output_path = csv_file_path
    
    # Save cleaned data back to CSV (without index)
    cleaned_df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to: {output_path}")
    
    return cleaned_df


def clean_all_csv_files(field_data_dir, output_suffix='_cleaned'):
    """
    Clean all CSV files in the field_data directory.
    Saves cleaned files with optional suffix (default: _cleaned).
    
    Args:
        field_data_dir: Path to the field_data directory
        output_suffix: Suffix to add to cleaned files. If None, overwrites original files.
    """
    field_data_path = Path(field_data_dir)
    csv_files = sorted(field_data_path.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {field_data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s)\n")
    
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"Processing: {csv_file.name}")
        print('='*60)
        
        try:
            if output_suffix:
                # Save with suffix
                output_path = csv_file.parent / f"{csv_file.stem}{output_suffix}.csv"
            else:
                # Overwrite original
                output_path = csv_file
            
            clean_duplicate_timestamps(str(csv_file), str(output_path))
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Clean duplicate timestamps in field data CSV files.'
    )
    parser.add_argument(
        'file',
        nargs='?',
        default=None,
        help='Specific CSV file to clean. If not provided, cleans all CSV files in field_data/'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Clean ALL CSV files in field_data/'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output file path (default: overwrites original or uses _cleaned suffix)'
    )
    parser.add_argument(
        '--inplace',
        action='store_true',
        help='Overwrite original files instead of creating _cleaned versions'
    )
    
    args = parser.parse_args()
    
    # Get the path to the field_data directory
    current_dir = Path(__file__).parent.parent
    field_data_dir = current_dir / 'field_data'
    
    if args.all:
        # Clean all files
        output_suffix = None if args.inplace else '_cleaned'
        clean_all_csv_files(str(field_data_dir), output_suffix)
    elif args.file:
        # Clean specific file
        file_path = Path(args.file)
        
        # If relative path, resolve relative to field_data_dir
        if not file_path.is_absolute():
            if not file_path.exists():
                alt_path = field_data_dir / file_path.name
                if alt_path.exists():
                    file_path = alt_path
        
        if file_path.exists():
            print(f"Processing: {file_path.name}\n")
            try:
                output_path = args.output if args.output else file_path
                clean_duplicate_timestamps(str(file_path), str(output_path))
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
        else:
            print(f"Error: File not found: {file_path}")
    else:
        # Show help if no arguments
        parser.print_help()


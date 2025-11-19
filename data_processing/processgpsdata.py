import numpy as np
import pandas as pd
from pathlib import Path
import argparse

#remove blanks
def remove_blanks(df):
    df_blanks = df.replace(['', ' '], np.nan)
    df_cleaned = df_blanks.dropna().reset_index(drop=True)
  
    return df_cleaned

#since at 10 Hz, smooth by 5 samples
def average(pos, num):
    return np.convolve(poss, np.ones(num)/num, mode='same')

#remove outliers
def remove_outliers(df, num = 5, thres = 3):
    x = df['ECEF_X'].to_numpy()
    y = df['ECEF_Y'].to_numpy()
    z = df['ECEF_Z'].to_numpy()
    #calculate residuals
    res_x = x - average(x, num)
    res_y = y - average(y, num)
    res_z = z - average(z, num)
    #calculate standard deviations
    sigma_x = np.std(res_x)
    sigma_y = np.std(res_y)
    sigma_z = np.std(res_z)

    #apply mask and remove outliers
    df_cleaned = df[(np.abs(res_x) <= thres * sigma_x) & (np.abs(res_y) <= thres * sigma_y) & (np.abs(res_z) <= thres * sigma_z)].reset_index(drop=True)
    
    return df_cleaned

#main function to process gps data before ekf
def process_gps_data(csv_file_path, output_path = None):
    #read in csv data
    df = pd.read_csv(csv_file_path, skiprows=1)

    #remove blanks
    df_cleaned = remove_blanks(df)

    #smooth data and remove gps outliers greater than 3 sigma
    df_cleaned = remove_outliers(df_cleaned)

    if output_path is None:
        output_path = csv_file_path

    df_cleaned.to_csv(output_path, index=False)

#calculate sigma for R matrix
def calculate_sigma_gps(df):
    x = df['ECEF_X'].to_numpy()
    y = df['ECEF_Y'].to_numpy()
    z = df['ECEF_Z'].to_numpy()

    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_z = np.std(z)

    sigma_gps = np.mean([sigma_x, sigma_y, sigma_z])

    return sigma_gps

#same as cleandata.py
def process_all_gps_files(field_data_dir, output_suffix='_gpscleaned'):
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
            
            process_gps_data(str(csv_file), str(output_path))
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prep GPS Data in field data CSV files.'
    )
    parser.add_argument(
        'file',
        nargs='?',
        default=None,
        help='Specific CSV file to access. If not provided, accesses all CSV files in field_data/'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Prep ALL CSV files in field_data/'
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
        output_suffix = None if args.inplace else '_gpscleaned'
        process_all_gps_files(str(field_data_dir), output_suffix)
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
                process_gps_data(str(file_path), str(output_path))
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
        else:
            print(f"Error: File not found: {file_path}")
    else:
        # Show help if no arguments
        parser.print_help()

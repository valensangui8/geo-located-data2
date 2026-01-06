"""
Data Loader Module
==================

Load and parse the Flickr geo-located data CSV file.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


# Expected column names after cleaning
EXPECTED_COLUMNS = [
    'id', 'user', 'lat', 'long', 'tags', 'title',
    'date_taken_minute', 'date_taken_hour', 'date_taken_day',
    'date_taken_month', 'date_taken_year',
    'date_upload_minute', 'date_upload_hour', 'date_upload_day',
    'date_upload_month', 'date_upload_year'
]


def load_flickr_data(filepath: str | Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load Flickr geo-located data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        nrows: Optional number of rows to load (for testing/sampling)
        
    Returns:
        DataFrame with cleaned column names and proper types
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Load CSV with proper handling of trailing columns
    df = pd.read_csv(
        filepath,
        nrows=nrows,
        low_memory=False,
        skipinitialspace=True
    )
    
    # Clean column names (remove whitespace, handle trailing empty columns)
    df.columns = df.columns.str.strip()
    
    # Remove unnamed/empty columns (trailing columns from CSV)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Keep only expected columns that exist
    existing_cols = [col for col in EXPECTED_COLUMNS if col in df.columns]
    df = df[existing_cols]
    
    # Convert numeric columns
    numeric_cols = ['lat', 'long', 'date_taken_minute', 'date_taken_hour',
                    'date_taken_day', 'date_taken_month', 'date_taken_year',
                    'date_upload_minute', 'date_upload_hour', 'date_upload_day',
                    'date_upload_month', 'date_upload_year']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill NaN in text columns with empty string
    text_cols = ['tags', 'title']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')
    
    # Convert id and user to string (they are identifiers, not numeric values)
    if 'id' in df.columns:
        df['id'] = df['id'].astype(str)
    if 'user' in df.columns:
        df['user'] = df['user'].astype(str)
    
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the loaded dataset.
    
    Args:
        df: Loaded DataFrame
        
    Returns:
        Dictionary with dataset statistics
    """
    info = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': list(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    
    # Coordinate ranges
    if 'lat' in df.columns and 'long' in df.columns:
        info['lat_range'] = (df['lat'].min(), df['lat'].max())
        info['long_range'] = (df['long'].min(), df['long'].max())
    
    # Date ranges
    if 'date_taken_year' in df.columns:
        info['year_range'] = (
            int(df['date_taken_year'].min()) if pd.notna(df['date_taken_year'].min()) else None,
            int(df['date_taken_year'].max()) if pd.notna(df['date_taken_year'].max()) else None
        )
    
    return info


def print_data_info(info: dict) -> None:
    """Print formatted data information."""
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Total rows: {info['total_rows']:,}")
    print(f"Total columns: {info['total_columns']}")
    print(f"Memory usage: {info['memory_usage_mb']:.2f} MB")
    print()
    
    print("Columns:", ", ".join(info['columns']))
    print()
    
    if 'lat_range' in info:
        print(f"Latitude range: {info['lat_range'][0]:.4f} to {info['lat_range'][1]:.4f}")
        print(f"Longitude range: {info['long_range'][0]:.4f} to {info['long_range'][1]:.4f}")
    
    if 'year_range' in info:
        print(f"Year range: {info['year_range'][0]} to {info['year_range'][1]}")
    
    print()
    print("Missing values per column:")
    for col, missing in info['missing_values'].items():
        if missing > 0:
            pct = (missing / info['total_rows']) * 100
            print(f"  {col}: {missing:,} ({pct:.1f}%)")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test loading
    data_path = Path(__file__).parent.parent / "data" / "flickr_data2.csv"
    
    print("Loading sample data (first 1000 rows)...")
    df = load_flickr_data(data_path, nrows=1000)
    info = get_data_info(df)
    print_data_info(info)
    
    print("\nSample rows:")
    print(df.head())


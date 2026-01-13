import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


LYON_BOUNDS = {
    'lat_min': 45.7,
    'lat_max': 45.85,
    'long_min': 4.75,
    'long_max': 4.95
}

VALID_YEAR_MIN = 2000
VALID_YEAR_MAX = 2025

VALIDATION_RULES = {
    'lat': {'type': 'float', 'min': -90.0, 'max': 90.0},
    'long': {'type': 'float', 'min': -180.0, 'max': 180.0},
    'date_taken_minute': {'type': 'int', 'min': 0, 'max': 59},
    'date_taken_hour': {'type': 'int', 'min': 0, 'max': 23},
    'date_taken_day': {'type': 'int', 'min': 1, 'max': 31},
    'date_taken_month': {'type': 'int', 'min': 1, 'max': 12},
    'date_taken_year': {'type': 'int', 'min': VALID_YEAR_MIN, 'max': VALID_YEAR_MAX},
    'date_upload_minute': {'type': 'int', 'min': 0, 'max': 59},
    'date_upload_hour': {'type': 'int', 'min': 0, 'max': 23},
    'date_upload_day': {'type': 'int', 'min': 1, 'max': 31},
    'date_upload_month': {'type': 'int', 'min': 1, 'max': 12},
    'date_upload_year': {'type': 'int', 'min': VALID_YEAR_MIN, 'max': VALID_YEAR_MAX},
    'id': {'type': 'string', 'non_empty': True},
    'user': {'type': 'string', 'non_empty': True},
    'tags': {'type': 'string', 'non_empty': False},
    'title': {'type': 'string', 'non_empty': False},
}

REQUIRED_COLUMNS = [
    'id', 'user', 'lat', 'long',
    'date_taken_minute', 'date_taken_hour', 'date_taken_day',
    'date_taken_month', 'date_taken_year',
    'date_upload_minute', 'date_upload_hour', 'date_upload_day',
    'date_upload_month', 'date_upload_year',
]


class CleaningReport:
    
    def __init__(self, initial_count: int):
        self.initial_count = initial_count
        self.operations = []
        self.current_count = initial_count
    
    def add_operation(self, name: str, removed: int, details: str = ""):
        self.operations.append({
            'name': name,
            'removed': removed,
            'remaining': self.current_count - removed,
            'details': details
        })
        self.current_count -= removed
    
    def print_report(self):
        print("\n" + "=" * 70)
        print("DATA CLEANING REPORT")
        print("=" * 70)
        print(f"Initial row count: {self.initial_count:,}")
        print()
        
        total_removed = 0
        for op in self.operations:
            total_removed += op['removed']
            pct = 100 * op['removed'] / self.initial_count
            print(f"  {op['name']}:")
            print(f"    Removed: {op['removed']:,} ({pct:.2f}%)")
            if op['details']:
                print(f"    Details: {op['details']}")
            print(f"    Remaining: {op['remaining']:,}")
            print()
        
        print("-" * 70)
        print(f"Total removed: {total_removed:,} ({100*total_removed/self.initial_count:.2f}%)")
        print(f"Final row count: {self.current_count:,}")
        print("=" * 70)


def validate_column_types(df: pd.DataFrame, 
                          rules: dict = VALIDATION_RULES,
                          report: Optional[CleaningReport] = None) -> Tuple[pd.DataFrame, dict]:
    before = len(df)
    df = df.copy()
    
    validation_details = {
        'columns_validated': [],
        'invalid_counts': {},
        'total_invalid_rows': 0
    }
    
    valid_mask = pd.Series([True] * len(df), index=df.index)
    
    for col, rule in rules.items():
        if col not in df.columns:
            continue
            
        validation_details['columns_validated'].append(col)
        col_invalid_count = 0
        col_type = rule.get('type')
        
        if col_type == 'float':
            numeric_mask = pd.to_numeric(df[col], errors='coerce').notna()
            if 'min' in rule and 'max' in rule:
                range_mask = (df[col] >= rule['min']) & (df[col] <= rule['max'])
                col_valid = numeric_mask & range_mask
            else:
                col_valid = numeric_mask
            col_invalid_count = (~col_valid).sum()
            valid_mask &= col_valid
            
        elif col_type == 'int':
            numeric_values = pd.to_numeric(df[col], errors='coerce')
            numeric_mask = numeric_values.notna()
            is_integer_mask = (numeric_values == numeric_values.astype(int, errors='ignore'))
            if 'min' in rule and 'max' in rule:
                range_mask = (numeric_values >= rule['min']) & (numeric_values <= rule['max'])
                col_valid = numeric_mask & is_integer_mask & range_mask
            else:
                col_valid = numeric_mask & is_integer_mask
            col_valid = col_valid.fillna(False)
            col_invalid_count = (~col_valid).sum()
            valid_mask &= col_valid
            
        elif col_type == 'string':
            if rule.get('non_empty', False):
                col_valid = df[col].notna() & (df[col].astype(str).str.strip() != '')
                col_invalid_count = (~col_valid).sum()
                valid_mask &= col_valid
        
        if col_invalid_count > 0:
            validation_details['invalid_counts'][col] = col_invalid_count
    
    df = df[valid_mask]
    after = len(df)
    removed = before - after
    validation_details['total_invalid_rows'] = removed
    
    if report:
        details_parts = []
        for col, count in validation_details['invalid_counts'].items():
            rule = rules.get(col, {})
            if 'min' in rule and 'max' in rule:
                details_parts.append(f"{col}: {count:,} invalid (expected {rule['min']}-{rule['max']})")
            else:
                details_parts.append(f"{col}: {count:,} invalid")
        details_str = "; ".join(details_parts) if details_parts else "All values valid"
        report.add_operation("Validate column data types and ranges", removed, details_str)
    
    return df, validation_details


def print_validation_summary(validation_details: dict) -> None:
    print("\n" + "-" * 50)
    print("VALIDATION DETAILS")
    print("-" * 50)
    print(f"Columns validated: {len(validation_details['columns_validated'])}")
    print(f"Total invalid rows removed: {validation_details['total_invalid_rows']:,}")
    
    if validation_details['invalid_counts']:
        print("\nInvalid values per column:")
        for col, count in sorted(validation_details['invalid_counts'].items(), 
                                  key=lambda x: x[1], reverse=True):
            rule = VALIDATION_RULES.get(col, {})
            if 'min' in rule and 'max' in rule:
                print(f"  {col}: {count:,} (expected range: {rule['min']}-{rule['max']})")
            else:
                print(f"  {col}: {count:,}")
    else:
        print("\nAll column values are within expected ranges!")
    print("-" * 50)


def remove_rows_with_nulls(df: pd.DataFrame, 
                           required_cols: list = REQUIRED_COLUMNS,
                           report: Optional[CleaningReport] = None) -> pd.DataFrame:
    before = len(df)
    cols_to_check = [col for col in required_cols if col in df.columns]
    
    null_counts = {}
    for col in cols_to_check:
        null_count = df[col].isna().sum()
        if null_count > 0:
            null_counts[col] = null_count
    
    df = df.dropna(subset=cols_to_check)
    after = len(df)
    removed = before - after
    
    if report:
        if null_counts:
            details_parts = [f"{col}: {count:,} nulls" for col, count in null_counts.items()]
            details = "Removed rows with nulls in: " + "; ".join(details_parts)
        else:
            details = "No null values found in required columns"
        report.add_operation("Remove rows with null values", removed, details)
    
    return df


def filter_lyon_bounds(df: pd.DataFrame, 
                       bounds: dict = LYON_BOUNDS,
                       report: Optional[CleaningReport] = None) -> pd.DataFrame:
    before = len(df)
    mask = (
        (df['lat'] >= bounds['lat_min']) & 
        (df['lat'] <= bounds['lat_max']) &
        (df['long'] >= bounds['long_min']) & 
        (df['long'] <= bounds['long_max'])
    )
    df = df[mask]
    after = len(df)
    removed = before - after
    
    if report:
        report.add_operation("Filter to Lyon bounding box", removed,
            f"Kept only lat [{bounds['lat_min']}, {bounds['lat_max']}], "
            f"long [{bounds['long_min']}, {bounds['long_max']}]")
    return df


def remove_duplicate_ids(df: pd.DataFrame, report: Optional[CleaningReport] = None) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=['id'], keep='first')
    after = len(df)
    removed = before - after
    
    if report:
        report.add_operation("Remove duplicate photo IDs", removed,
                            "Kept first occurrence of each duplicate ID")
    return df


def remove_duplicate_locations(df: pd.DataFrame, 
                               precision: int = 4,
                               report: Optional[CleaningReport] = None) -> pd.DataFrame:
    before = len(df)
    df = df.copy()
    df['_lat_round'] = df['lat'].round(precision)
    df['_long_round'] = df['long'].round(precision)
    
    subset_cols = ['user', '_lat_round', '_long_round', 
                   'date_taken_year', 'date_taken_month', 'date_taken_day']
    subset_cols = [c for c in subset_cols if c in df.columns]
    
    df = df.drop_duplicates(subset=subset_cols, keep='first')
    df = df.drop(columns=['_lat_round', '_long_round'])
    
    after = len(df)
    removed = before - after
    
    if report:
        report.add_operation("Remove location duplicates", removed,
            f"Removed same user/location/day (precision: {precision} decimals)")
    return df


def filter_valid_dates(df: pd.DataFrame,
                       year_min: int = VALID_YEAR_MIN,
                       year_max: int = VALID_YEAR_MAX,
                       report: Optional[CleaningReport] = None) -> pd.DataFrame:
    before = len(df)
    
    if 'date_taken_year' in df.columns:
        df = df[(df['date_taken_year'] >= year_min) & (df['date_taken_year'] <= year_max)]
    if 'date_taken_month' in df.columns:
        df = df[(df['date_taken_month'] >= 1) & (df['date_taken_month'] <= 12)]
    if 'date_taken_day' in df.columns:
        df = df[(df['date_taken_day'] >= 1) & (df['date_taken_day'] <= 31)]
    if 'date_taken_hour' in df.columns:
        df = df[(df['date_taken_hour'] >= 0) & (df['date_taken_hour'] <= 23)]
    
    after = len(df)
    removed = before - after
    
    if report:
        report.add_operation("Filter valid dates", removed,
            f"Kept years [{year_min}, {year_max}], valid month/day/hour ranges")
    return df


def normalize_tags(df: pd.DataFrame) -> pd.DataFrame:
    if 'tags' not in df.columns:
        return df
    
    df = df.copy()
    
    def normalize_tag_string(tags_str):
        if pd.isna(tags_str) or tags_str == '':
            return ''
        tags = [t.strip().lower() for t in str(tags_str).split(',') if t.strip()]
        return ','.join(tags)
    
    df['tags'] = df['tags'].apply(normalize_tag_string)
    return df


def clean_data(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport(len(df))
    validation_details = None
    
    if verbose:
        print("Starting data cleaning pipeline...")
    
    if verbose:
        print("  [1/7] Removing duplicate photo IDs...")
    df = remove_duplicate_ids(df, report)
    
    if verbose:
        print("  [2/7] Removing location duplicates...")
    df = remove_duplicate_locations(df, report=report)
    
    if verbose:
        print("  [3/7] Removing rows with null values...")
    df = remove_rows_with_nulls(df, report=report)
    
    if verbose:
        print("  [4/7] Validating data types and ranges...")
    df, validation_details = validate_column_types(df, report=report)
    
    if verbose:
        print("  [5/7] Filtering to Lyon area...")
    df = filter_lyon_bounds(df, report=report)
    
    if verbose:
        print("  [6/7] Filtering valid dates...")
    df = filter_valid_dates(df, report=report)
    
    if verbose:
        print("  [7/7] Normalizing tags...")
    df = normalize_tags(df)
    
    df = df.reset_index(drop=True)
    
    if verbose:
        print("Data cleaning complete!")
        if validation_details:
            print_validation_summary(validation_details)
    
    return df, report


def save_cleaned_data(df: pd.DataFrame, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")


if __name__ == "__main__":
    from data_loader import load_flickr_data
    
    data_path = Path(__file__).parent.parent / "data" / "flickr_data2.csv"
    output_path = Path(__file__).parent.parent / "outputs" / "flickr_cleaned.csv"
    
    print("Loading raw data...")
    df = load_flickr_data(data_path)
    print(f"Loaded {len(df):,} rows")
    
    print("\nCleaning data...")
    df_clean, report = clean_data(df)
    
    report.print_report()
    save_cleaned_data(df_clean, output_path)
    
    print(f"\nCleaned data shape: {df_clean.shape}")
    print(f"Sample of cleaned data:")
    print(df_clean.head())

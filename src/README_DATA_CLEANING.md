# Data Cleaning - Flickr Lyon Dataset

## Summary

- **Original:** 420,240 photos
- **After cleaning:** 72,606 photos
- **Removed:** 347,634 (82.72%)
- **Performance:** ~0.25 seconds

## Pipeline (7 steps - Optimized Order)

| Step | Description | Removed | Remaining |
|------|-------------|---------|-----------|
| 1 | Duplicate photo IDs | 252,143 (60.00%) | 168,097 |
| 2 | Location duplicates | 93,896 (22.34%) | 74,201 |
| 3 | Null values | 46 (0.01%) | 74,155 |
| 4 | Data type validation | 89 (0.02%) | 74,066 |
| 5 | Lyon bounding box | 1,460 (0.35%) | 72,606 |
| 6 | Invalid dates | 0 | 72,606 |
| 7 | Normalize tags | - | 72,606 |

## Configuration

```python
LYON_BOUNDS = {
    'lat_min': 45.7,
    'lat_max': 45.85,
    'long_min': 4.75,
    'long_max': 4.95
}

VALID_YEAR_MIN = 2000
VALID_YEAR_MAX = 2025
```

## Required Columns (cannot be null)

```python
REQUIRED_COLUMNS = [
    'id', 'user', 'lat', 'long',
    'date_taken_minute', 'date_taken_hour', 
    'date_taken_day', 'date_taken_month', 'date_taken_year',
    'date_upload_minute', 'date_upload_hour',
    'date_upload_day', 'date_upload_month', 'date_upload_year',
]
```

## Data Type Validations

| Column | Type | Range |
|--------|------|-------|
| `lat` | float | -90 to 90 |
| `long` | float | -180 to 180 |
| `date_*_minute` | int | 0-59 |
| `date_*_hour` | int | 0-23 |
| `date_*_day` | int | 1-31 |
| `date_*_month` | int | 1-12 |
| `date_*_year` | int | 2000-2025 |

## Usage

```python
from src.data_loader import load_flickr_data
from src.data_cleaning import clean_data

df = load_flickr_data('data/flickr_data2.csv')
df_clean, report = clean_data(df)
report.print_report()
```

#!/usr/bin/env python3
"""
Geo-Located Data Mining Pipeline - Lyon Areas of Interest
Usage: python main.py
"""

from pathlib import Path
from src.data_loader import load_flickr_data, get_data_info, print_data_info
from src.data_cleaning import clean_data
from src.visualization import create_map
from src.clustering import compare_algorithms, print_results

DATA_PATH = Path("data/flickr_data2.csv")
OUTPUT_DIR = Path("outputs")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("LYON GEO-LOCATED DATA MINING")
    print("=" * 70)
    
    # 1. Load
    print("\n[1/4] Loading data...")
    df_raw = load_flickr_data(DATA_PATH)
    info = get_data_info(df_raw)
    print_data_info(info)
    
    # 2. Clean
    print("\n[2/4] Cleaning data...")
    df_clean, report = clean_data(df_raw, verbose=True)
    report.print_report()
    df_clean.to_csv(OUTPUT_DIR / "flickr_cleaned.csv", index=False)
    
    # 3. Cluster - Run both algorithms
    print("\n[3/4] Clustering...")
    results = compare_algorithms(df_clean)
    
    print_results(results['dbscan']['analysis'], "DBSCAN")
    print_results(results['hdbscan']['analysis'], "HDBSCAN")
    
    # 4. Visualize with both algorithms
    print("\n[4/4] Creating interactive map...")
    create_map(
        df_clean, 
        labels_dbscan=results['dbscan']['labels'],
        labels_hdbscan=results['hdbscan']['labels'],
        output_path=str(OUTPUT_DIR / "lyon_map.html")
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  Raw data:     {len(df_raw):,} rows")
    print(f"  Clean data:   {len(df_clean):,} rows")
    print(f"\n  DBSCAN:   {results['dbscan']['analysis']['n_clusters']} clusters")
    print(f"  HDBSCAN:  {results['hdbscan']['analysis']['n_clusters']} clusters")
    print(f"\nOutputs:")
    print(f"  - {OUTPUT_DIR / 'flickr_cleaned.csv'}")
    print(f"  - {OUTPUT_DIR / 'lyon_map.html'}")
    print("\n  💡 Use the DBSCAN/HDBSCAN buttons on the map to switch algorithms!")
    print("=" * 70)


if __name__ == "__main__":
    main()

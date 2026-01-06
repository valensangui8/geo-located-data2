#!/usr/bin/env python3
"""
Geo-Located Data Mining Pipeline - Lyon Areas of Interest
Usage: python main.py
"""

from pathlib import Path
from src.data_loader import load_flickr_data, get_data_info, print_data_info
from src.data_cleaning import clean_data
from src.visualization import create_map
from src.clustering import run_dbscan, analyze_clusters, print_results

DATA_PATH = Path("data/flickr_data2.csv")
OUTPUT_DIR = Path("outputs")
DBSCAN_EPS = 0.003
DBSCAN_MIN_SAMPLES = 10


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("LYON GEO-LOCATED DATA MINING")
    print("=" * 60)
    
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
    
    # 3. Cluster
    print("\n[3/4] Running DBSCAN clustering...")
    print(f"  Parameters: eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}")
    labels = run_dbscan(df_clean, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    analysis = analyze_clusters(df_clean, labels)
    print_results(analysis)
    
    # 4. Visualize
    print("\n[4/4] Creating map with clusters...")
    create_map(df_clean, labels=labels, output_path=str(OUTPUT_DIR / "lyon_map.html"))
    
    # Summary
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Raw data:     {len(df_raw):,} rows")
    print(f"  Clean data:   {len(df_clean):,} rows")
    print(f"  Clusters:     {analysis['n_clusters']}")
    print(f"  Noise:        {analysis['noise_pct']:.1f}%")
    print(f"\nOutputs:")
    print(f"  - {OUTPUT_DIR / 'flickr_cleaned.csv'}")
    print(f"  - {OUTPUT_DIR / 'lyon_map.html'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

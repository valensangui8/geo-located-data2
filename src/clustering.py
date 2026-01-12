"""
Clustering Algorithms for Geo-located Data
Supports DBSCAN and HDBSCAN with Haversine distance
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import hdbscan
from collections import Counter

EARTH_RADIUS_KM = 6371.0

# Generic words to skip when naming clusters
SKIP_WORDS = {'lyon', 'france', 'europe', 'photo', 'square', 'squareformat', 
              'iphoneography', 'instagram', 'iphone', 'photography', 'flickr',
              '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020'}


def run_dbscan(df: pd.DataFrame, eps_meters: float = 100, min_samples: int = 10) -> np.ndarray:
    """
    Run DBSCAN with Haversine distance.
    
    Args:
        df: DataFrame with 'lat' and 'long' columns
        eps_meters: Maximum distance between points in meters
        min_samples: Minimum points to form a cluster
    """
    coords = df[['lat', 'long']].values
    eps_rad = eps_meters / (EARTH_RADIUS_KM * 1000)
    
    dbscan = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric='haversine',
        n_jobs=-1
    )
    
    coords_rad = np.radians(coords)
    labels = dbscan.fit_predict(coords_rad)
    
    return labels


def run_hdbscan(df: pd.DataFrame, min_cluster_size: int = 15, min_samples: int = 5) -> np.ndarray:
    """
    Run HDBSCAN - better for varying density clusters.
    """
    coords = df[['lat', 'long']].values
    coords_rad = np.radians(coords)
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='haversine',
        cluster_selection_method='eom',
        core_dist_n_jobs=-1
    )
    
    labels = clusterer.fit_predict(coords_rad)
    
    return labels


def filter_single_user_clusters(df: pd.DataFrame, labels: np.ndarray, min_users: int = 3) -> np.ndarray:
    """Re-label clusters with fewer than min_users unique users as noise."""
    labels = labels.copy()
    
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        
        mask = labels == cluster_id
        unique_users = df.loc[mask, 'user'].nunique()
        
        if unique_users < min_users:
            labels[mask] = -1
    
    return labels


def get_cluster_name(df: pd.DataFrame, skip_words: set = SKIP_WORDS) -> str:
    """Generate a name for a cluster based on its most distinctive tag."""
    all_tags = []
    for tags_str in df['tags'].dropna():
        if tags_str:
            tags = [t.strip().lower() for t in str(tags_str).split(',') if t.strip()]
            all_tags.extend(tags)
    
    if not all_tags:
        return "Unknown"
    
    tag_counts = Counter(all_tags)
    
    # Find first tag that's not in skip list
    for tag, count in tag_counts.most_common(10):
        if tag.lower() not in skip_words and len(tag) > 2:
            return tag.title()
    
    # Fallback to most common
    return tag_counts.most_common(1)[0][0].title() if tag_counts else "Unknown"


def get_top_tags(df: pd.DataFrame, n: int = 5) -> list:
    """Get top n tags from a dataframe."""
    all_tags = []
    for tags_str in df['tags'].dropna():
        if tags_str:
            tags = [t.strip() for t in str(tags_str).split(',') if t.strip()]
            all_tags.extend(tags)
    
    return Counter(all_tags).most_common(n)


def analyze_clusters(df: pd.DataFrame, labels: np.ndarray) -> dict:
    """Get cluster statistics with names."""
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    n_clustered = (labels >= 0).sum()
    
    clusters = []
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        
        mask = labels == cluster_id
        cluster_df = df[mask]
        
        clusters.append({
            'id': cluster_id,
            'name': get_cluster_name(cluster_df),
            'size': len(cluster_df),
            'center': (cluster_df['lat'].mean(), cluster_df['long'].mean()),
            'users': cluster_df['user'].nunique(),
            'top_tags': get_top_tags(cluster_df, n=5)
        })
    
    clusters.sort(key=lambda x: x['size'], reverse=True)
    
    return {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'n_clustered': n_clustered,
        'noise_pct': 100 * n_noise / len(labels) if len(labels) > 0 else 0,
        'clusters': clusters
    }


def print_results(analysis: dict, algorithm: str = "DBSCAN") -> None:
    """Print clustering results with cluster names."""
    print("\n" + "=" * 70)
    print(f"{algorithm} CLUSTERING RESULTS")
    print("=" * 70)
    print(f"Clusters found: {analysis['n_clusters']}")
    print(f"Clustered points: {analysis['n_clustered']:,}")
    print(f"Noise points: {analysis['n_noise']:,} ({analysis['noise_pct']:.1f}%)")
    
    print("\nTop 15 clusters by size:")
    print("-" * 70)
    
    for c in analysis['clusters'][:15]:
        tags = ", ".join([t[0] for t in c['top_tags'][:3]])
        print(f"  📍 {c['name']}")
        print(f"     {c['size']:,} photos | {c['users']} users | ({c['center'][0]:.4f}, {c['center'][1]:.4f})")
        print(f"     Tags: {tags}")
        print()
    
    print("=" * 70)


def compare_algorithms(df: pd.DataFrame) -> dict:
    """Run both algorithms and return comparison."""
    
    print("\n" + "=" * 70)
    print("COMPARING CLUSTERING ALGORITHMS")
    print("=" * 70)
    
    # DBSCAN
    print("\n[1/2] Running DBSCAN (eps=80m, min_samples=8)...")
    labels_dbscan = run_dbscan(df, eps_meters=80, min_samples=8)
    labels_dbscan = filter_single_user_clusters(df, labels_dbscan, min_users=3)
    analysis_dbscan = analyze_clusters(df, labels_dbscan)
    
    # HDBSCAN
    print("[2/2] Running HDBSCAN (min_cluster_size=15, min_samples=5)...")
    labels_hdbscan = run_hdbscan(df, min_cluster_size=15, min_samples=5)
    labels_hdbscan = filter_single_user_clusters(df, labels_hdbscan, min_users=3)
    analysis_hdbscan = analyze_clusters(df, labels_hdbscan)
    
    # Comparison table
    print("\n" + "-" * 70)
    print("COMPARISON SUMMARY")
    print("-" * 70)
    print(f"{'Metric':<30} {'DBSCAN':<20} {'HDBSCAN':<20}")
    print("-" * 70)
    print(f"{'Clusters found':<30} {analysis_dbscan['n_clusters']:<20} {analysis_hdbscan['n_clusters']:<20}")
    print(f"{'Clustered points':<30} {analysis_dbscan['n_clustered']:,<20} {analysis_hdbscan['n_clustered']:,<20}")
    print(f"{'Noise points':<30} {analysis_dbscan['n_noise']:,<20} {analysis_hdbscan['n_noise']:,<20}")
    print(f"{'Noise %':<30} {analysis_dbscan['noise_pct']:.1f}%{'':<17} {analysis_hdbscan['noise_pct']:.1f}%")
    print(f"{'Avg cluster size':<30} {analysis_dbscan['n_clustered']//max(1,analysis_dbscan['n_clusters']):<20} {analysis_hdbscan['n_clustered']//max(1,analysis_hdbscan['n_clusters']):<20}")
    print("-" * 70)
    
    return {
        'dbscan': {'labels': labels_dbscan, 'analysis': analysis_dbscan},
        'hdbscan': {'labels': labels_hdbscan, 'analysis': analysis_hdbscan}
    }


if __name__ == "__main__":
    from data_loader import load_flickr_data
    from data_cleaning import clean_data
    from pathlib import Path
    
    print("Loading and cleaning data...")
    df = load_flickr_data(Path(__file__).parent.parent / "data" / "flickr_data2.csv")
    df_clean, _ = clean_data(df, verbose=False)
    
    results = compare_algorithms(df_clean)
    
    print_results(results['dbscan']['analysis'], "DBSCAN")
    print_results(results['hdbscan']['analysis'], "HDBSCAN")

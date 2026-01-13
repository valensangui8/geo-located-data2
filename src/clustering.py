import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter


def run_dbscan(df: pd.DataFrame, eps: float = 0.003, min_samples: int = 100) -> np.ndarray:
    coords = df[['lat', 'long']].values
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    labels = dbscan.fit_predict(coords)
    return labels


def analyze_clusters(df: pd.DataFrame, labels: np.ndarray) -> dict:
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
        'noise_pct': 100 * n_noise / len(labels),
        'clusters': clusters
    }


def get_top_tags(df: pd.DataFrame, n: int = 5) -> list:
    all_tags = []
    for tags_str in df['tags'].dropna():
        if tags_str:
            tags = [t.strip() for t in str(tags_str).split(',') if t.strip()]
            all_tags.extend(tags)
    
    return Counter(all_tags).most_common(n)


def print_results(analysis: dict) -> None:
    print("\n" + "=" * 60)
    print("DBSCAN CLUSTERING RESULTS")
    print("=" * 60)
    print(f"Clusters found: {analysis['n_clusters']}")
    print(f"Clustered points: {analysis['n_clustered']:,}")
    print(f"Noise points: {analysis['n_noise']:,} ({analysis['noise_pct']:.1f}%)")
    
    print("\nTop 10 clusters:")
    print("-" * 60)
    
    for c in analysis['clusters'][:10]:
        tags = ", ".join([f"{t[0]}({t[1]})" for t in c['top_tags'][:3]])
        print(f"  Cluster {c['id']}: {c['size']:,} photos, {c['users']} users")
        print(f"    Center: ({c['center'][0]:.4f}, {c['center'][1]:.4f})")
        print(f"    Tags: {tags}")
    
    print("=" * 60)


if __name__ == "__main__":
    from data_loader import load_flickr_data
    from data_cleaning import clean_data
    from pathlib import Path
    
    print("Loading and cleaning data...")
    df = load_flickr_data(Path(__file__).parent.parent / "data" / "flickr_data2.csv")
    df_clean, _ = clean_data(df, verbose=False)
    
    print(f"\nRunning DBSCAN on {len(df_clean):,} points...")
    print("  eps=0.003 (~300m), min_samples=10")
    
    labels = run_dbscan(df_clean, eps=0.003, min_samples=10)
    analysis = analyze_clusters(df_clean, labels)
    print_results(analysis)

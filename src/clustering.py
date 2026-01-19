import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import hdbscan

try:
    from .geo_utils import meters_to_radians, project_to_meters, to_radians_coords
except ImportError:
    from geo_utils import meters_to_radians, project_to_meters, to_radians_coords


def run_dbscan(df: pd.DataFrame, eps_meters: float = 300, min_samples: int = 10,
               coords_rad: np.ndarray = None, eps_radians: float = None):
    if coords_rad is None:
        coords_rad = to_radians_coords(df)
    if eps_radians is None:
        eps_radians = meters_to_radians(eps_meters)
    dbscan = DBSCAN(eps=eps_radians, min_samples=min_samples, metric='haversine', n_jobs=-1)
    labels = dbscan.fit_predict(coords_rad)
    return labels, dbscan


def run_kmeans(df: pd.DataFrame, n_clusters: int = 10, coords_meters: np.ndarray = None):
    coords = coords_meters if coords_meters is not None else project_to_meters(df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    return labels, kmeans


def run_hdbscan(df: pd.DataFrame, min_cluster_size: int = 50, min_samples: int = 10,
                coords_rad: np.ndarray = None):
    coords = coords_rad if coords_rad is not None else to_radians_coords(df)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='haversine'
    )
    labels = clusterer.fit_predict(coords)
    return labels, clusterer




def optimize_kmeans(df: pd.DataFrame, k_range=range(5, 21), coords_meters: np.ndarray = None):
    coords = coords_meters if coords_meters is not None else project_to_meters(df)
    silhouettes = []
    
    print("  Optimizing K-Means (testing k from 5 to 20)...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        if k > 1:
            silhouettes.append(silhouette_score(coords, labels))
        else:
            silhouettes.append(0)
    
    best_k_idx = np.argmax(silhouettes)
    best_k = list(k_range)[best_k_idx]
    
    print(f"  Best k: {best_k} (silhouette: {silhouettes[best_k_idx]:.3f})")
    return best_k


def estimate_dbscan_eps(coords_rad: np.ndarray, min_samples: int, percentile: float = 90) -> float:
    neighbors = NearestNeighbors(n_neighbors=min_samples, metric='haversine')
    neighbors.fit(coords_rad)
    distances, _ = neighbors.kneighbors(coords_rad)
    kth_distances = distances[:, -1]
    return float(np.percentile(kth_distances, percentile))


def optimize_dbscan(df: pd.DataFrame, min_samples_options=(5, 10, 20),
                    percentile: float = 90, sample_size: int = 8000) -> dict:
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df

    coords_rad = to_radians_coords(sample_df)
    coords_meters = project_to_meters(sample_df)

    best = {'score': -1.0, 'min_samples': None, 'eps_radians': None}
    for min_samples in min_samples_options:
        eps_radians = estimate_dbscan_eps(coords_rad, min_samples, percentile=percentile)
        labels, _ = run_dbscan(sample_df, min_samples=min_samples,
                               coords_rad=coords_rad, eps_radians=eps_radians)

        valid_mask = labels != -1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if valid_mask.sum() > 1 and n_clusters > 1:
            score = silhouette_score(coords_meters[valid_mask], labels[valid_mask])
        else:
            score = -1.0

        if score > best['score']:
            best = {'score': score, 'min_samples': min_samples, 'eps_radians': eps_radians}

    return best


def optimize_hdbscan(df: pd.DataFrame, min_cluster_sizes=(30, 50, 100),
                     min_samples_options=(5, 10), sample_size: int = 8000) -> dict:
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df

    coords_rad = to_radians_coords(sample_df)
    coords_meters = project_to_meters(sample_df)

    best = {'score': -1.0, 'min_cluster_size': None, 'min_samples': None}
    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_options:
            labels, _ = run_hdbscan(sample_df, min_cluster_size=min_cluster_size,
                                    min_samples=min_samples, coords_rad=coords_rad)

            valid_mask = labels != -1
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if valid_mask.sum() > 1 and n_clusters > 1:
                score = silhouette_score(coords_meters[valid_mask], labels[valid_mask])
            else:
                score = -1.0

            if score > best['score']:
                best = {
                    'score': score,
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples
                }

    return best


def dbscan_parameter_sensitivity(df: pd.DataFrame,
                                 eps_meters_list=(200, 300, 400, 500),
                                 min_samples_list=(5, 10, 20),
                                 sample_size: int = 12000) -> pd.DataFrame:
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df

    coords_rad = to_radians_coords(sample_df)
    coords_meters = project_to_meters(sample_df)

    rows = []
    for eps_m in eps_meters_list:
        for min_samples in min_samples_list:
            labels, _ = run_dbscan(
                sample_df,
                eps_meters=eps_m,
                min_samples=min_samples,
                coords_rad=coords_rad
            )
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_pct = (labels == -1).sum() / len(labels) * 100

            valid_mask = labels != -1
            if valid_mask.sum() > 1 and n_clusters > 1:
                score = silhouette_score(coords_meters[valid_mask], labels[valid_mask])
            else:
                score = -1.0

            rows.append({
                'eps_meters': eps_m,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'noise_pct': noise_pct,
                'silhouette_score': score
            })

    return pd.DataFrame(rows).sort_values(['silhouette_score'], ascending=False)




def run_subclustering(df: pd.DataFrame, parent_labels: np.ndarray, 
                      parent_cluster_id: int, algorithm: str = 'hdbscan'):
    mask = parent_labels == parent_cluster_id
    cluster_df = df[mask]
    
    print(f"\n  Running sub-clustering on cluster {parent_cluster_id} ({len(cluster_df):,} points)...")
    
    if algorithm == 'hdbscan':
        sub_labels, _ = run_hdbscan(cluster_df, min_cluster_size=30, min_samples=5)
    elif algorithm == 'dbscan':
        sub_labels, _ = run_dbscan(cluster_df, eps_meters=120, min_samples=10)
    elif algorithm == 'kmeans':
        optimal_k = optimize_kmeans(cluster_df, k_range=range(3, 11))
        sub_labels, _ = run_kmeans(cluster_df, n_clusters=optimal_k)
    else:
        sub_labels = np.zeros(len(cluster_df), dtype=int)
    
    full_labels = parent_labels.copy()
    
    max_parent_label = parent_labels.max()
    unique_sublabels = set(sub_labels)
    unique_sublabels.discard(-1)
    
    for idx, sub_label in enumerate(sorted(unique_sublabels)):
        sub_mask = sub_labels == sub_label
        new_label = max_parent_label + 1 + idx
        full_labels[np.where(mask)[0][sub_mask]] = new_label
    
    noise_mask = sub_labels == -1
    if noise_mask.any():
        full_labels[np.where(mask)[0][noise_mask]] = -1
    
    n_subclusters = len(unique_sublabels)
    print(f"  Found {n_subclusters} sub-clusters within cluster {parent_cluster_id}")
    
    return full_labels


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
    print("  eps=300m, min_samples=10")
    
    labels, _ = run_dbscan(df_clean, eps_meters=300, min_samples=10)
    analysis = analyze_clusters(df_clean, labels)
    print_results(analysis)

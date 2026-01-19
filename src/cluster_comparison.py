import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

try:
    from .geo_utils import project_to_meters
except ImportError:
    from geo_utils import project_to_meters


def evaluate_clustering(df: pd.DataFrame, labels: np.ndarray, algorithm_name: str,
                        sample_size: int = 10000) -> dict:
    coords = project_to_meters(df)

    if len(df) > sample_size:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(df), size=sample_size, replace=False)
        coords = coords[sample_idx]
        labels = labels[sample_idx]
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).sum() / len(labels) * 100
    
    valid_mask = labels != -1
    if valid_mask.sum() > 1 and n_clusters > 1:
        try:
            silhouette = silhouette_score(coords[valid_mask], labels[valid_mask])
        except:
            silhouette = 0.0
        
        try:
            davies_bouldin = davies_bouldin_score(coords[valid_mask], labels[valid_mask])
        except:
            davies_bouldin = 0.0
    else:
        silhouette = 0.0
        davies_bouldin = 0.0
    
    cluster_sizes = []
    for cluster_id in set(labels):
        if cluster_id != -1:
            cluster_sizes.append((labels == cluster_id).sum())
    
    metrics = {
        'algorithm': algorithm_name,
        'n_clusters': n_clusters,
        'noise_pct': noise_pct,
        'silhouette_score': silhouette,
        'davies_bouldin_index': davies_bouldin,
        'largest_cluster': max(cluster_sizes) if cluster_sizes else 0,
        'smallest_cluster': min(cluster_sizes) if cluster_sizes else 0,
        'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0
    }
    
    return metrics


def compare_algorithms(df: pd.DataFrame, all_labels: dict, sample_size: int = 10000) -> pd.DataFrame:
    results = []
    for name, labels in all_labels.items():
        print(f"  Evaluating {name}...")
        metrics = evaluate_clustering(df, labels, name, sample_size=sample_size)
        results.append(metrics)
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('silhouette_score', ascending=False)
    
    return comparison_df


def print_comparison_table(comparison_df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("CLUSTERING ALGORITHMS COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Algorithm':<15} {'Clusters':<10} {'Noise %':<10} {'Silhouette':<12} {'Davies-Bouldin':<15}")
    print("-" * 80)
    
    for _, row in comparison_df.iterrows():
        print(f"{row['algorithm']:<15} {row['n_clusters']:<10} "
              f"{row['noise_pct']:<10.2f} {row['silhouette_score']:<12.3f} "
              f"{row['davies_bouldin_index']:<15.3f}")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("  - Silhouette Score: Higher is better (range: -1 to 1)")
    print("  - Davies-Bouldin Index: Lower is better")
    print("  - Noise %: Percentage of points not assigned to any cluster")
    print("=" * 80)

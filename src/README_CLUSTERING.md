# Clustering - DBSCAN for Areas of Interest

## Summary

DBSCAN clustering to discover areas of interest from geo-located photo data.

## Why DBSCAN?

- Finds clusters of arbitrary shape
- No need to specify number of clusters
- Handles noise (outliers labeled as -1)
- Works well with spatial data

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eps` | 0.003 | Max distance between points (~300m at Lyon's latitude) |
| `min_samples` | 10 | Min points to form a cluster |

## Results (eps=0.003, min_samples=10)

| Metric | Value |
|--------|-------|
| Clusters | 65 |
| Clustered | 71,872 (99%) |
| Noise | 734 (1%) |

## Top Clusters

| Cluster | Photos | Location | Top Tags |
|---------|--------|----------|----------|
| 0 | 67,926 | Lyon Center | lyon, france |
| 1 | 1,604 | Demeure du Chaos | abodeofchaos |
| 22 | 217 | Parilly | parilly |
| 3 | 161 | Villeurbanne | villeurbanne |

## Usage

```python
from src.clustering import run_dbscan, analyze_clusters, print_results

labels = run_dbscan(df_clean, eps=0.003, min_samples=10)
analysis = analyze_clusters(df_clean, labels)
print_results(analysis)
```

## Analysis Output

```python
analysis = {
    'n_clusters': 65,
    'n_noise': 734,
    'n_clustered': 71872,
    'noise_pct': 1.0,
    'clusters': [
        {
            'id': 0,
            'size': 67926,
            'center': (45.7622, 4.8348),
            'users': 4845,
            'top_tags': [('lyon', 35299), ('france', 21588), ...]
        },
        ...
    ]
}
```

## Tuning Tips

- Smaller `eps` = more smaller clusters
- Larger `eps` = fewer larger clusters
- Higher `min_samples` = denser clusters required


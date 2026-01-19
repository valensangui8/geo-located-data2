# Milestone 2 Defense Notes (Clustering + Text Mining)

This document summarizes what was implemented for Milestone 2 and the theory
you should be able to explain during the defense.

---

## 1. What we delivered for Milestone 2

### Functional deliverables
- **3+ clustering algorithms** with parameter optimization:
  - DBSCAN
  - K-Means
  - HDBSCAN
- **Quantitative comparison** of algorithms:
  - Silhouette score
  - Davies–Bouldin index
  - Number of clusters
  - Noise percentage
- **Text mining** to describe clusters:
  - TF-IDF over tags + titles
  - Top keywords per cluster exported to CSV
- **Visualization**:
  - Single best-algorithm interactive map
- Multi-layer comparison map (DBSCAN, K-Means, HDBSCAN)

### Files generated (outputs)
- `outputs/flickr_cleaned.csv`
- `outputs/clustering_comparison.csv`
- `outputs/cluster_keywords.csv`
- `outputs/lyon_map_best.html`
- `outputs/clustering_comparison.html`

---

## 2. Data preparation (Milestone 1 recap)

We reuse the Milestone 1 cleaning pipeline:
- remove duplicate photo IDs
- remove near-duplicate photos (same user/location/day)
- remove nulls
- validate numeric ranges for dates and coordinates
- filter to Lyon bounding box
- normalize tags (lowercase, deduplicate format)

---

## 3. Spatial representation choices

### Why not use raw degrees directly?
Lat/long are in degrees; Euclidean distance in degrees is not uniform in meters.

### Two coordinate representations used
- **Haversine distance in radians** for density-based algorithms
  - DBSCAN, HDBSCAN, OPTICS
  - More accurate for geospatial clustering
- **Projected meters** for K-Means and evaluation metrics
  - Simple equirectangular projection:
    - `x = lon_rad * cos(mean_lat) * R`
    - `y = lat_rad * R`
  - Allows standard Euclidean distance for K-Means and silhouette/Davies–Bouldin

### Why this is defensible
Haversine gives realistic geographic distances; K-Means assumes Euclidean geometry,
so we transform to meters before using it.

---

## 4. Clustering algorithms used

### 4.1 DBSCAN
- **Type**: density-based
- **Key parameters**:
  - `eps`: neighborhood radius
  - `min_samples`: min points to form a core point
- **Pros**:
  - Detects arbitrary shapes
  - Handles noise naturally
  - No need to predefine K
- **Cons**:
  - Sensitive to parameters
  - Single density level (can struggle with varying densities)

### 4.2 K-Means
- **Type**: partition-based
- **Key parameter**:
  - `n_clusters` (K)
- **Pros**:
  - Very fast and scalable
  - Simple to interpret
- **Cons**:
  - Assumes spherical clusters
  - All points assigned to some cluster (no noise)
  - Sensitive to K and initialization

### 4.3 HDBSCAN
- **Type**: hierarchical density-based
- **Key parameters**:
  - `min_cluster_size`
  - `min_samples`
- **Pros**:
  - Handles varying densities
  - Automatically finds number of clusters
  - Robust to noise
- **Cons**:
  - More complex
  - Heavier computation

## 5. Parameter optimization strategy

### DBSCAN
- Estimate `eps` from **k-distance distribution**:
  - Compute distance to the k-th nearest neighbor
  - Choose `eps` at a high percentile (90th)
- Try multiple `min_samples`
- Keep the pair with the best silhouette score

### K-Means
- Test `k` from 5 to 25
- Select `k` with **max silhouette score**

### HDBSCAN
- Grid search:
  - `min_cluster_size` in {30, 50, 100}
  - `min_samples` in {5, 10}
- Pick best silhouette score

### Why silhouette score?
It measures intra-cluster cohesion vs inter-cluster separation.

---

## 6. Evaluation metrics explained

- **Silhouette score (−1 to +1)**:
  - Higher is better
  - Works when there are at least 2 clusters
- **Davies–Bouldin index**:
  - Lower is better
  - Penalizes clusters that are too close or overlapping
- **Noise percentage**:
  - Only for density-based methods
  - % of points labeled `-1`
- **Cluster sizes**:
  - Largest, smallest, and average cluster size
  - Helps identify imbalance

---

## 7. Text mining (TF-IDF) to describe clusters

### Preprocessing
- Use both **title + tags** as text
- Lowercase, remove punctuation
- Remove English + French stopwords
- Remove dataset-specific generic words (e.g., “photo”, “lyon”, “france”)

### TF-IDF intuition
TF-IDF favors words that are:
1. frequent **inside a cluster**
2. rare **across other clusters**

### Equation
- **TF-IDF(t, d)** = TF(t, d) × IDF(t)
  - **TF**: term frequency in document
  - **IDF**: log(total_docs / docs_with_term)

### Output
- `outputs/cluster_keywords.csv`:
  - `algorithm`, `cluster_id`, `keyword`, `score`

---

## 8. Visualization

### `lyon_map_best.html`
- Shows clusters for the best algorithm (by silhouette)
- Click cluster center to inspect:
  - number of photos
  - number of users
  - top tags

### `clustering_comparison.html`
- 3 algorithm layers, toggle on/off
- Compare shape and placement of clusters
- Popups include TF-IDF keywords

---

## 9. What to say in defense (key points)

### Scientific methodology
- We **tested multiple algorithms**, not just one
- We **optimized parameters** systematically
- We **validated** with quantitative metrics + map visualization

### Why clustering matters to Grand Lyon
- Reveals **high-density tourist areas**
- Helps prioritize zones for transport or tourism services

### Why text mining is essential
- Clusters become **interpretable**
- Keywords connect clusters to **real places/events**

---

## 10. Expected Q&A

### “Why use Haversine?”
- Because Euclidean on degrees is distorted; haversine gives real distances on Earth.

### “Why not only use K-Means?”
- K-Means forces spherical clusters and cannot detect noise.

### “How do you know if clusters are good?”
- We combine silhouette/Davies–Bouldin with map interpretation.

### “Could TF-IDF be biased?”
- Yes, but it still highlights discriminant words per cluster.
  We also remove generic and stop words to reduce noise.

---

## 11. How to run

```bash
pip install -r requirements.txt
python main.py
```

---

## 12. Limitations to acknowledge

- Density-based methods can be sensitive to parameters.
- TF-IDF ignores word order and semantics.
- Some clusters may mix multiple landmarks if they are close.

These are acceptable limitations for Milestone 2, and will be improved in Milestone 3.


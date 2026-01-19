# Milestone 3: Temporal Analysis (Events & Seasonality)

## Summary

Milestone 3 extends the project with a temporal axis and finalizes text mining:
- Build time series per cluster (monthly)
- Detect event candidates using spike detection
- Analyze seasonality (month-of-year, weekday)
- Two automatic cluster naming methods (TF-IDF + TF)
- Extra: association rules for tag co-occurrences
- DBSCAN parameter sensitivity table (eps/min_samples impact)
- Known-event validation (Fête des Lumières month check)
- Export CSV outputs for reporting and visualization

---

## 1. Temporal Features

We use the **date_taken** fields from Flickr:
- `date_taken_year`, `date_taken_month`, `date_taken_day`

These are combined into a `taken_date` timestamp. Rows with invalid dates are ignored
for temporal analysis only (not removed from spatial analysis).

---

## 2. Time Series per Cluster

We aggregate photo counts per cluster with a monthly frequency:
- `cluster_time_series.csv`

Columns:
- `cluster`
- `period_start`
- `photo_count`
- `user_count`

This shows how activity evolves over time for each area of interest.

---

## 3. Event Detection (Spike Analysis)

We detect abnormal peaks in each cluster’s time series using z-scores:

- Compute mean and standard deviation of monthly counts
- Compute z-score per month
- Keep months with:
  - `zscore >= 3.0`
  - `photo_count >= 50`

Output:
- `event_candidates.csv`

Columns:
- `cluster`
- `period_start`
- `photo_count`
- `user_count`
- `zscore`
- `event_match`
- `known_event`

Known-event validation:
- We tag spikes by month for known Lyon events:
  - Fête des Lumières (December)
  - Nuits de Fourvière (June–July)
  - Festival Lumière (October)
  - Quais du Polar (March–April)
  - Fête de la Musique (June)
  - Run In Lyon (October)
  - Biennale de la Danse (September)
  - Biennale d Art Contemporain (Sep-Dec)

These spikes are potential events (festivals, exceptional exhibitions, etc.).

---

## 4. Seasonality

We compute two seasonal profiles:

1. **Monthly seasonality**
   - `seasonality_monthly.csv`
   - Photo activity per month-of-year

2. **Weekly seasonality**
   - `seasonality_weekday.csv`
   - Photo activity per weekday

These help determine if clusters are seasonal or related to weekend tourism.

---

## 5. Parameter Sensitivity (DBSCAN)

We generate a table to show how clusters change with:
- `eps` in meters
- `min_samples`

Output:
- `dbscan_sensitivity.csv`

Columns:
- `eps_meters`
- `min_samples`
- `n_clusters`
- `noise_pct`
- `silhouette_score`

---

## 6. How to Run

```bash
pip install -r requirements.txt
python main.py
```

Fast run with saved optimal parameters:

```bash
python main_optimal.py
```

Outputs are written to `outputs/`.

Association rule outputs:
- `outputs/cluster_rules_dbscan.csv`
- `outputs/cluster_rules_k-means.csv`
- `outputs/cluster_rules_hdbscan.csv`

Plots generated:
- `outputs/dbscan_sensitivity.png` (heatmap)
- `outputs/event_candidates_timeline.png` (timeline)

---

## 6. Limitations

- Event detection is statistical and can produce false positives.
- Monthly aggregation can hide short events (1–2 days).
- Clusters with few photos may not yield reliable temporal patterns.

---

## 7. Next Steps (Optional)

- Daily aggregation for finer events
- Combine spatial + temporal clustering
- Visual timeline or animated map

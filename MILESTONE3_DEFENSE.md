# Milestone 3 Defense Notes (Temporal Analysis)

This document summarizes what was implemented for Milestone 3 and the theory
you should be able to explain during the defense.

---

## 1. What we delivered for Milestone 3

- **Temporal aggregation per cluster** (monthly time series)
- **Event detection** using statistical spike detection
- **Seasonality analysis** (month-of-year and weekday)
- **CSV outputs** ready for demo and slides
- **Two cluster naming methods**: TF-IDF and TF
- **Extra**: association rules for tag co-occurrence
- **DBSCAN sensitivity table** (eps/min_samples)
- **Known-event validation** (Fête des Lumières month check)
- **Plots** for clarity in the demo (sensitivity heatmap + event timeline)

Outputs:
- `outputs/cluster_time_series.csv`
- `outputs/event_candidates.csv`
- `outputs/seasonality_monthly.csv`
- `outputs/seasonality_weekday.csv`

---

## 2. Why temporal analysis matters

Spatial clusters alone do not tell if an area is:
- a constant tourist hotspot
- seasonal (summer vs winter)
- driven by specific events (festivals, exhibitions)

Temporal patterns provide this missing dimension.

---

## 3. Temporal data preparation

We build a `taken_date` timestamp from:
- `date_taken_year`
- `date_taken_month`
- `date_taken_day`

Invalid dates are ignored **only for temporal analysis** so spatial clustering
remains unchanged.

---

## 4. Time series per cluster

We aggregate photos per cluster by month:
- count of photos
- count of unique users

This gives a time signal per area of interest.

---

## 5. Event detection (spike analysis)

We use **z-score** on monthly counts:

```
z = (count - mean) / std
```

Event candidate if:
- `z >= 3.0`
- `photo_count >= 50`

This detects unusual surges in activity.

---

## 6. Seasonality analysis

Two seasonal profiles are computed:
- **Monthly**: photo counts by month-of-year
- **Weekly**: photo counts by weekday

This tells us if a cluster is linked to:
- summer tourism
- weekend visits
- recurring annual events

---

## 7. How to explain the outputs

- `cluster_time_series.csv`:
  - base temporal signal
- `event_candidates.csv`:
  - possible events (spikes)
- `dbscan_sensitivity.csv`:
  - how DBSCAN clusters change with parameters
- `dbscan_sensitivity.png`:
  - visual heatmap for parameter sensitivity
- `event_candidates_timeline.png`:
  - quick view of detected event spikes
- `seasonality_monthly.csv`:
  - seasonal profile
- `seasonality_weekday.csv`:
  - weekday vs weekend activity
- `cluster_rules_*.csv`:
  - association rules per algorithm (tags that co-occur)

---

## 8. Possible questions

### “Why use monthly aggregation?”
- Stable signal for large datasets; reduces noise.

### “Is z-score enough for event detection?”
- For milestone scope, yes; it’s simple and interpretable.
  Future work can use more advanced models.

### “How to validate events?”
- Cross-check with known festival dates (Fête des Lumières, etc.)
  We tag spikes by month for known Lyon events:
  - Fete des Lumieres (December)
  - Nuits de Fourviere (June-July)
  - Festival Lumiere (October)
  - Quais du Polar (March-April)
  - Fete de la Musique (June)
  - Run In Lyon (October)
  - Biennale de la Danse (September)
  - Biennale d Art Contemporain (Sep-Dec)

---

## 9. Limitations (be transparent)

- Short events may be diluted in monthly bins.
- Sparse clusters do not have reliable temporal signals.
- Statistical spikes need manual validation.


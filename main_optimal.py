#!/usr/bin/env python3

from pathlib import Path
import json
import pandas as pd

from src.data_loader import load_flickr_data, get_data_info, print_data_info
from src.data_cleaning import clean_data
from src.visualization import create_map, create_comparison_map
from src.clustering import (run_dbscan, run_kmeans, run_hdbscan,
                            analyze_clusters)
from src.cluster_comparison import compare_algorithms, print_comparison_table
from src.text_mining import (prepare_cluster_documents, compute_tfidf, compute_tf,
                             build_keywords_table, print_tfidf_results)
from src.temporal_analysis import (compute_cluster_time_series, detect_event_spikes,
                                   annotate_known_events, compute_seasonality,
                                   summarize_temporal_span)
from src.temporal_plots import (plot_dbscan_sensitivity_heatmap,
                                plot_event_timeline)
from src.association_rules import mine_rules_per_cluster
from src.geo_utils import project_to_meters

DATA_PATH = Path("data/flickr_data2.csv")
OUTPUT_DIR = Path("outputs")
OPTIMAL_PARAMS_PATH = OUTPUT_DIR / "optimal_params.json"


def load_optimal_params() -> dict:
    if OPTIMAL_PARAMS_PATH.exists():
        with open(OPTIMAL_PARAMS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'dbscan': {'eps_radians': None, 'eps_meters': 350.0, 'min_samples': 20},
        'kmeans': {'n_clusters': 6},
        'hdbscan': {'min_cluster_size': 30, 'min_samples': 10},
        'best_algorithm': 'HDBSCAN'
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("LYON GEO-LOCATED DATA MINING - MILESTONE 3 (OPTIMAL RUN)")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    df_raw = load_flickr_data(DATA_PATH)
    info = get_data_info(df_raw)
    print_data_info(info)

    print("\n[2/6] Cleaning data...")
    df_clean, report = clean_data(df_raw, verbose=True)
    report.print_report()
    df_clean.to_csv(OUTPUT_DIR / "flickr_cleaned.csv", index=False)

    params = load_optimal_params()
    coords_meters = project_to_meters(df_clean)

    print("\n[3/6] Running clustering with optimal parameters...")
    labels_dbscan, _ = run_dbscan(
        df_clean,
        min_samples=params['dbscan']['min_samples'],
        eps_radians=params['dbscan'].get('eps_radians'),
        eps_meters=params['dbscan'].get('eps_meters', 350.0)
    )
    labels_kmeans, _ = run_kmeans(df_clean, n_clusters=params['kmeans']['n_clusters'],
                                  coords_meters=coords_meters)
    labels_hdbscan, _ = run_hdbscan(
        df_clean,
        min_cluster_size=params['hdbscan']['min_cluster_size'],
        min_samples=params['hdbscan']['min_samples']
    )

    all_labels = {
        'DBSCAN': labels_dbscan,
        'K-Means': labels_kmeans,
        'HDBSCAN': labels_hdbscan
    }

    comparison_df = compare_algorithms(df_clean, all_labels, sample_size=12000)
    print_comparison_table(comparison_df)
    comparison_df.to_csv(OUTPUT_DIR / "clustering_comparison.csv", index=False)

    best_algorithm = params.get('best_algorithm', comparison_df.iloc[0]['algorithm'])
    best_labels = all_labels.get(best_algorithm, all_labels[comparison_df.iloc[0]['algorithm']])

    print("\n[4/6] Running text mining and association rules...")
    tfidf_results_by_algo = {}
    tf_results_by_algo = {}
    rules_by_algo = {}
    keyword_tables = []
    for algo_name, labels in all_labels.items():
        cluster_docs = prepare_cluster_documents(df_clean, labels)
        tfidf_results = compute_tfidf(cluster_docs, top_n=10)
        tf_results = compute_tf(cluster_docs, top_n=10)
        tfidf_results_by_algo[algo_name] = tfidf_results
        tf_results_by_algo[algo_name] = tf_results
        keyword_tables.append(build_keywords_table(tfidf_results, algo_name, method='tfidf'))
        keyword_tables.append(build_keywords_table(tf_results, algo_name, method='tf'))

        rules_display, rules_df = mine_rules_per_cluster(df_clean, labels)
        rules_by_algo[algo_name] = rules_display
        rules_df.insert(0, 'algorithm', algo_name)
        file_name = f"cluster_rules_{algo_name.lower().replace(' ', '_')}.csv"
        rules_df.to_csv(OUTPUT_DIR / file_name, index=False)

    if keyword_tables:
        keywords_df = pd.concat(keyword_tables, ignore_index=True)
        keywords_df.to_csv(OUTPUT_DIR / "cluster_keywords.csv", index=False)

    print_tfidf_results(tfidf_results_by_algo[best_algorithm], df_clean, best_labels, top_clusters=10)

    print("\n[5/6] Temporal analysis...")
    time_series = compute_cluster_time_series(df_clean, best_labels, freq='MS')
    time_series.to_csv(OUTPUT_DIR / "cluster_time_series.csv", index=False)

    events = detect_event_spikes(time_series, z_threshold=3.0, min_count=50)
    events = annotate_known_events(events)
    events.to_csv(OUTPUT_DIR / "event_candidates.csv", index=False)
    plot_event_timeline(events, str(OUTPUT_DIR / "event_candidates_timeline.png"))

    monthly, weekday = compute_seasonality(df_clean, best_labels)
    monthly.to_csv(OUTPUT_DIR / "seasonality_monthly.csv", index=False)
    weekday.to_csv(OUTPUT_DIR / "seasonality_weekday.csv", index=False)

    min_date, max_date = summarize_temporal_span(df_clean)

    print("\n[6/6] Creating visualizations...")
    create_map(
        df_clean,
        labels=best_labels,
        output_path=str(OUTPUT_DIR / "lyon_map_best.html")
    )
    create_comparison_map(
        df_clean,
        all_labels,
        tfidf_results=tfidf_results_by_algo,
        tf_results=tf_results_by_algo,
        rules_results=rules_by_algo,
        output_path=str(OUTPUT_DIR / "clustering_comparison.html")
    )

    analysis = analyze_clusters(df_clean, best_labels)

    print("\nSummary")
    print("\n" + "=" * 60)
    print("MILESTONE 3 COMPLETE (OPTIMAL RUN)")
    print("=" * 60)
    print(f"  Raw data:              {len(df_raw):,} rows")
    print(f"  Clean data:            {len(df_clean):,} rows")
    print(f"  Best algorithm:        {best_algorithm}")
    print(f"  Total clusters:        {analysis['n_clusters']}")
    print(f"  Noise:                 {analysis['noise_pct']:.1f}%")
    if pd.notna(min_date) and pd.notna(max_date):
        print(f"  Temporal span:         {min_date.date()} to {max_date.date()}")
    print(f"\nOutputs:")
    print(f"  - {OUTPUT_DIR / 'flickr_cleaned.csv'}")
    print(f"  - {OUTPUT_DIR / 'clustering_comparison.csv'}")
    print(f"  - {OUTPUT_DIR / 'cluster_keywords.csv'}")
    print(f"  - {OUTPUT_DIR / 'cluster_rules_dbscan.csv'}")
    print(f"  - {OUTPUT_DIR / 'cluster_rules_k-means.csv'}")
    print(f"  - {OUTPUT_DIR / 'cluster_rules_hdbscan.csv'}")
    print(f"  - {OUTPUT_DIR / 'cluster_time_series.csv'}")
    print(f"  - {OUTPUT_DIR / 'event_candidates.csv'}")
    print(f"  - {OUTPUT_DIR / 'event_candidates_timeline.png'}")
    print(f"  - {OUTPUT_DIR / 'seasonality_monthly.csv'}")
    print(f"  - {OUTPUT_DIR / 'seasonality_weekday.csv'}")
    print(f"  - {OUTPUT_DIR / 'lyon_map_best.html'}")
    print(f"  - {OUTPUT_DIR / 'clustering_comparison.html'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

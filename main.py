#!/usr/bin/env python3
"""
LYON GEO-LOCATED DATA MINING
Proyecto de Data Mining - INSA Lyon 2025-2026

Descubrimiento de áreas de interés y eventos a partir de datos geolocalizados.
"""

from pathlib import Path
from src.data_loader import load_flickr_data, get_data_info, print_data_info
from src.data_cleaning import clean_data
from src.visualization import create_map
from src.clustering import (
    run_hdbscan, analyze_clusters, print_results
)

DATA_PATH = Path("data/flickr_data2.csv")
OUTPUT_DIR = Path("outputs")

# Parámetros HDBSCAN para detectar zonas turísticas
# HDBSCAN ajusta automáticamente la densidad según la zona
HDBSCAN_MIN_CLUSTER_SIZE = 50   # Mínimo 50 fotos para ser un punto de interés
HDBSCAN_MIN_SAMPLES = 10        # Puntos mínimos para ser "core point"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("LYON GEO-LOCATED DATA MINING")
    print("    Descubrimiento de áreas de interés turístico")
    print("=" * 70)
    
    # Cargar datos
    print("\n[1/4] Cargando datos...")
    df_raw = load_flickr_data(DATA_PATH)
    info = get_data_info(df_raw)
    print_data_info(info)
    
    # Limpieza de datos
    print("\n[2/4] Limpiando datos...")
    df_clean, report = clean_data(df_raw, verbose=True)
    report.print_report()
    
    df_clean.to_csv(OUTPUT_DIR / "flickr_cleaned.csv", index=False)
    print(f"  Datos limpios guardados en: {OUTPUT_DIR / 'flickr_cleaned.csv'}")
    
    # HDBSCAN - Clustering de densidad variable
    print("\n[3/4] Ejecutando HDBSCAN...")
    print(f"      min_cluster_size = {HDBSCAN_MIN_CLUSTER_SIZE}")
    print(f"      min_samples = {HDBSCAN_MIN_SAMPLES}")
    print("      (HDBSCAN ajusta automáticamente la densidad)")
    
    labels = run_hdbscan(df_clean, 
                         min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, 
                         min_samples=HDBSCAN_MIN_SAMPLES)
    analysis = analyze_clusters(df_clean, labels, "HDBSCAN")
    
    # Mostrar resultados
    print_results(analysis)
    
    # Crear visualización
    print("\n[4/4] Creando mapa interactivo...")
    create_map(df_clean, labels=labels, output_path=str(OUTPUT_DIR / "lyon_map.html"))
    
    # RESUMEN FINAL
    print("\n" + "=" * 70)
    print("PROCESO COMPLETADO")
    print("=" * 70)
    print(f"  Datos originales:  {len(df_raw):,} filas")
    print(f"  Datos limpios:     {len(df_clean):,} filas")
    print(f"  Reducción:         {100*(1-len(df_clean)/len(df_raw)):.1f}%")
    print()
    print(f"  HDBSCAN (min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, min_samples={HDBSCAN_MIN_SAMPLES}):")
    print(f"     Clusters: {analysis['n_clusters']}")
    print(f"     Ruido: {analysis['n_noise']:,} puntos ({analysis['noise_pct']:.1f}%)")
    if analysis.get('metrics'):
        print(f"     Silhouette: {analysis['metrics'].get('silhouette', 'N/A'):.4f}")
        print(f"     Calinski-Harabasz: {analysis['metrics'].get('calinski_harabasz', 'N/A'):.1f}")
        print(f"     Davies-Bouldin: {analysis['metrics'].get('davies_bouldin', 'N/A'):.4f}")
    print()
    print(f"  Archivos generados:")
    print(f"     - {OUTPUT_DIR / 'flickr_cleaned.csv'}")
    print(f"     - {OUTPUT_DIR / 'lyon_map.html'}")
    print("=" * 70)
    
    return analysis, labels, df_clean


if __name__ == "__main__":
    analysis, labels, df = main()

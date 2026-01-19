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
    run_dbscan, run_kmeans, run_hierarchical,
    analyze_clusters, print_results,
    compare_algorithms, print_comparison
)

DATA_PATH = Path("data/flickr_data2.csv")
OUTPUT_DIR = Path("outputs")

# Parámetros por defecto (se optimizan automáticamente)
DBSCAN_EPS = 0.003
DBSCAN_MIN_SAMPLES = 50
KMEANS_CLUSTERS = 20
HIERARCHICAL_CLUSTERS = 20


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🗺️  LYON GEO-LOCATED DATA MINING")
    print("    Descubrimiento de áreas de interés turístico")
    print("=" * 70)
    
    # =========================================================================
    # PASO 1: Cargar datos
    # =========================================================================
    print("\n[1/5] 📂 Cargando datos...")
    df_raw = load_flickr_data(DATA_PATH)
    info = get_data_info(df_raw)
    print_data_info(info)
    
    # =========================================================================
    # PASO 2: Limpieza de datos
    # =========================================================================
    print("\n[2/5] 🧹 Limpiando datos...")
    df_clean, report = clean_data(df_raw, verbose=True)
    report.print_report()
    
    df_clean.to_csv(OUTPUT_DIR / "flickr_cleaned.csv", index=False)
    print(f"  ✓ Datos limpios guardados en: {OUTPUT_DIR / 'flickr_cleaned.csv'}")
    
    # =========================================================================
    # PASO 3: Comparación de algoritmos de clustering
    # =========================================================================
    print("\n[3/5] 🔬 Comparando algoritmos de clustering...")
    print("      (DBSCAN vs K-Means vs Hierarchical)")
    
    # Ejecutar comparación con optimización
    results = compare_algorithms(df_clean, optimize=True, verbose=True)
    print_comparison(results)
    
    # =========================================================================
    # PASO 4: Seleccionar mejor algoritmo y crear mapa
    # =========================================================================
    print("\n[4/5] 🏆 Seleccionando mejor algoritmo...")
    
    # Elegir el algoritmo con mejor silhouette (excluyendo DBSCAN si tiene mucho ruido)
    best_algo = None
    best_score = -1
    
    for name, data in results.items():
        sil = data.get('metrics', {}).get('silhouette', -1)
        # Penalizar si hay mucho ruido
        if data['noise_pct'] > 50:
            sil *= 0.5
        if sil > best_score:
            best_score = sil
            best_algo = name
    
    print(f"  ✓ Algoritmo seleccionado: {results[best_algo]['algorithm']}")
    print(f"    Silhouette score: {best_score:.4f}")
    print(f"    Clusters: {results[best_algo]['n_clusters']}")
    
    best_labels = results[best_algo]['labels']
    best_analysis = results[best_algo]
    
    # =========================================================================
    # PASO 5: Crear visualización
    # =========================================================================
    print("\n[5/5] 🗺️  Creando mapa interactivo...")
    create_map(df_clean, labels=best_labels, output_path=str(OUTPUT_DIR / "lyon_map.html"))
    
    # También crear mapas para cada algoritmo
    for name, data in results.items():
        output_file = OUTPUT_DIR / f"lyon_map_{name}.html"
        create_map(df_clean, labels=data['labels'], output_path=str(output_file))
    
    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ PROCESO COMPLETADO")
    print("=" * 70)
    print(f"  📊 Datos originales:  {len(df_raw):,} filas")
    print(f"  📊 Datos limpios:     {len(df_clean):,} filas")
    print(f"  📊 Reducción:         {100*(1-len(df_clean)/len(df_raw)):.1f}%")
    print()
    print(f"  🏆 Mejor algoritmo:   {results[best_algo]['algorithm']}")
    print(f"  📍 Clusters:          {best_analysis['n_clusters']}")
    print(f"  🔇 Ruido:             {best_analysis['noise_pct']:.1f}%")
    print()
    print(f"  📁 Archivos generados:")
    print(f"     - {OUTPUT_DIR / 'flickr_cleaned.csv'}")
    print(f"     - {OUTPUT_DIR / 'lyon_map.html'} (mejor algoritmo)")
    print(f"     - {OUTPUT_DIR / 'lyon_map_dbscan.html'}")
    print(f"     - {OUTPUT_DIR / 'lyon_map_kmeans.html'}")
    print(f"     - {OUTPUT_DIR / 'lyon_map_hierarchical.html'}")
    print("=" * 70)
    
    return results, df_clean


if __name__ == "__main__":
    results, df = main()

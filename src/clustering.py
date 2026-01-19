import pandas as pd
import numpy as np
import time
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuración para datasets grandes
MAX_SAMPLES_HIERARCHICAL = 8000  # Muestra máxima para hierarchical (es O(n²))
MAX_SAMPLES_SILHOUETTE = 10000   # Muestra para calcular silhouette (es lento)
USE_MINIBATCH_KMEANS = True      # Usar MiniBatchKMeans para mayor velocidad


# =============================================================================
# DBSCAN CLUSTERING
# =============================================================================

def run_dbscan(df: pd.DataFrame, eps: float = 0.003, min_samples: int = 50) -> np.ndarray:
    """
    DBSCAN: Density-Based Spatial Clustering of Applications with Noise.
    
    Ventajas:
    - No requiere especificar número de clusters
    - Detecta clusters de forma arbitraria
    - Identifica puntos de ruido (outliers)
    - Ideal para datos geográficos con densidades variables
    
    Parámetros:
    - eps: Radio máximo de vecindad (~0.001 ≈ 100m en Lyon)
    - min_samples: Mínimo de puntos para formar un cluster
    """
    coords = df[['lat', 'long']].values
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    labels = dbscan.fit_predict(coords)
    return labels


def optimize_dbscan(df: pd.DataFrame, 
                    eps_range: List[float] = None,
                    min_samples_range: List[int] = None,
                    verbose: bool = True) -> Tuple[dict, np.ndarray]:
    """
    Optimiza los parámetros de DBSCAN usando el método del codo y métricas de evaluación.
    """
    if eps_range is None:
        eps_range = [0.001, 0.002, 0.003, 0.004, 0.005]
    if min_samples_range is None:
        min_samples_range = [20, 50, 100, 150, 200]
    
    coords = df[['lat', 'long']].values
    
    # Método del codo para estimar eps óptimo
    if verbose:
        print("    ⏳ Calculando eps óptimo con k-distance...")
    
    start_time = time.time()
    k = 50  # min_samples típico
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    k_distances = np.sort(distances[:, k-1])
    
    # Encontrar el "codo" usando la segunda derivada
    gradients = np.gradient(k_distances)
    elbow_idx = np.argmax(gradients > np.mean(gradients) * 2)
    suggested_eps = k_distances[elbow_idx] if elbow_idx > 0 else 0.003
    
    if verbose:
        print(f"    ✓ Eps sugerido: {suggested_eps:.4f} ({time.time()-start_time:.1f}s)")
    
    best_score = -1
    best_params = {'eps': 0.003, 'min_samples': 50}
    best_labels = None
    results = []
    
    total_combos = len(eps_range) * len(min_samples_range)
    if verbose:
        print(f"    ⏳ Probando {total_combos} combinaciones de parámetros...")
    
    combo_count = 0
    for eps in eps_range:
        for min_samples in min_samples_range:
            combo_count += 1
            start_combo = time.time()
            
            if verbose:
                print(f"      [{combo_count}/{total_combos}] eps={eps}, min_samples={min_samples}...", end=" ", flush=True)
            
            labels = run_dbscan(df, eps=eps, min_samples=min_samples)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            noise_pct = 100 * n_noise / len(labels)
            
            # Solo evaluar si hay al menos 2 clusters y menos del 95% es ruido
            if n_clusters >= 2 and noise_pct < 95:
                # Usar solo puntos no-ruido para silhouette (con muestra para rapidez)
                mask = labels >= 0
                n_valid = mask.sum()
                if n_valid > 100:
                    try:
                        # Usar muestra para silhouette si hay muchos puntos
                        if n_valid > MAX_SAMPLES_SILHOUETTE:
                            sample_idx = np.random.choice(np.where(mask)[0], MAX_SAMPLES_SILHOUETTE, replace=False)
                            sil_score = silhouette_score(coords[sample_idx], labels[sample_idx])
                        else:
                            sil_score = silhouette_score(coords[mask], labels[mask])
                    except:
                        sil_score = -1
                else:
                    sil_score = -1
                
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'noise_pct': noise_pct,
                    'silhouette': sil_score
                })
                
                # Balancear: buen silhouette + número razonable de clusters + ruido controlado
                combined_score = sil_score * (1 - noise_pct/100) * min(1, n_clusters/10)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_params = {'eps': eps, 'min_samples': min_samples}
                    best_labels = labels.copy()
                
                if verbose:
                    print(f"clusters={n_clusters}, ruido={noise_pct:.1f}%, sil={sil_score:.3f} ({time.time()-start_combo:.1f}s)")
            else:
                if verbose:
                    print(f"clusters={n_clusters}, ruido={noise_pct:.1f}% (descartado) ({time.time()-start_combo:.1f}s)")
    
    if verbose:
        print(f"    ✓ Mejores parámetros: eps={best_params['eps']}, min_samples={best_params['min_samples']}")
    
    return best_params, best_labels, results


# =============================================================================
# K-MEANS CLUSTERING
# =============================================================================

def run_kmeans(df: pd.DataFrame, n_clusters: int = 20, random_state: int = 42) -> np.ndarray:
    """
    K-Means: Clustering por partición basado en centroides.
    
    Ventajas:
    - Rápido y escalable
    - Fácil de interpretar
    - Buenos resultados con clusters esféricos
    
    Desventajas:
    - Requiere especificar K (número de clusters)
    - Sensible a outliers
    - Asume clusters de tamaño similar
    
    Parámetros:
    - n_clusters: Número de clusters a formar
    """
    coords = df[['lat', 'long']].values
    
    # Escalar coordenadas para mejor convergencia
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    # Usar MiniBatchKMeans para datasets grandes (MUCHO más rápido)
    if USE_MINIBATCH_KMEANS and len(coords) > 10000:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, 
                                  n_init=3, max_iter=100, batch_size=1024)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300)
    
    labels = kmeans.fit_predict(coords_scaled)
    
    return labels


def optimize_kmeans(df: pd.DataFrame,
                    k_range: List[int] = None,
                    verbose: bool = True) -> Tuple[dict, np.ndarray]:
    """
    Optimiza K usando el método del codo y silhouette score.
    """
    if k_range is None:
        k_range = list(range(5, 51, 5))  # 5, 10, 15, ..., 50
    
    coords = df[['lat', 'long']].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    # Usar muestra para evaluación si dataset es grande
    if len(coords) > MAX_SAMPLES_SILHOUETTE:
        np.random.seed(42)
        sample_idx = np.random.choice(len(coords), MAX_SAMPLES_SILHOUETTE, replace=False)
        coords_eval = coords_scaled[sample_idx]
        if verbose:
            print(f"    ℹ️  Usando muestra de {MAX_SAMPLES_SILHOUETTE} puntos para evaluación")
    else:
        coords_eval = coords_scaled
        sample_idx = None
    
    inertias = []
    silhouettes = []
    calinski_scores = []
    
    if verbose:
        print(f"    ⏳ Probando K de {k_range[0]} a {k_range[-1]}...")
    
    for i, k in enumerate(k_range):
        start = time.time()
        if verbose:
            print(f"      [{i+1}/{len(k_range)}] K={k}...", end=" ", flush=True)
        
        # Usar MiniBatchKMeans para rapidez
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=1024)
        labels_full = kmeans.fit_predict(coords_scaled)
        
        inertias.append(kmeans.inertia_)
        
        # Calcular silhouette solo en muestra
        if sample_idx is not None:
            labels_sample = labels_full[sample_idx]
            sil = silhouette_score(coords_eval, labels_sample)
            cal = calinski_harabasz_score(coords_eval, labels_sample)
        else:
            sil = silhouette_score(coords_scaled, labels_full)
            cal = calinski_harabasz_score(coords_scaled, labels_full)
        
        silhouettes.append(sil)
        calinski_scores.append(cal)
        
        if verbose:
            print(f"silhouette={sil:.3f}, calinski={cal:.0f} ({time.time()-start:.1f}s)")
    
    # Encontrar el codo usando la segunda derivada de inertia
    inertia_diff = np.diff(inertias)
    inertia_diff2 = np.diff(inertia_diff)
    elbow_idx = np.argmax(inertia_diff2) + 1 if len(inertia_diff2) > 0 else len(k_range) // 2
    
    # También considerar el mejor silhouette
    best_sil_idx = np.argmax(silhouettes)
    
    # Compromiso entre codo y silhouette
    optimal_k = k_range[max(elbow_idx, best_sil_idx)]
    
    if verbose:
        print(f"    ✓ K óptimo por método del codo: {k_range[elbow_idx]}")
        print(f"    ✓ K óptimo por silhouette: {k_range[best_sil_idx]}")
        print(f"    ✓ K seleccionado: {optimal_k}")
    
    best_labels = run_kmeans(df, n_clusters=optimal_k)
    
    results = {
        'k_range': k_range,
        'inertias': inertias,
        'silhouettes': silhouettes,
        'calinski_scores': calinski_scores,
        'elbow_k': k_range[elbow_idx],
        'best_silhouette_k': k_range[best_sil_idx]
    }
    
    return {'n_clusters': optimal_k}, best_labels, results


# =============================================================================
# HIERARCHICAL CLUSTERING
# =============================================================================

def run_hierarchical(df: pd.DataFrame, 
                     n_clusters: int = 20,
                     linkage: str = 'ward',
                     sample_size: int = None) -> np.ndarray:
    """
    Clustering Jerárquico Aglomerativo.
    
    Ventajas:
    - No requiere especificar K inicialmente (se puede cortar el dendrograma)
    - Produce una jerarquía de clusters (dendrograma)
    - Diferentes criterios de enlace para diferentes formas de clusters
    
    Desventajas:
    - Computacionalmente costoso O(n²) o O(n³)
    - No escala bien con grandes datasets
    
    Parámetros:
    - n_clusters: Número de clusters
    - linkage: 'ward' (minimiza varianza), 'complete', 'average', 'single'
    """
    coords = df[['lat', 'long']].values
    
    # Para datasets grandes, usar una muestra
    if sample_size and len(coords) > sample_size:
        np.random.seed(42)
        sample_idx = np.random.choice(len(coords), sample_size, replace=False)
        coords_sample = coords[sample_idx]
        
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords_sample)
        
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        sample_labels = agg.fit_predict(coords_scaled)
        
        # Asignar puntos no muestreados al cluster más cercano
        labels = np.full(len(coords), -1)
        labels[sample_idx] = sample_labels
        
        # Para los puntos no muestreados, encontrar el centroide más cercano
        centroids = []
        for c in range(n_clusters):
            mask = sample_labels == c
            if mask.sum() > 0:
                centroids.append(coords_sample[mask].mean(axis=0))
        centroids = np.array(centroids)
        
        non_sample_mask = labels == -1
        if non_sample_mask.sum() > 0 and len(centroids) > 0:
            from sklearn.metrics import pairwise_distances_argmin
            labels[non_sample_mask] = pairwise_distances_argmin(
                coords[non_sample_mask], centroids
            )
    else:
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = agg.fit_predict(coords_scaled)
    
    return labels


def optimize_hierarchical(df: pd.DataFrame,
                          k_range: List[int] = None,
                          linkages: List[str] = None,
                          sample_size: int = None,
                          verbose: bool = True) -> Tuple[dict, np.ndarray]:
    """
    Optimiza parámetros del clustering jerárquico.
    """
    if k_range is None:
        k_range = list(range(10, 41, 10))  # Menos valores para rapidez: 10, 20, 30, 40
    if linkages is None:
        linkages = ['ward', 'average']  # Solo 2 linkages para rapidez
    if sample_size is None:
        sample_size = MAX_SAMPLES_HIERARCHICAL
    
    coords = df[['lat', 'long']].values
    
    # SIEMPRE usar muestra para hierarchical
    if len(coords) > sample_size:
        if verbose:
            print(f"  Usando muestra de {sample_size} puntos para optimización...")
        np.random.seed(42)
        sample_idx = np.random.choice(len(coords), sample_size, replace=False)
        coords_eval = coords[sample_idx]
    else:
        coords_eval = coords
    
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords_eval)
    
    best_score = -1
    best_params = {'n_clusters': 20, 'linkage': 'ward'}
    results = []
    
    total_combos = len(k_range) * len(linkages)
    if verbose:
        print(f"    ⏳ Probando {total_combos} combinaciones...")
    
    combo_count = 0
    for linkage in linkages:
        for k in k_range:
            combo_count += 1
            try:
                start = time.time()
                if verbose:
                    print(f"      [{combo_count}/{total_combos}] linkage={linkage}, K={k}...", end=" ", flush=True)
                
                agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                labels_sample = agg.fit_predict(coords_scaled)
                
                sil = silhouette_score(coords_scaled, labels_sample)
                ch = calinski_harabasz_score(coords_scaled, labels_sample)
                db = davies_bouldin_score(coords_scaled, labels_sample)
                
                results.append({
                    'linkage': linkage,
                    'n_clusters': k,
                    'silhouette': sil,
                    'calinski_harabasz': ch,
                    'davies_bouldin': db
                })
                
                if verbose:
                    print(f"silhouette={sil:.3f} ({time.time()-start:.1f}s)")
                
                if sil > best_score:
                    best_score = sil
                    best_params = {'n_clusters': k, 'linkage': linkage}
            except Exception as e:
                if verbose:
                    print(f"Error: {e}")
                continue
    
    if verbose:
        print(f"    ✓ Mejores parámetros: k={best_params['n_clusters']}, linkage={best_params['linkage']}")
        print(f"    ✓ Mejor silhouette: {best_score:.4f}")
    
    # Ejecutar con mejores parámetros en dataset completo
    if verbose:
        print(f"    ⏳ Ejecutando con parámetros óptimos en datos completos...")
    best_labels = run_hierarchical(df, **best_params, sample_size=sample_size)
    
    return best_params, best_labels, results


# =============================================================================
# ANÁLISIS Y COMPARACIÓN
# =============================================================================

def analyze_clusters(df: pd.DataFrame, labels: np.ndarray, algorithm_name: str = "Clustering") -> dict:
    """Analiza los resultados de cualquier algoritmo de clustering."""
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
    
    # Calcular métricas de evaluación
    coords = df[['lat', 'long']].values
    metrics = {}
    
    # Para métricas, excluir ruido si existe
    if n_noise > 0:
        mask = labels >= 0
        if mask.sum() > 100 and n_clusters >= 2:
            try:
                metrics['silhouette'] = silhouette_score(coords[mask], labels[mask])
                metrics['calinski_harabasz'] = calinski_harabasz_score(coords[mask], labels[mask])
                metrics['davies_bouldin'] = davies_bouldin_score(coords[mask], labels[mask])
            except:
                pass
    elif n_clusters >= 2:
        try:
            metrics['silhouette'] = silhouette_score(coords, labels)
            metrics['calinski_harabasz'] = calinski_harabasz_score(coords, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(coords, labels)
        except:
            pass
    
    return {
        'algorithm': algorithm_name,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'n_clustered': n_clustered,
        'noise_pct': 100 * n_noise / len(labels) if len(labels) > 0 else 0,
        'clusters': clusters,
        'metrics': metrics
    }


def compare_algorithms(df: pd.DataFrame, 
                       optimize: bool = True,
                       verbose: bool = True) -> Dict[str, dict]:
    """
    Ejecuta y compara los tres algoritmos de clustering.
    
    Retorna un diccionario con resultados de cada algoritmo.
    """
    results = {}
    total_start = time.time()
    
    print("\n" + "=" * 70)
    print("COMPARACIÓN DE ALGORITMOS DE CLUSTERING")
    print(f"  📊 Dataset: {len(df):,} puntos")
    print("=" * 70)
    
    # 1. DBSCAN
    print("\n" + "-" * 70)
    print("[1/3] 🔵 DBSCAN (Density-Based)")
    print("-" * 70)
    dbscan_start = time.time()
    if optimize:
        print("  ⏳ Optimizando parámetros...")
        params, labels, opt_results = optimize_dbscan(df, verbose=verbose)
    else:
        params = {'eps': 0.003, 'min_samples': 50}
        print(f"  Usando parámetros por defecto: {params}")
        labels = run_dbscan(df, **params)
        opt_results = None
    
    analysis = analyze_clusters(df, labels, "DBSCAN")
    analysis['params'] = params
    analysis['optimization'] = opt_results
    analysis['labels'] = labels
    results['dbscan'] = analysis
    print(f"  ✅ DBSCAN completado en {time.time()-dbscan_start:.1f}s")
    print(f"     Clusters: {analysis['n_clusters']}, Ruido: {analysis['noise_pct']:.1f}%")
    
    # 2. K-Means
    print("\n" + "-" * 70)
    print("[2/3] 🟢 K-Means (Partición por centroides)")
    print("-" * 70)
    kmeans_start = time.time()
    if optimize:
        print("  ⏳ Optimizando K...")
        params, labels, opt_results = optimize_kmeans(df, verbose=verbose)
    else:
        params = {'n_clusters': 20}
        print(f"  Usando parámetros por defecto: {params}")
        labels = run_kmeans(df, **params)
        opt_results = None
    
    analysis = analyze_clusters(df, labels, "K-Means")
    analysis['params'] = params
    analysis['optimization'] = opt_results
    analysis['labels'] = labels
    results['kmeans'] = analysis
    print(f"  ✅ K-Means completado en {time.time()-kmeans_start:.1f}s")
    print(f"     Clusters: {analysis['n_clusters']}")
    
    # 3. Hierarchical
    print("\n" + "-" * 70)
    print("[3/3] 🟣 Clustering Jerárquico (Aglomerativo)")
    print("-" * 70)
    hier_start = time.time()
    if optimize:
        print("  ⏳ Optimizando parámetros...")
        params, labels, opt_results = optimize_hierarchical(df, verbose=verbose)
    else:
        params = {'n_clusters': 20, 'linkage': 'ward'}
        print(f"  Usando parámetros por defecto: {params}")
        labels = run_hierarchical(df, **params)
        opt_results = None
    
    analysis = analyze_clusters(df, labels, "Hierarchical")
    analysis['params'] = params
    analysis['optimization'] = opt_results
    analysis['labels'] = labels
    results['hierarchical'] = analysis
    print(f"  ✅ Hierarchical completado en {time.time()-hier_start:.1f}s")
    print(f"     Clusters: {analysis['n_clusters']}")
    
    print("\n" + "=" * 70)
    print(f"✅ COMPARACIÓN COMPLETADA en {time.time()-total_start:.1f}s")
    print("=" * 70)
    
    return results

def print_comparison(results: Dict[str, dict]) -> None:
    """Imprime una tabla comparativa de los resultados."""
    print("\n" + "=" * 70)
    print("RESUMEN COMPARATIVO")
    print("=" * 70)
    
    # Tabla de métricas
    print(f"\n{'Algoritmo':<20} {'Clusters':<10} {'Ruido %':<10} {'Silhouette':<12} {'Calinski-H':<12} {'Davies-B':<10}")
    print("-" * 74)
    
    for name, data in results.items():
        metrics = data.get('metrics', {})
        sil = f"{metrics.get('silhouette', 0):.4f}" if 'silhouette' in metrics else "N/A"
        ch = f"{metrics.get('calinski_harabasz', 0):.1f}" if 'calinski_harabasz' in metrics else "N/A"
        db = f"{metrics.get('davies_bouldin', 0):.4f}" if 'davies_bouldin' in metrics else "N/A"
        
        print(f"{data['algorithm']:<20} {data['n_clusters']:<10} {data['noise_pct']:<10.1f} {sil:<12} {ch:<12} {db:<10}")
    
    print("-" * 74)
    
    # Interpretación de métricas
    print("\n📊 INTERPRETACIÓN DE MÉTRICAS:")
    print("  • Silhouette: [-1, 1] Mayor es mejor (cohesión y separación)")
    print("  • Calinski-Harabasz: Mayor es mejor (varianza entre/dentro clusters)")
    print("  • Davies-Bouldin: Menor es mejor (similitud entre clusters)")
    
    # Recomendación
    print("\n💡 ANÁLISIS:")
    
    # Encontrar el mejor por silhouette
    best_sil = max(results.items(), 
                   key=lambda x: x[1].get('metrics', {}).get('silhouette', -1))
    print(f"  • Mejor silhouette: {best_sil[1]['algorithm']}")
    
    # DBSCAN es especial por detectar ruido
    if 'dbscan' in results and results['dbscan']['n_noise'] > 0:
        print(f"  • DBSCAN detectó {results['dbscan']['n_noise']:,} puntos de ruido ({results['dbscan']['noise_pct']:.1f}%)")
        print("    → Útil para identificar fotos aisladas que no pertenecen a áreas de interés")
    
    print("\n" + "=" * 70)


def print_results(analysis: dict) -> None:
    """Imprime resultados detallados de un algoritmo."""
    print("\n" + "=" * 60)
    print(f"{analysis['algorithm'].upper()} - RESULTADOS")
    print("=" * 60)
    print(f"Clusters encontrados: {analysis['n_clusters']}")
    print(f"Puntos agrupados: {analysis['n_clustered']:,}")
    print(f"Puntos de ruido: {analysis['n_noise']:,} ({analysis['noise_pct']:.1f}%)")
    
    if analysis.get('metrics'):
        print(f"\nMétricas:")
        for metric, value in analysis['metrics'].items():
            print(f"  {metric}: {value:.4f}")
    
    print("\nTop 10 clusters:")
    print("-" * 60)
    
    for c in analysis['clusters'][:10]:
        tags = ", ".join([f"{t[0]}({t[1]})" for t in c['top_tags'][:3]])
        print(f"  Cluster {c['id']}: {c['size']:,} fotos, {c['users']} usuarios")
        print(f"    Centro: ({c['center'][0]:.4f}, {c['center'][1]:.4f})")
        print(f"    Tags: {tags}")
    
    print("=" * 60)


def get_top_tags(df: pd.DataFrame, n: int = 5) -> list:
    """Obtiene los tags más frecuentes de un DataFrame."""
    all_tags = []
    for tags_str in df['tags'].dropna():
        if tags_str:
            tags = [t.strip() for t in str(tags_str).split(',') if t.strip()]
            all_tags.extend(tags)
    
    return Counter(all_tags).most_common(n)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from data_loader import load_flickr_data
    from data_cleaning import clean_data
    from pathlib import Path
    
    print("Cargando y limpiando datos...")
    df = load_flickr_data(Path(__file__).parent.parent / "data" / "flickr_data2.csv")
    df_clean, _ = clean_data(df, verbose=False)
    
    print(f"\nDataset: {len(df_clean):,} puntos")
    
    # Comparar los tres algoritmos
    results = compare_algorithms(df_clean, optimize=True, verbose=True)
    
    # Imprimir comparación
    print_comparison(results)
    
    # Imprimir detalles de cada uno
    for name, analysis in results.items():
        print_results(analysis)

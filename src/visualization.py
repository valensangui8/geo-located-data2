"""
Interactive Map Visualization for Flickr Lyon Data with Clustering
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from pathlib import Path
from collections import Counter
import json

LYON_CENTER = (45.7640, 4.8357)

CLUSTER_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
    '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62', '#8da0cb',
    '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3', '#1b9e77'
]

SKIP_WORDS = {'lyon', 'france', 'europe', 'photo', 'square', 'squareformat', 
              'iphoneography', 'instagram', 'iphone', 'photography', 'flickr'}


def create_map(df: pd.DataFrame, 
               labels_dbscan: np.ndarray = None,
               labels_hdbscan: np.ndarray = None,
               output_path: str = None) -> folium.Map:
    """Create interactive map with both DBSCAN and HDBSCAN clusters."""
    
    m = folium.Map(location=LYON_CENTER, zoom_start=13, tiles='cartodbpositron')
    
    if labels_dbscan is not None and labels_hdbscan is not None:
        _add_dual_clusters(m, df, labels_dbscan, labels_hdbscan)
    elif labels_hdbscan is not None:
        _add_single_algorithm(m, df, labels_hdbscan, "HDBSCAN")
    elif labels_dbscan is not None:
        _add_single_algorithm(m, df, labels_dbscan, "DBSCAN")
    else:
        _add_simple_markers(m, df)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(output_path)
        print(f"Map saved to: {output_path}")
    
    return m


def _get_cluster_name(df: pd.DataFrame) -> str:
    """Generate a name for a cluster based on its most distinctive tag."""
    all_tags = []
    for tags_str in df['tags'].dropna():
        if tags_str:
            tags = [t.strip().lower() for t in str(tags_str).split(',') if t.strip()]
            all_tags.extend(tags)
    
    if not all_tags:
        return "Unknown"
    
    tag_counts = Counter(all_tags)
    
    for tag, count in tag_counts.most_common(10):
        if tag.lower() not in SKIP_WORDS and len(tag) > 2:
            return tag.title()
    
    return tag_counts.most_common(1)[0][0].title() if tag_counts else "Unknown"


def _get_top_tags(df: pd.DataFrame, n: int = 5) -> list:
    """Get top n tags from dataframe."""
    all_tags = []
    for tags_str in df['tags'].dropna():
        if tags_str:
            tags = [t.strip() for t in str(tags_str).split(',') if t.strip()]
            all_tags.extend(tags)
    return Counter(all_tags).most_common(n)


def _prepare_cluster_data(df: pd.DataFrame, labels: np.ndarray, algorithm: str):
    """Prepare cluster data for JavaScript."""
    df_work = df.copy()
    df_work['cluster'] = labels
    
    unique_clusters = sorted([c for c in set(labels) if c >= 0], 
                            key=lambda x: (labels == x).sum(), reverse=True)
    
    clusters_data = {}
    markers_data = []
    
    for idx, cluster_id in enumerate(unique_clusters[:25]):
        mask = labels == cluster_id
        cluster_df = df_work[mask]
        color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
        
        cluster_name = _get_cluster_name(cluster_df)
        center_lat = cluster_df['lat'].mean()
        center_long = cluster_df['long'].mean()
        size = len(cluster_df)
        users = cluster_df['user'].nunique()
        
        top_tags = _get_top_tags(cluster_df, 5)
        tags_html = "<br>".join([f"• {t[0]} ({t[1]})" for t in top_tags]) if top_tags else "N/A"
        
        # Sample photos
        if len(cluster_df) > 300:
            sample_df = cluster_df.sample(n=300, random_state=42)
        else:
            sample_df = cluster_df
        
        photos = []
        for _, row in sample_df.iterrows():
            photos.append({
                'lat': float(row['lat']),
                'lng': float(row['long']),
                'id': str(row['id']),
                'user': str(row['user']),
                'date': f"{int(row['date_taken_year'])}-{int(row['date_taken_month']):02d}-{int(row['date_taken_day']):02d}",
                'tags': str(row['tags'])[:80],
                'color': color
            })
        
        clusters_data[str(cluster_id)] = {
            'photos': photos,
            'center': [float(center_lat), float(center_long)],
            'color': color,
            'name': cluster_name
        }
        
        markers_data.append({
            'id': int(cluster_id),
            'name': cluster_name,
            'lat': float(center_lat),
            'lng': float(center_long),
            'size': int(size),
            'users': int(users),
            'color': color,
            'tags_html': tags_html,
            'radius': float(min(25, max(10, size / 30)))
        })
    
    return clusters_data, markers_data


def _add_dual_clusters(m: folium.Map, df: pd.DataFrame, 
                       labels_dbscan: np.ndarray, labels_hdbscan: np.ndarray) -> None:
    """Add both DBSCAN and HDBSCAN clusters with toggle."""
    
    # Prepare data for both algorithms
    dbscan_clusters, dbscan_markers = _prepare_cluster_data(df, labels_dbscan, "DBSCAN")
    hdbscan_clusters, hdbscan_markers = _prepare_cluster_data(df, labels_hdbscan, "HDBSCAN")
    
    # Add JavaScript with all data
    js_code = f"""
    <script>
    var dbscanClusters = {json.dumps(dbscan_clusters)};
    var hdbscanClusters = {json.dumps(hdbscan_clusters)};
    var dbscanMarkers = {json.dumps(dbscan_markers)};
    var hdbscanMarkers = {json.dumps(hdbscan_markers)};
    
    var currentAlgorithm = 'hdbscan';
    var currentMarkers = [];
    var clusterCenterMarkers = [];
    var map = null;
    
    document.addEventListener('DOMContentLoaded', function() {{
        setTimeout(function() {{
            map = Object.values(window).find(v => v && v instanceof L.Map);
            if (map) {{
                showAlgorithm('hdbscan');
            }}
        }}, 500);
    }});
    
    function showAlgorithm(algo) {{
        if (!map) return;
        
        currentAlgorithm = algo;
        
        // Clear all markers
        currentMarkers.forEach(m => map.removeLayer(m));
        clusterCenterMarkers.forEach(m => map.removeLayer(m));
        currentMarkers = [];
        clusterCenterMarkers = [];
        
        // Update button styles
        document.getElementById('btn-dbscan').className = algo === 'dbscan' ? 'algo-btn active' : 'algo-btn';
        document.getElementById('btn-hdbscan').className = algo === 'hdbscan' ? 'algo-btn active' : 'algo-btn';
        document.getElementById('cluster-title').style.display = 'none';
        
        // Get data for selected algorithm
        var markers = algo === 'dbscan' ? dbscanMarkers : hdbscanMarkers;
        
        // Add cluster centers
        markers.forEach(function(c) {{
            var popupContent = '<div style="font-family: Arial; width: 240px;">' +
                '<h3 style="margin: 0 0 5px 0; color: ' + c.color + ';">📍 ' + c.name + '</h3>' +
                '<p style="margin: 5px 0; color: #666; font-size: 12px;">Cluster #' + c.id + '</p>' +
                '<hr style="margin: 10px 0; border: none; border-top: 1px solid #eee;">' +
                '<p style="margin: 5px 0;"><b>📷 Photos:</b> ' + c.size.toLocaleString() + '</p>' +
                '<p style="margin: 5px 0;"><b>👥 Users:</b> ' + c.users + '</p>' +
                '<p style="margin: 5px 0;"><b>🏷️ Top tags:</b></p>' +
                '<p style="margin: 0 0 10px 10px; font-size: 11px;">' + c.tags_html + '</p>' +
                '<button onclick="showCluster(' + c.id + ')" ' +
                'style="background: ' + c.color + '; color: white; padding: 10px 15px; ' +
                'border: none; border-radius: 4px; cursor: pointer; width: 100%; font-size: 14px;">' +
                '🔍 Explore photos</button></div>';
            
            var marker = L.circleMarker([c.lat, c.lng], {{
                radius: c.radius,
                color: c.color,
                fillColor: c.color,
                fillOpacity: 0.7,
                weight: 3
            }}).bindPopup(popupContent);
            
            marker.addTo(map);
            clusterCenterMarkers.push(marker);
        }});
        
        // Update stats
        var totalClusters = markers.length;
        document.getElementById('stats-text').textContent = algo.toUpperCase() + ': ' + totalClusters + ' clusters shown';
        
        map.setView([45.7640, 4.8357], 13);
    }}
    
    function showCluster(clusterId) {{
        if (!map) return;
        
        // Clear photo markers only
        currentMarkers.forEach(m => map.removeLayer(m));
        currentMarkers = [];
        
        var clusters = currentAlgorithm === 'dbscan' ? dbscanClusters : hdbscanClusters;
        var data = clusters[clusterId];
        if (!data) return;
        
        document.getElementById('cluster-title').textContent = '📍 ' + data.name;
        document.getElementById('cluster-title').style.display = 'block';
        
        map.setView(data.center, 16);
        
        data.photos.forEach(function(photo) {{
            var popupContent = '<div style="font-family: Arial; width: 240px;">' +
                '<p style="margin: 5px 0; color: #666; font-size: 11px;">Part of: ' + data.name + '</p>' +
                '<hr style="margin: 8px 0; border: none; border-top: 1px solid #eee;">' +
                '<p><b>📷 ID:</b> ' + photo.id + '</p>' +
                '<p><b>👤 User:</b> ' + photo.user + '</p>' +
                '<p><b>📅 Date:</b> ' + photo.date + '</p>' +
                '<p><b>🏷️ Tags:</b> ' + photo.tags + '</p>' +
                '<a href="https://www.flickr.com/photos/' + photo.user + '/' + photo.id + '" ' +
                'target="_blank" style="display: block; background: #0063dc; color: white; ' +
                'padding: 10px; text-align: center; text-decoration: none; border-radius: 4px; margin-top: 10px;">' +
                '🔗 View on Flickr</a></div>';
            
            var marker = L.circleMarker([photo.lat, photo.lng], {{
                radius: 8,
                color: photo.color,
                fillColor: photo.color,
                fillOpacity: 0.8,
                weight: 2
            }}).bindPopup(popupContent);
            
            marker.addTo(map);
            currentMarkers.push(marker);
        }});
    }}
    
    function clearPhotos() {{
        currentMarkers.forEach(m => map.removeLayer(m));
        currentMarkers = [];
        document.getElementById('cluster-title').style.display = 'none';
        if (map) map.setView([45.7640, 4.8357], 13);
    }}
    </script>
    
    <style>
    .control-panel {{
        position: fixed;
        top: 10px;
        left: 60px;
        z-index: 1000;
        display: flex;
        gap: 10px;
        align-items: center;
    }}
    .algo-selector {{
        background: white;
        padding: 5px;
        border-radius: 6px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        display: flex;
        gap: 5px;
    }}
    .algo-btn {{
        padding: 8px 16px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-family: Arial;
        font-size: 13px;
        font-weight: bold;
        background: #f0f0f0;
        color: #333;
        transition: all 0.2s;
    }}
    .algo-btn:hover {{
        background: #e0e0e0;
    }}
    .algo-btn.active {{
        background: #0063dc;
        color: white;
    }}
    .stats-box {{
        background: white;
        padding: 8px 12px;
        border-radius: 4px;
        font-family: Arial;
        font-size: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
    .right-panel {{
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 1000;
        display: flex;
        gap: 10px;
        align-items: center;
    }}
    .cluster-title {{
        background: white;
        padding: 8px 15px;
        border-radius: 4px;
        font-family: Arial;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        display: none;
    }}
    .clear-btn {{
        background: white;
        border: 2px solid #ccc;
        padding: 8px 15px;
        border-radius: 4px;
        cursor: pointer;
        font-family: Arial;
        font-size: 14px;
    }}
    .clear-btn:hover {{
        background: #f0f0f0;
    }}
    </style>
    
    <div class="control-panel">
        <div class="algo-selector">
            <button id="btn-dbscan" class="algo-btn" onclick="showAlgorithm('dbscan')">DBSCAN</button>
            <button id="btn-hdbscan" class="algo-btn active" onclick="showAlgorithm('hdbscan')">HDBSCAN</button>
        </div>
        <div class="stats-box">
            <span id="stats-text">HDBSCAN: 25 clusters shown</span>
        </div>
    </div>
    
    <div class="right-panel">
        <div id="cluster-title" class="cluster-title"></div>
        <button class="clear-btn" onclick="clearPhotos()">✕ Back</button>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(js_code))


def _add_single_algorithm(m: folium.Map, df: pd.DataFrame, labels: np.ndarray, algorithm: str) -> None:
    """Add clusters for a single algorithm (fallback)."""
    clusters_data, markers_data = _prepare_cluster_data(df, labels, algorithm)
    
    for c in markers_data:
        popup_html = f"""
        <div style="font-family: Arial; width: 240px;">
            <h3 style="margin: 0 0 5px 0; color: {c['color']};">📍 {c['name']}</h3>
            <p style="margin: 5px 0;"><b>📷 Photos:</b> {c['size']:,}</p>
            <p style="margin: 5px 0;"><b>👥 Users:</b> {c['users']}</p>
        </div>
        """
        folium.CircleMarker(
            location=[c['lat'], c['lng']],
            radius=c['radius'],
            color=c['color'],
            fill=True,
            fillColor=c['color'],
            fillOpacity=0.7,
            weight=3,
            popup=folium.Popup(popup_html, max_width=260)
        ).add_to(m)


def _add_simple_markers(m: folium.Map, df: pd.DataFrame) -> None:
    """Add simple marker cluster without clustering labels."""
    sample = df.sample(n=min(5000, len(df)), random_state=42)
    marker_cluster = MarkerCluster(name='Photos')
    
    for _, row in sample.iterrows():
        flickr_url = f"https://www.flickr.com/photos/{row['user']}/{row['id']}"
        popup_html = f"""
        <div style="width: 220px;">
            <p><b>ID:</b> {row['id']}</p>
            <p><b>User:</b> {row['user']}</p>
            <a href="{flickr_url}" target="_blank">View on Flickr</a>
        </div>
        """
        folium.Marker(
            location=[row['lat'], row['long']],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color='blue', icon='camera', prefix='fa')
        ).add_to(marker_cluster)
    
    marker_cluster.add_to(m)


if __name__ == "__main__":
    from data_loader import load_flickr_data
    from data_cleaning import clean_data
    from clustering import run_dbscan, run_hdbscan, filter_single_user_clusters
    
    print("Loading and cleaning data...")
    df = load_flickr_data(Path(__file__).parent.parent / "data" / "flickr_data2.csv")
    df_clean, _ = clean_data(df, verbose=False)
    
    print("Running DBSCAN...")
    labels_dbscan = run_dbscan(df_clean, eps_meters=80, min_samples=8)
    labels_dbscan = filter_single_user_clusters(df_clean, labels_dbscan)
    
    print("Running HDBSCAN...")
    labels_hdbscan = run_hdbscan(df_clean)
    labels_hdbscan = filter_single_user_clusters(df_clean, labels_hdbscan)
    
    print("Creating map...")
    create_map(df_clean, labels_dbscan=labels_dbscan, labels_hdbscan=labels_hdbscan, 
               output_path="outputs/lyon_map.html")
    print("Done!")

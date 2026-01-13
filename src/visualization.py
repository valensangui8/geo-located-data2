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


def create_map(df: pd.DataFrame, labels: np.ndarray = None, output_path: str = None) -> folium.Map:
    m = folium.Map(location=LYON_CENTER, zoom_start=13, tiles='cartodbpositron')
    
    if labels is not None:
        _add_interactive_clusters(m, df, labels)
    else:
        _add_simple_markers(m, df)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(output_path)
        print(f"Map saved to: {output_path}")
    
    return m


def _add_interactive_clusters(m: folium.Map, df: pd.DataFrame, labels: np.ndarray) -> None:
    df_work = df.copy()
    df_work['cluster'] = labels
    
    unique_clusters = sorted([c for c in set(labels) if c >= 0], 
                            key=lambda x: (labels == x).sum(), reverse=True)
    
    clusters_data = {}
    
    for idx, cluster_id in enumerate(unique_clusters[:20]):
        mask = labels == cluster_id
        cluster_df = df_work[mask]
        color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
        
        if len(cluster_df) > 300:
            sample_df = cluster_df.sample(n=300, random_state=42)
        else:
            sample_df = cluster_df
        
        photos = []
        for _, row in sample_df.iterrows():
            photos.append({
                'lat': row['lat'],
                'lng': row['long'],
                'id': str(row['id']),
                'user': str(row['user']),
                'date': f"{int(row['date_taken_year'])}-{int(row['date_taken_month']):02d}-{int(row['date_taken_day']):02d}",
                'tags': str(row['tags'])[:80],
                'color': color
            })
        
        clusters_data[str(cluster_id)] = {
            'photos': photos,
            'center': [cluster_df['lat'].mean(), cluster_df['long'].mean()],
            'color': color
        }
    
    for idx, cluster_id in enumerate(unique_clusters[:20]):
        mask = labels == cluster_id
        cluster_df = df[mask]
        
        center_lat = cluster_df['lat'].mean()
        center_long = cluster_df['long'].mean()
        size = len(cluster_df)
        users = cluster_df['user'].nunique()
        color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
        
        top_tags = _get_top_tags(cluster_df, 5)
        tags_html = "<br>".join([f"• {t[0]} ({t[1]})" for t in top_tags]) if top_tags else "N/A"
        
        radius = min(25, max(12, size / 400))
        
        marker = folium.CircleMarker(
            location=[center_lat, center_long],
            radius=radius,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=3
        )
        
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 220px;">
            <h4 style="margin: 0 0 10px 0; color: {color};">Cluster {cluster_id}</h4>
            <p style="margin: 5px 0;"><b>📷 Photos:</b> {size:,}</p>
            <p style="margin: 5px 0;"><b>👥 Users:</b> {users}</p>
            <p style="margin: 5px 0;"><b>🏷️ Top tags:</b></p>
            <p style="margin: 0 0 10px 10px; font-size: 12px;">{tags_html}</p>
            <button onclick="showCluster({cluster_id})" 
                    style="background: {color}; color: white; padding: 8px 15px; 
                           border: none; border-radius: 4px; cursor: pointer; width: 100%;">
                🔍 Show {min(300, size)} photos
            </button>
        </div>
        """
        marker.add_child(folium.Popup(popup_html, max_width=250))
        marker.add_to(m)
    
    js_code = f"""
    <script>
    var clustersData = {json.dumps(clusters_data)};
    var currentMarkers = [];
    var map = null;
    
    document.addEventListener('DOMContentLoaded', function() {{
        setTimeout(function() {{
            var maps = document.querySelectorAll('.folium-map');
            if (maps.length > 0) {{
                map = maps[0]._leaflet_map || Object.values(window).find(v => v instanceof L.Map);
            }}
        }}, 500);
    }});
    
    function showCluster(clusterId) {{
        currentMarkers.forEach(function(m) {{
            if (map) map.removeLayer(m);
        }});
        currentMarkers = [];
        
        var data = clustersData[clusterId];
        if (!data || !map) {{
            map = Object.values(window).find(v => v && v._container && v._container.classList && v._container.classList.contains('folium-map'));
            if (!map) return;
        }}
        
        map.setView(data.center, 16);
        
        data.photos.forEach(function(photo) {{
            var popupContent = '<div style="font-family: Arial; width: 240px;">' +
                '<p><b>📷 ID:</b> ' + photo.id + '</p>' +
                '<p><b>👤 User:</b> ' + photo.user + '</p>' +
                '<p><b>📅 Date:</b> ' + photo.date + '</p>' +
                '<p><b>🏷️ Tags:</b> ' + photo.tags + '</p>' +
                '<a href="https://www.flickr.com/photos/' + photo.user + '/' + photo.id + '" ' +
                'target="_blank" style="display: block; background: #0063dc; color: white; ' +
                'padding: 8px; text-align: center; text-decoration: none; border-radius: 4px; margin-top: 10px;">' +
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
    
    function clearMarkers() {{
        currentMarkers.forEach(function(m) {{
            if (map) map.removeLayer(m);
        }});
        currentMarkers = [];
        if (map) map.setView([45.7640, 4.8357], 13);
    }}
    </script>
    
    <style>
    .clear-btn {{
        position: fixed;
        top: 10px;
        right: 60px;
        z-index: 1000;
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
    
    <button class="clear-btn" onclick="clearMarkers()">✕ Clear photos</button>
    """
    
    m.get_root().html.add_child(folium.Element(js_code))


def _add_simple_markers(m: folium.Map, df: pd.DataFrame) -> None:
    sample = df.sample(n=min(5000, len(df)), random_state=42)
    marker_cluster = MarkerCluster(name='Photos')
    
    for _, row in sample.iterrows():
        flickr_url = f"https://www.flickr.com/photos/{row['user']}/{row['id']}"
        popup_html = f"""
        <div style="width: 220px;">
            <p><b>ID:</b> {row['id']}</p>
            <p><b>User:</b> {row['user']}</p>
            <p><b>Date:</b> {int(row['date_taken_year'])}-{int(row['date_taken_month']):02d}-{int(row['date_taken_day']):02d}</p>
            <p><b>Tags:</b> {str(row['tags'])[:100]}...</p>
            <a href="{flickr_url}" target="_blank">View on Flickr</a>
        </div>
        """
        folium.Marker(
            location=[row['lat'], row['long']],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color='blue', icon='camera', prefix='fa')
        ).add_to(marker_cluster)
    
    marker_cluster.add_to(m)


def _get_top_tags(df: pd.DataFrame, n: int = 5) -> list:
    all_tags = []
    for tags_str in df['tags'].dropna():
        if tags_str:
            tags = [t.strip() for t in str(tags_str).split(',') if t.strip()]
            all_tags.extend(tags)
    return Counter(all_tags).most_common(n)


if __name__ == "__main__":
    from data_loader import load_flickr_data
    from data_cleaning import clean_data
    from clustering import run_dbscan
    
    print("Loading and cleaning data...")
    df = load_flickr_data(Path(__file__).parent.parent / "data" / "flickr_data2.csv")
    df_clean, _ = clean_data(df, verbose=False)
    
    print("Running clustering...")
    labels = run_dbscan(df_clean)
    
    print(f"Creating map with {len(df_clean):,} points...")
    create_map(df_clean, labels=labels, output_path="outputs/lyon_map.html")
    print("Done!")

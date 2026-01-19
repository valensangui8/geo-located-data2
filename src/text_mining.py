import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import Dict, List, Tuple


CUSTOM_STOPWORDS = [
    'lyon', 'france', 'europe', 'french', 'rhone', 'rhône',
    'uploaded', 'photo', 'photos', 'image', 'picture',
    'flickr', 'instagram', 'foursquare', 'facebook',
    'day', 'night', 'morning', 'evening', 'today',
    'beautiful', 'great', 'nice', 'amazing', 'good',
    'city', 'place', 'street', 'rue', 'avenue'
]

FRENCH_STOPWORDS = [
    'alors', 'au', 'aucuns', 'aussi', 'autre', 'avant', 'avec', 'avoir', 'bon',
    'car', 'ce', 'cela', 'ces', 'ceux', 'chaque', 'ci', 'comme', 'comment',
    'dans', 'des', 'du', 'dedans', 'dehors', 'depuis', 'devrait', 'doit', 'donc',
    'dos', 'droite', 'début', 'elle', 'elles', 'en', 'encore', 'essai', 'est',
    'et', 'eu', 'fait', 'faites', 'fois', 'font', 'force', 'haut', 'hors',
    'ici', 'il', 'ils', 'je', 'juste', 'la', 'le', 'les', 'leur', 'là', 'ma',
    'maintenant', 'mais', 'mes', 'mine', 'moins', 'mon', 'mot', 'même', 'ni',
    'nommés', 'notre', 'nous', 'nouveaux', 'ou', 'où', 'par', 'parce', 'parole',
    'pas', 'personnes', 'peut', 'peu', 'pièce', 'plupart', 'pour', 'pourquoi',
    'quand', 'que', 'quel', 'quelle', 'quelles', 'quels', 'qui', 'sa', 'sans',
    'ses', 'seulement', 'si', 'sien', 'son', 'sont', 'sous', 'soyez', 'sujet',
    'sur', 'ta', 'tandis', 'tellement', 'tels', 'tes', 'ton', 'tous', 'tout',
    'trop', 'très', 'tu', 'valeur', 'voie', 'voient', 'vont', 'votre', 'vous',
    'vu', 'ça', 'étaient', 'état', 'étions', 'été', 'être'
]


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-ZÀ-ÖØ-öø-ÿ]+", " ", text)
    return text.strip()


def prepare_cluster_documents(df: pd.DataFrame, labels: np.ndarray,
                              text_cols: Tuple[str, ...] = ('title', 'tags')) -> Dict[int, str]:
    df_work = df.copy()
    df_work['cluster'] = labels
    
    cluster_docs = {}
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        
        cluster_df = df_work[df_work['cluster'] == cluster_id]
        text_parts = []
        for col in text_cols:
            if col in cluster_df.columns:
                text_parts.append(cluster_df[col].fillna('').astype(str))
        if text_parts:
            combined_text = ' '.join(pd.concat(text_parts).tolist())
            cleaned = preprocess_text(combined_text)
            if cleaned:
                cluster_docs[cluster_id] = cleaned
    
    return cluster_docs


def compute_tfidf(cluster_docs: Dict[int, str], top_n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
    if not cluster_docs:
        return {}
    
    docs = list(cluster_docs.values())
    cluster_ids = list(cluster_docs.keys())
    
    english_stopwords = list(TfidfVectorizer(stop_words='english').get_stop_words())
    all_stopwords = english_stopwords + CUSTOM_STOPWORDS + FRENCH_STOPWORDS
    
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words=all_stopwords,
        min_df=1,
        max_df=0.8,
        token_pattern=r'(?u)\b[a-zA-ZÀ-ÖØ-öø-ÿ][a-zA-ZÀ-ÖØ-öø-ÿ]+\b'
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()
        
        cluster_keywords = {}
        for idx, cluster_id in enumerate(cluster_ids):
            scores = tfidf_matrix[idx].toarray().flatten()
            top_indices = scores.argsort()[-top_n:][::-1]
            top_words = [(feature_names[i], float(scores[i])) for i in top_indices if scores[i] > 0]
            cluster_keywords[cluster_id] = top_words
        
        return cluster_keywords
    except Exception as e:
        print(f"Warning: TF-IDF computation failed: {e}")
        return {cid: [] for cid in cluster_ids}


def compute_tf(cluster_docs: Dict[int, str], top_n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
    if not cluster_docs:
        return {}

    docs = list(cluster_docs.values())
    cluster_ids = list(cluster_docs.keys())

    english_stopwords = list(TfidfVectorizer(stop_words='english').get_stop_words())
    all_stopwords = english_stopwords + CUSTOM_STOPWORDS + FRENCH_STOPWORDS

    vectorizer = CountVectorizer(
        stop_words=all_stopwords,
        token_pattern=r'(?u)\b[a-zA-ZÀ-ÖØ-öø-ÿ][a-zA-ZÀ-ÖØ-öø-ÿ]+\b'
    )

    try:
        count_matrix = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()

        cluster_keywords = {}
        for idx, cluster_id in enumerate(cluster_ids):
            counts = count_matrix[idx].toarray().flatten()
            total = counts.sum()
            if total == 0:
                cluster_keywords[cluster_id] = []
                continue
            freqs = counts / total
            top_indices = freqs.argsort()[-top_n:][::-1]
            top_words = [(feature_names[i], float(freqs[i])) for i in top_indices if freqs[i] > 0]
            cluster_keywords[cluster_id] = top_words

        return cluster_keywords
    except Exception as e:
        print(f"Warning: TF computation failed: {e}")
        return {cid: [] for cid in cluster_ids}

def describe_cluster(cluster_id: int, df: pd.DataFrame, 
                     labels: np.ndarray, tfidf_results: Dict[int, List[Tuple[str, float]]]) -> dict:
    mask = labels == cluster_id
    cluster_df = df[mask]
    
    if len(cluster_df) == 0:
        return None
    
    description = {
        'cluster_id': cluster_id,
        'size': len(cluster_df),
        'center': (float(cluster_df['lat'].mean()), float(cluster_df['long'].mean())),
        'users': cluster_df['user'].nunique(),
        'distinctive_words': tfidf_results.get(cluster_id, []),
        'date_range': (
            int(cluster_df['date_taken_year'].min()) if pd.notna(cluster_df['date_taken_year'].min()) else None,
            int(cluster_df['date_taken_year'].max()) if pd.notna(cluster_df['date_taken_year'].max()) else None
        )
    }
    
    return description


def print_tfidf_results(tfidf_results: Dict[int, List[Tuple[str, float]]], 
                       df: pd.DataFrame, labels: np.ndarray, top_clusters: int = 5):
    print("\n" + "=" * 70)
    print("TF-IDF TEXT MINING RESULTS")
    print("=" * 70)


def build_keywords_table(results: Dict[int, List[Tuple[str, float]]],
                         algorithm: str,
                         method: str) -> pd.DataFrame:
    rows = []
    for cluster_id, words in results.items():
        for word, score in words:
            rows.append({
                'algorithm': algorithm,
                'method': method,
                'cluster_id': cluster_id,
                'keyword': word,
                'score': score
            })
    return pd.DataFrame(rows)


def print_tfidf_results(tfidf_results: Dict[int, List[Tuple[str, float]]], 
                       df: pd.DataFrame, labels: np.ndarray, top_clusters: int = 5):
    print("\n" + "=" * 70)
    print("TF-IDF TEXT MINING RESULTS")
    print("=" * 70)
    print(f"Showing distinctive words for top {top_clusters} clusters\n")
    
    cluster_sizes = {}
    for cluster_id in tfidf_results.keys():
        cluster_sizes[cluster_id] = (labels == cluster_id).sum()
    
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    
    for cluster_id, size in sorted_clusters[:top_clusters]:
        if cluster_id in tfidf_results and tfidf_results[cluster_id]:
            print(f"Cluster {cluster_id} ({size:,} photos):")
            words = tfidf_results[cluster_id][:5]
            for word, score in words:
                print(f"  - {word}: {score:.3f}")
            print()
    
    print("=" * 70)

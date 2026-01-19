import pandas as pd
import numpy as np
from typing import Tuple


def add_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_parts = {
        'year': df.get('date_taken_year'),
        'month': df.get('date_taken_month'),
        'day': df.get('date_taken_day')
    }
    df['taken_date'] = pd.to_datetime(date_parts, errors='coerce')
    return df


def compute_cluster_time_series(df: pd.DataFrame, labels: np.ndarray,
                                freq: str = 'MS') -> pd.DataFrame:
    df_work = add_datetime_column(df)
    df_work['cluster'] = labels
    df_work = df_work.dropna(subset=['taken_date'])

    grouped = (
        df_work
        .groupby(['cluster', pd.Grouper(key='taken_date', freq=freq)])
        .agg(photo_count=('id', 'count'), user_count=('user', 'nunique'))
        .reset_index()
        .rename(columns={'taken_date': 'period_start'})
    )
    return grouped


def detect_event_spikes(time_series: pd.DataFrame,
                        z_threshold: float = 3.0,
                        min_count: int = 50) -> pd.DataFrame:
    events = []
    for cluster_id, group in time_series.groupby('cluster'):
        counts = group['photo_count'].astype(float)
        if len(counts) < 4:
            continue
        mean = counts.mean()
        std = counts.std(ddof=0)
        if std == 0:
            continue
        zscores = (counts - mean) / std
        spikes = group[(zscores >= z_threshold) & (group['photo_count'] >= min_count)]
        for _, row in spikes.iterrows():
            events.append({
                'cluster': int(cluster_id),
                'period_start': row['period_start'],
                'photo_count': int(row['photo_count']),
                'user_count': int(row['user_count']),
                'zscore': float((row['photo_count'] - mean) / std)
            })
    return pd.DataFrame(events).sort_values(['zscore'], ascending=False)


def annotate_known_events(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return events_df

    df = events_df.copy()
    df['known_event'] = ''
    df['event_match'] = False

    known_events = [
        {'name': 'Fete des Lumieres', 'months': [12]},
        {'name': 'Nuits de Fourviere', 'months': [6, 7]},
        {'name': 'Festival Lumiere', 'months': [10]},
        {'name': 'Quais du Polar', 'months': [3, 4]},
        {'name': 'Fete de la Musique', 'months': [6]},
        {'name': 'Run In Lyon', 'months': [10]},
        {'name': 'Biennale de la Danse', 'months': [9]},
        {'name': 'Biennale d Art Contemporain', 'months': [9, 10, 11, 12]}
    ]

    months = df['period_start'].dt.month
    matched_names = []
    for m in months:
        names = [e['name'] for e in known_events if m in e['months']]
        matched_names.append(", ".join(names))

    df['known_event'] = matched_names
    df['event_match'] = df['known_event'] != ''

    return df


def compute_seasonality(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df_work = add_datetime_column(df)
    df_work['cluster'] = labels
    df_work = df_work.dropna(subset=['taken_date'])

    df_work['month'] = df_work['taken_date'].dt.month
    df_work['weekday'] = df_work['taken_date'].dt.day_name()

    monthly = (
        df_work.groupby(['cluster', 'month'])
        .agg(photo_count=('id', 'count'), user_count=('user', 'nunique'))
        .reset_index()
    )

    weekday = (
        df_work.groupby(['cluster', 'weekday'])
        .agg(photo_count=('id', 'count'), user_count=('user', 'nunique'))
        .reset_index()
    )

    return monthly, weekday


def summarize_temporal_span(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    df_work = add_datetime_column(df)
    min_date = df_work['taken_date'].min()
    max_date = df_work['taken_date'].max()
    return min_date, max_date

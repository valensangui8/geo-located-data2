import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6_371_000


def meters_to_radians(distance_m: float) -> float:
    return distance_m / EARTH_RADIUS_M


def to_radians_coords(df: pd.DataFrame) -> np.ndarray:
    coords = df[['lat', 'long']].to_numpy()
    return np.radians(coords)


def project_to_meters(df: pd.DataFrame) -> np.ndarray:
    lat = np.radians(df['lat'].to_numpy())
    lon = np.radians(df['long'].to_numpy())
    lat0 = float(np.mean(lat)) if len(lat) else 0.0
    x = lon * np.cos(lat0) * EARTH_RADIUS_M
    y = lat * EARTH_RADIUS_M
    return np.column_stack([x, y])

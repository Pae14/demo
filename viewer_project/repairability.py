"""Repairable / non-repairable classification from PFH/SHOT descriptors."""
import csv
import os
from pathlib import Path

import numpy as np

PFH_DIM = 125
SHOT_DIM = 352
GEOMETRY_NAMES = (
    'point_count', 'depth_range', 'depth_std', 'extent_x', 'extent_y',
    'footprint_area',
)

def read_ascii_pcd(path):
    with open(path, 'r', encoding='ascii') as stream:
        for line in stream:
            if line.startswith('DATA'):
                break
        return np.loadtxt(stream, dtype=np.float32, ndmin=2)


def read_descriptors(path):
    with open(path, 'r', encoding='utf-8', newline='') as stream:
        rows = list(csv.reader(stream))
    return np.asarray([[float(v) for v in r[1:]] for r in rows[1:]], dtype=np.float64)


def geometry_features(points):
    points = np.asarray(points, dtype=np.float64)
    extent = points.max(axis=0) - points.min(axis=0)
    return np.array([
        len(points), extent[2], points[:, 2].std(), extent[0], extent[1],
        extent[0] * extent[1],
    ])


def descriptor_features(descriptors):
    descriptors = np.asarray(descriptors, dtype=np.float64)
    if descriptors.shape[1] != PFH_DIM + SHOT_DIM:
        raise ValueError(f'Expected {PFH_DIM + SHOT_DIM} descriptor columns')
    descriptors = np.nan_to_num(descriptors)
    return np.concatenate((descriptors.mean(axis=0), descriptors.std(axis=0)))


def feature_vector(points, descriptors, kind='both'):
    geo = geometry_features(points)
    desc = descriptor_features(descriptors)
    return {'geometry': geo, 'descriptors': desc,
            'both': np.concatenate((geo, desc))}[kind]


def load_model(path):
    path = Path(path)
    if not path.is_file():
        return None
    import joblib
    return joblib.load(path)


def predict_repairable(bundle, points, descriptors):
    """Return (repairable: bool, probability_repairable: float)."""
    x = feature_vector(points, descriptors, bundle['kind']).reshape(1, -1)
    proba = float(bundle['model'].predict_proba(x)[0][
        list(bundle['model'].classes_).index(1)])
    return proba >= bundle.get('threshold', 0.5), proba

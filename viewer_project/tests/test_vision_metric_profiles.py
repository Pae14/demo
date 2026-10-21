import numpy as np

from utils_vision import _compute_metrics, _repairability_feature_line


def test_shape_dimensions_are_physical_and_ordered():
    mask = np.zeros((240, 320), dtype=bool)
    mask[100:104, 50:150] = True

    metrics = _compute_metrics(mask, np.zeros(mask.shape, dtype=np.uint8))

    assert metrics['major_axis_length_mm'] > metrics['minor_axis_length_mm']
    assert metrics['major_axis_length_mm'] > 5.0
    assert metrics['minor_axis_length_mm'] < 1.0


def test_defect_profile_uses_class_relevant_metrics():
    metrics = {
        'area_mm2': 4.5,
        'major_axis_length_mm': 8.2,
        'minor_axis_length_mm': 0.7,
        'depth_p95_mm': 0.12,
        'depth_max_mm': 0.2,
        'depth_volume_mm3': 1.4,
    }

    crack = _repairability_feature_line(1, metrics)
    hole = _repairability_feature_line(3, metrics)
    abrasion = _repairability_feature_line(0, metrics)
    friction = _repairability_feature_line(2, metrics)

    assert 'Length: 8.20 mm' in crack and 'Width: 0.70 mm' in crack
    assert 'Volume: 1.400 mm³' in hole and 'Max depth: 0.200 mm' in hole
    assert 'P95 depth: 0.120 mm' in abrasion
    assert 'P95 depth: 0.120 mm' in friction


def test_profile_handles_missing_depth_metrics():
    line = _repairability_feature_line(3, {'area_mm2': 2.0})

    assert 'Area: 2.0 mm²' in line
    assert 'P95 depth: —' in line


def test_repairable_3d_uses_depth_volume_and_size():
    from utils_vision import _repairable_3d

    ok = {'area_mm2': 20, 'depth_p95_mm': 0.3, 'depth_volume_mm3': 5}
    assert _repairable_3d(3, ok) is True
    assert _repairable_3d(3, {**ok, 'depth_volume_mm3': 20}) is False
    assert _repairable_3d(3, {**ok, 'depth_p95_mm': 0.5}) is False
    assert _repairable_3d(3, {**ok, 'area_mm2': 70}) is False
    assert _repairable_3d(3, {'area_mm2': 20, 'depth_p95_mm': None}) is None

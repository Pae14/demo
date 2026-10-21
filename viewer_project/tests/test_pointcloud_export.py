import csv
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

import pointcloud_export
from pointcloud_export import (
    _estimate_feature_radius,
    _write_ascii_pcd,
    SegmentationRegion,
    export_segmented_pointclouds,
    extract_segmentation_regions,
    render_depth_segmentation_overlay,
)


def test_yolo_regions_exclude_no_defect_and_resize_masks():
    prediction = SimpleNamespace(
        masks=SimpleNamespace(data=np.stack((
            np.ones((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
        ))),
        boxes=SimpleNamespace(
            cls=np.array([0, 4], dtype=np.float32),
            conf=np.array([0.9, 0.99], dtype=np.float32),
        ),
    )

    regions = extract_segmentation_regions(0, prediction, (4, 4), 0.25)

    assert len(regions) == 1
    assert regions[0].class_id == 0
    assert np.isclose(regions[0].confidence, 0.9)
    assert regions[0].mask.shape == (4, 4)
    assert regions[0].mask.all()


def test_dinov3_regions_are_connected_components():
    probabilities = np.zeros((5, 4, 4), dtype=np.float32)
    probabilities[4] = 0.9
    probabilities[2, 1:3, 1:3] = 0.95
    probabilities[4, 1:3, 1:3] = 0.05

    regions = extract_segmentation_regions(3, probabilities, (4, 4), 0.25)

    assert len(regions) == 1
    assert regions[0].class_id == 2
    assert regions[0].mask.sum() == 4


def test_rfdetr_regions_preserve_instance_confidence():
    prediction = SimpleNamespace(
        mask=np.stack((
            np.eye(4, dtype=bool),
            np.ones((4, 4), dtype=bool),
        )),
        class_id=np.array([1, 4]),
        confidence=np.array([0.82, 0.99], dtype=np.float32),
    )

    regions = extract_segmentation_regions(2, prediction, (4, 4), 0.25)

    assert len(regions) == 1
    assert regions[0].class_id == 1
    assert np.isclose(regions[0].confidence, 0.82)
    assert regions[0].mask.sum() == 4


def test_ascii_pcd_header_and_feature_radius(tmp_path):
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
    ], dtype=np.float32)
    path = tmp_path / 'region.pcd'
    _write_ascii_pcd(path, points)

    lines = path.read_text(encoding='ascii').splitlines()
    assert 'FIELDS x y z' in lines
    assert 'POINTS 3' in lines
    assert lines[-3:] == ['0 0 0', '0.100000001 0 0', '0 0.100000001 0']

    grid = np.zeros((3, 3, 3), dtype=np.float32)
    grid[:, :, 0] = np.arange(3, dtype=np.float32)[None, :] * 0.1
    grid[:, :, 1] = np.arange(3, dtype=np.float32)[:, None] * 0.1
    assert np.isclose(_estimate_feature_radius(grid), 0.3)


def test_depth_segmentation_overlay_keeps_image_shape(monkeypatch):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    cloud = np.zeros((4, 4, 3), dtype=np.float32)
    cloud[:, :, 0] = np.arange(4, dtype=np.float32)[None, :]
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    region = SimpleNamespace(mask=mask, class_id=0)
    monkeypatch.setattr(
        pointcloud_export,
        '_render_depth',
        lambda _: np.full((4, 4, 3), 90, dtype=np.uint8),
    )

    overlay = render_depth_segmentation_overlay(
        image, cloud, [region], {0: (210, 80, 70)}
    )

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    assert np.any(overlay[mask] != overlay[0, 0])


@pytest.mark.skipif(
    not os.environ.get('PCL_DESCRIPTOR_EXPORTER'),
    reason='native PCL exporter is not configured',
)
def test_export_writes_masked_pcd_and_both_descriptors(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pointcloud_export,
        '_render_depth',
        lambda _: np.full((20, 20, 3), 90, dtype=np.uint8),
    )
    y, x = np.indices((20, 20), dtype=np.float32)
    point_cloud = np.stack((
        x * 0.0000634,
        y * 0.0000634,
        (x * x + y * y) * 0.00000001,
    ), axis=-1)
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    region = SegmentationRegion(
        mask=np.ones((20, 20), dtype=bool),
        class_id=0,
        confidence=0.91,
    )
    small_mask = np.zeros((20, 20), dtype=bool)
    small_mask[:2, :2] = True
    small_region = SegmentationRegion(
        mask=small_mask,
        class_id=1,
        confidence=0.8,
    )

    capture_dir, exported_count, skipped_count = export_segmented_pointclouds(
        image,
        point_cloud,
        [region, small_region],
        tmp_path,
        'YOLO12n-seg',
        {0: (210, 80, 70)},
        {0: 'Abrasion'},
    )

    manifest = json.loads((capture_dir / 'manifest.json').read_text())
    instance = manifest['instances'][0]
    assert exported_count == 1
    assert skipped_count == 1
    assert instance['point_count'] == 400
    assert instance['descriptor_rows'] == 400
    assert manifest['skipped_instances'][0]['valid_point_count'] == 4
    assert (capture_dir / 'depth_segmentation_overlay.png').is_file()
    assert (capture_dir / instance['point_cloud']).is_file()
    with (capture_dir / instance['descriptors']).open(newline='') as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 401
    assert len(rows[0]) == 478


def test_pcl_normals_view_renders_vectors():
    import numpy as np
    from pointcloud_export import render_pcl_normals_view, estimate_grid_normals

    height, width = 60, 80
    rows, cols = np.indices((height, width), dtype=np.float32)
    z = 0.001 * np.exp(-(((rows - 30) ** 2 + (cols - 40) ** 2) / 200.0))
    cloud = np.stack((cols * 6.3e-5, rows * 6.3e-5, z), axis=-1)
    normals = estimate_grid_normals(cloud)
    assert np.isfinite(normals[30, 20]).all()
    assert (normals[1:-1, 1:-1, 2] > 0).all()
    image = render_pcl_normals_view(
        np.zeros((height, width, 3), np.uint8), cloud, [], {}
    )
    assert image.shape == (height * 2, width * 4, 3)
    assert image.max() > 100

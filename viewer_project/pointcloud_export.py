import csv
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess

import cv2
import numpy as np


@dataclass(frozen=True)
class SegmentationRegion:
    mask: np.ndarray
    class_id: int
    confidence: float


def _as_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value)


def _region(class_id, confidence, mask):
    return SegmentationRegion(
        mask=np.asarray(mask, dtype=bool),
        class_id=int(class_id),
        confidence=float(confidence),
    )


def _deduplicate_instances(class_ids, confidences, masks, image_shape):
    height, width = image_shape
    regions = []
    for class_id in np.unique(class_ids):
        class_id = int(class_id)
        if class_id == 4:
            continue
        indices = np.flatnonzero(class_ids == class_id)
        indices = indices[np.argsort(-confidences[indices])]
        accepted = np.zeros((height, width), dtype=bool)
        accepted_count = 0

        for index in indices:
            if accepted_count == 2:
                break
            mask = cv2.resize(
                masks[index].astype(np.float32),
                (width, height),
            ) > 0.5
            if not mask.any():
                continue
            if accepted.any():
                overlap = np.logical_and(accepted, mask).sum()
                union = np.logical_or(accepted, mask).sum()
                if overlap / max(union, 1) > 0.7:
                    continue
            accepted |= mask
            accepted_count += 1
            regions.append(_region(class_id, confidences[index], mask))
    return regions


def extract_segmentation_regions(model_index, prediction, image_shape, confidence):
    height, width = image_shape
    if model_index in (0, 1):
        if prediction.masks is None or prediction.boxes is None:
            return []
        masks = _as_numpy(prediction.masks.data)
        class_ids = _as_numpy(prediction.boxes.cls).astype(np.int64)
        confidences = _as_numpy(prediction.boxes.conf).astype(np.float32)
        keep = confidences >= confidence
        return _deduplicate_instances(
            class_ids[keep], confidences[keep], masks[keep], image_shape
        )

    if model_index == 2:
        if prediction is None or prediction.mask is None:
            return []
        masks = _as_numpy(prediction.mask).astype(bool)
        if masks.ndim != 3:
            raise ValueError('RF-DETR masks must have shape (instances, height, width)')
        if masks.shape[0] == 0:
            return []
        class_ids = _as_numpy(prediction.class_id).astype(np.int64)
        confidences = _as_numpy(prediction.confidence).astype(np.float32)
        if len(class_ids) != masks.shape[0] or len(confidences) != masks.shape[0]:
            raise ValueError('RF-DETR masks, classes and confidences must align')
        keep = confidences >= confidence
        return _deduplicate_instances(
            class_ids[keep], confidences[keep], masks[keep], image_shape
        )

    probabilities = np.asarray(prediction, dtype=np.float32)
    if probabilities.ndim != 3 or probabilities.shape[0] != 5:
        raise ValueError('DINOv3 must return probabilities for five viewer classes')
    probabilities = np.stack([
        cv2.resize(probabilities[class_id], (width, height), interpolation=cv2.INTER_LINEAR)
        for class_id in range(5)
    ])
    class_ids = probabilities.argmax(axis=0)
    pixel_confidence = probabilities.max(axis=0)
    class_ids[pixel_confidence < confidence] = 4

    regions = []
    for class_id in range(4):
        component_count, components = cv2.connectedComponents(
            (class_ids == class_id).astype(np.uint8),
            connectivity=8,
        )
        for component_id in range(1, component_count):
            mask = components == component_id
            regions.append(_region(
                class_id,
                pixel_confidence[mask].mean(),
                mask,
            ))
    return regions


def render_depth_segmentation_overlay(bgr, point_cloud, regions, class_colors):
    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    if point_cloud.shape != (*bgr.shape[:2], 3):
        raise ValueError('Point cloud and image dimensions do not match')

    depth_rgb = _render_depth(point_cloud)
    overlay = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    for region in regions:
        color = np.asarray(
            class_colors.get(region.class_id, (180, 180, 180)),
            dtype=np.float32,
        )
        mask = region.mask
        if mask.shape != bgr.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (bgr.shape[1], bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if not mask.any():
            continue
        fused = (
            depth_rgb[mask].astype(np.float32) * 0.7
            + color * 0.3
        ).clip(0, 255).astype(np.uint8)
        overlay[mask] = fused
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            overlay,
            contours,
            -1,
            tuple(int(channel) for channel in color),
            1,
        )
    return overlay


def estimate_grid_normals(point_cloud):
    """Surface normals of an organized (H, W, 3) cloud; z-up, NaN where invalid."""
    cloud = np.asarray(point_cloud, dtype=np.float32)
    d_col = np.zeros_like(cloud)
    d_row = np.zeros_like(cloud)
    d_col[:, 1:-1] = cloud[:, 2:] - cloud[:, :-2]
    d_row[1:-1] = cloud[2:] - cloud[:-2]
    normals = np.cross(d_col, d_row)
    normals[normals[..., 2] < 0] *= -1
    length = np.linalg.norm(normals, axis=2, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        normals = normals / length
    normals[~np.isfinite(normals).all(axis=2) | (length[..., 0] == 0)] = np.nan
    return normals


def render_pcl_normals_view(
    bgr, point_cloud, regions, class_colors, stride=6, scale=2
):
    """Two panels: top view with normal tilt vectors and an oblique 3D view."""
    cloud = np.asarray(point_cloud, dtype=np.float32)
    if cloud.shape != (*bgr.shape[:2], 3):
        raise ValueError('Point cloud and image dimensions do not match')
    height, width = cloud.shape[:2]

    valid = np.isfinite(cloud).all(axis=2)
    selected = np.zeros((height, width), dtype=bool)
    label = np.full((height, width), -1, dtype=np.int32)
    for region in regions:
        mask = region.mask
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8), (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        selected |= mask
        label[mask] = region.class_id
    if not selected.any():
        selected = valid.copy()
    selected &= valid

    z = cloud[:, :, 2]
    xy_extent = max(width, height) * float(
        np.nanmedian(np.abs(np.diff(cloud[:, :, 0], axis=1))) or 1.0
    )
    z_valid = z[valid]
    z_span = float(np.ptp(z_valid)) if z_valid.size else 0.0
    z_gain = 0.3 * xy_extent / z_span if z_span > 1e-9 else 1.0
    display_cloud = cloud.copy()
    display_cloud[:, :, 2] -= float(np.nanmin(z_valid)) if z_valid.size else 0.0
    display_cloud[:, :, 2] *= z_gain
    normals = estimate_grid_normals(display_cloud)

    panel_w, panel_h = width * scale, height * scale
    top = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    top = cv2.resize(top, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)
    top = (top * 0.45).astype(np.uint8)
    oblique = np.full((panel_h, panel_w, 3), 18, dtype=np.uint8)

    rows, cols = np.nonzero(selected)
    keep = (rows % stride == stride // 2) & (cols % stride == stride // 2)
    rows, cols = rows[keep], cols[keep]
    colors = [
        tuple(int(c) for c in class_colors.get(int(label[r, c]), (230, 230, 230)))
        for r, c in zip(rows, cols)
    ]

    tilt = np.linalg.norm(normals[rows, cols, :2], axis=1)
    max_tilt = max(float(np.nanpercentile(tilt, 98)) if tilt.size else 0.0, 1e-6)
    arrow = stride * scale * 1.6
    for r, c, color, n in zip(rows, cols, colors, normals[rows, cols]):
        if not np.isfinite(n).all():
            continue
        start = (int(c * scale + scale / 2), int(r * scale + scale / 2))
        gain = arrow / max_tilt
        end = (int(start[0] + n[0] * gain), int(start[1] + n[1] * gain))
        cv2.arrowedLine(top, start, end, color, 1, cv2.LINE_AA, tipLength=0.4)
        cv2.circle(top, start, 1, (255, 255, 255), -1)

    azimuth, elevation = np.radians(35.0), np.radians(40.0)
    ca, sa, ce, se = np.cos(azimuth), np.sin(azimuth), np.cos(elevation), np.sin(elevation)

    def project(points):
        x = points[:, 0] - float(np.nanmean(display_cloud[:, :, 0]))
        y = points[:, 1] - float(np.nanmean(display_cloud[:, :, 1]))
        zz = points[:, 2]
        xr = x * ca - y * sa
        yr = x * sa + y * ca
        screen_x = xr
        screen_y = -(zz * ce) + yr * se
        return screen_x, screen_y, yr * ce + zz * se

    pts = display_cloud[rows, cols]
    nrm = normals[rows, cols]
    ok = np.isfinite(nrm).all(axis=1)
    if ok.any():
        px, py, depth = project(pts[ok])
        ex, ey, _ = project(pts[ok] + nrm[ok] * 0.05 * xy_extent)
        span = max(
            float(np.ptp(px)), float(np.ptp(py)), 1e-9
        )
        fit = 0.78 * min(panel_w, panel_h) / span
        cx = panel_w / 2 - float(px.mean()) * fit
        cy = panel_h / 2 - float(py.mean()) * fit
        ok_colors = [c for c, flag in zip(colors, ok) if flag]
        for i in np.argsort(depth)[::-1]:
            a = (int(px[i] * fit + cx), int(py[i] * fit + cy))
            b = (int(ex[i] * fit + cx), int(ey[i] * fit + cy))
            cv2.arrowedLine(
                oblique, a, b, ok_colors[i], 1, cv2.LINE_AA, tipLength=0.3
            )
            cv2.circle(oblique, a, 1, (255, 255, 255), -1)

    try:
        radius_mm = _estimate_feature_radius(cloud) * 1000.0
        radius_text = f'PFH/SHOT radius {radius_mm:.2f} mm'
    except ValueError:
        radius_text = 'PFH/SHOT radius n/a'
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(top, 'Top view: normal tilt vectors', (8, 18), font, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(top, f'{int(selected.sum())} pts, {radius_text}',
                (8, panel_h - 8), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(oblique, f'3D view: normals (Z x{z_gain:.0f} for display)',
                (8, 18), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack((top, oblique))


def _render_depth(point_cloud):
    from utils_vision import _render_pointcloud_depth

    return _render_pointcloud_depth(point_cloud)


def _write_ascii_pcd(path, points):
    count = len(points)
    with path.open('w', encoding='ascii', newline='') as stream:
        stream.write(
            '# .PCD v0.7 - Point Cloud Data file format\n'
            'VERSION 0.7\n'
            'FIELDS x y z\n'
            'SIZE 4 4 4\n'
            'TYPE F F F\n'
            'COUNT 1 1 1\n'
            f'WIDTH {count}\n'
            'HEIGHT 1\n'
            'VIEWPOINT 0 0 0 1 0 0 0\n'
            f'POINTS {count}\n'
            'DATA ascii\n'
        )
        np.savetxt(stream, points, fmt='%.9g')


def _estimate_feature_radius(point_cloud):
    spacings = []
    for axis in (0, 1):
        neighbors = np.linalg.norm(
            np.diff(point_cloud[:, :, :2], axis=axis),
            axis=2,
        )
        finite = neighbors[np.isfinite(neighbors) & (neighbors > 0)]
        if finite.size:
            spacings.append(finite)
    if not spacings:
        raise ValueError('Cannot estimate point spacing for PCL descriptors')
    spacing = float(np.median(np.concatenate(spacings)))
    return spacing * 3.0


def _safe_name(value):
    normalized = re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-')
    return normalized or 'class'


def export_segmented_pointclouds(
    bgr,
    point_cloud,
    regions,
    output_root,
    model_name,
    class_colors,
    class_names,
    repair_model=None,
):
    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    if point_cloud.shape != (*bgr.shape[:2], 3):
        raise ValueError('Point cloud and image dimensions do not match')

    valid_regions = []
    skipped_regions = []
    for region in regions:
        if region.mask.shape != bgr.shape[:2]:
            raise ValueError('A segmentation mask does not match the image dimensions')
        mask = region.mask & np.isfinite(point_cloud).all(axis=2)
        points = np.ascontiguousarray(point_cloud[mask], dtype=np.float32)
        if len(points) >= 12:
            valid_regions.append((region, points))
        else:
            skipped_regions.append({
                'class_id': region.class_id,
                'confidence': region.confidence,
                'valid_point_count': len(points),
                'reason': 'fewer than 12 valid 3D points',
            })
    if not valid_regions:
        raise ValueError('No segmented region contains at least 12 valid 3D points')

    exporter = Path(os.environ.get(
        'PCL_DESCRIPTOR_EXPORTER',
        '/usr/local/bin/pcl_descriptor_export',
    ))
    if not exporter.is_file() or not os.access(exporter, os.X_OK):
        raise FileNotFoundError(
            f'PCL descriptor executable not found or not executable: {exporter}'
        )

    radius = _estimate_feature_radius(point_cloud)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    capture_dir = Path(output_root) / f'{timestamp}_{_safe_name(model_name)}'
    capture_dir.mkdir(parents=True, exist_ok=False)
    overlay_path = capture_dir / 'depth_segmentation_overlay.png'
    if not cv2.imwrite(
        str(overlay_path),
        cv2.cvtColor(
            render_depth_segmentation_overlay(
                bgr, point_cloud, regions, class_colors
            ),
            cv2.COLOR_RGB2BGR,
        ),
    ):
        raise OSError(f'Could not write depth/segmentation overlay: {overlay_path}')
    manifest = {
        'capture': timestamp,
        'model': model_name,
        'image_shape': list(bgr.shape[:2]),
        'descriptor_radius_m': radius,
        'descriptor_columns': {
            'PFH': 125,
            'SHOT': 352,
        },
        'instances': [],
        'skipped_instances': skipped_regions,
    }

    for instance_index, (region, points) in enumerate(valid_regions, start=1):
        class_name = class_names.get(region.class_id, str(region.class_id))
        stem = f'instance_{instance_index:03d}_{_safe_name(class_name)}'
        cloud_path = capture_dir / f'{stem}.pcd'
        descriptors_path = capture_dir / f'{stem}_pfh_shot.csv'
        _write_ascii_pcd(cloud_path, points)
        try:
            subprocess.run(
                [str(exporter), str(cloud_path), str(descriptors_path), f'{radius:.12g}'],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise RuntimeError(
                f'PCL feature extraction failed for {cloud_path.name}: {detail}'
            ) from exc

        with descriptors_path.open('r', encoding='utf-8', newline='') as stream:
            descriptor_rows = max(sum(1 for _ in csv.reader(stream)) - 1, 0)
        if descriptor_rows != len(points):
            raise RuntimeError(
                f'PCL returned {descriptor_rows} descriptor rows for {len(points)} points'
            )
        repair = {}
        if repair_model is not None:
            from repairability import predict_repairable, read_descriptors
            repairable, probability = predict_repairable(
                repair_model, points, read_descriptors(descriptors_path)
            )
            repair = {
                'repairable': repairable,
                'repairable_probability': probability,
            }
        manifest['instances'].append({
            **repair,
            'file_stem': stem,
            'class_id': region.class_id,
            'class_name': class_name,
            'confidence': region.confidence,
            'point_count': len(points),
            'descriptor_rows': descriptor_rows,
            'point_cloud': cloud_path.name,
            'descriptors': descriptors_path.name,
        })

    manifest_path = capture_dir / 'manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    return capture_dir, len(valid_regions), len(skipped_regions)

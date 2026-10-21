"""Convert saved GelSight RGB images into depth maps, point clouds and 3D metrics.

Usage (inside the viewer container):
    python3 rgb_to_depth.py --images data/test2/images --labels data/test2/labels \
        --out exports/depth_test2

The reconstruction is the same network used by the live node (gs3drecon.py).
The per-pixel zero reference is the median depth of the "No Defect" images
(class 4) when labels are available, or of the whole set otherwise.
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
for candidate in (
    os.path.join(ROOT, '..', 'vb_tactile_sensors', 'vb_tactile_sensors'),
    '/ros2_ws/src/vb_tactile_sensors/vb_tactile_sensors',
):
    if os.path.isfile(os.path.join(candidate, 'gs3drecon.py')):
        sys.path.insert(0, candidate)
        break

MPP_MM = 0.0634  # mm per pixel, same value as gs_pub_img_depth.py
NO_DEFECT = 4


class _Dev:
    imgw, imgh = 240, 320


def _load_reconstructor(weights, device):
    import gs3drecon

    recon = gs3drecon.Reconstruction3D(_Dev())
    if recon.load_nn(weights, device) is None:
        raise SystemExit(f'Could not load weights: {weights}')
    recon.dm_zero_counter = 50  # skip the live "zeroing" phase
    recon.dm_zero = 0.0
    return recon


def _label(labels_dir, stem):
    path = os.path.join(labels_dir, stem + '.txt') if labels_dir else None
    if path and os.path.isfile(path):
        text = open(path).read().split()
        if text:
            return int(text[0])
    return -1


def _entropy(values, bins=64):
    hist, _ = np.histogram(values, bins=bins)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def _metrics(depth_mm):
    # Plane fitted on the border removes residual tilt of the gel
    h, w = depth_mm.shape
    ys, xs = np.mgrid[:h, :w]
    border = np.zeros((h, w), bool)
    border[:12, :] = border[-12:, :] = border[:, :12] = border[:, -12:] = True
    A = np.column_stack((xs[border], ys[border], np.ones(border.sum())))
    coef = np.linalg.lstsq(A, depth_mm[border], rcond=None)[0]
    plane = coef[0] * xs + coef[1] * ys + coef[2]
    rel = depth_mm - plane
    pix_area = MPP_MM ** 2
    return {
        'depth_mean_mm': float(rel.mean()),
        'depth_std_mm': float(rel.std()),
        'depth_p95_mm': float(np.percentile(rel, 95)),
        'depth_max_mm': float(rel.max()),
        'depth_range_mm': float(rel.max() - rel.min()),
        'volume_mm3': float(np.clip(rel, 0, None).sum() * pix_area),
        'density_pts_mm2': float(rel.size / (rel.size * pix_area)),
        'entropy_depth': _entropy(rel),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--labels', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--weights', default=os.path.join(
        ROOT, '..', 'vb_tactile_sensors', 'vb_tactile_sensors', 'nnmini.pt'))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--rgb', action='store_true',
                    help='Convert to RGB first (the live node feeds BGR as captured)')
    args = ap.parse_args()

    names = sorted(
        f for f in os.listdir(args.images)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')))
    if args.limit:
        names = names[:args.limit]
    for sub in ('depth', 'cloud', 'depth_png'):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    recon = _load_reconstructor(args.weights, args.device)
    depths, labels = {}, {}
    for i, name in enumerate(names, 1):
        stem = os.path.splitext(name)[0]
        img = cv2.imread(os.path.join(args.images, name))
        if img.shape[:2] != (240, 320):
            img = cv2.resize(img, (320, 240))
        if args.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depths[stem] = recon.get_depthmap(img, False).astype(np.float32)
        labels[stem] = _label(args.labels, stem)
        if i % 50 == 0 or i == len(names):
            print(f'{i}/{len(names)} reconstructed', flush=True)

    ref_stems = [s for s in depths if labels[s] == NO_DEFECT] or list(depths)
    reference = np.median(np.stack([depths[s] for s in ref_stems]), axis=0)
    print(f'Zero reference from {len(ref_stems)} images')

    rows = []
    for stem, depth in depths.items():
        depth_mm = (depth - reference) * MPP_MM
        h, w = depth_mm.shape
        ys, xs = np.indices((h, w), dtype=np.float32)
        cloud = np.stack((xs * MPP_MM, ys * MPP_MM, depth_mm), -1) / 1000.0
        np.save(os.path.join(args.out, 'depth', stem + '.npy'), depth_mm)
        np.save(os.path.join(args.out, 'cloud', stem + '.npy'), cloud.astype(np.float32))
        lo, hi = np.percentile(depth_mm, [1, 99])
        norm = np.clip((depth_mm - lo) / max(hi - lo, 1e-6), 0, 1)
        cv2.imwrite(os.path.join(args.out, 'depth_png', stem + '.png'),
                    cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET))
        rows.append({'image': stem, 'class_id': labels[stem], **_metrics(depth_mm)})

    with open(os.path.join(args.out, 'metrics_3d.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Done: {len(rows)} images -> {args.out}')


if __name__ == '__main__':
    main()

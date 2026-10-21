"""Segment defects with YOLO and compute 2D + 3D metrics per defect region.

Run after rgb_to_depth.py (it needs <depth-dir>/cloud/<image>.npy):
    python3 segment_metrics.py --images data/test2/images --labels data/test2/labels \
        --depth-dir exports/depth_test2 --out exports/depth_test2/defect_metrics.csv

Output: one row per detected defect region with the depth metrics computed
inside the mask (plane fitted on the ring around it), plus a per-class summary.
"""
import argparse
import csv
import os

import cv2
import numpy as np

from config import CLASS_NAMES_SHORT, YOLO12_PATH, YOLO26_PATH
from utils_vision import _compute_metrics

NO_DEFECT = 4
MODELS = {'yolo12': YOLO12_PATH, 'yolo26': YOLO26_PATH}


def _gt(labels_dir, stem):
    path = os.path.join(labels_dir, stem + '.txt') if labels_dir else ''
    if os.path.isfile(path):
        text = open(path).read().split()
        if text:
            return int(text[0])
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--labels', default=None)
    ap.add_argument('--depth-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--overlay-dir', default=None)
    ap.add_argument('--model', choices=list(MODELS), default='yolo12')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(MODELS[args.model])
    overlay_dir = args.overlay_dir or os.path.splitext(args.out)[0] + '_overlays'
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    names = sorted(f for f in os.listdir(args.images)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')))
    if args.limit:
        names = names[:args.limit]

    rows, missed = [], 0
    for i, name in enumerate(names, 1):
        stem = os.path.splitext(name)[0]
        cloud_path = os.path.join(args.depth_dir, 'cloud', stem + '.npy')
        if not os.path.isfile(cloud_path):
            continue
        bgr = cv2.imread(os.path.join(args.images, name))
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cloud = np.load(cloud_path)
        result = model.predict(bgr, conf=args.conf, imgsz=640, verbose=False)[0]
        gt = _gt(args.labels, stem)
        z_mm = cloud[:, :, 2] * 1000.0
        lo, hi = np.percentile(z_mm[np.isfinite(z_mm)], [1, 99])
        depth_gray = np.clip((z_mm - lo) / max(hi - lo, 1e-6), 0, 1)
        overlay = cv2.applyColorMap((depth_gray * 255).astype(np.uint8), cv2.COLORMAP_JET)

        found = False
        if result.masks is not None:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for j in np.argsort(-confs):
                if classes[j] == NO_DEFECT:
                    continue
                mask = cv2.resize(result.masks.data[j].cpu().numpy(), (w, h)) > 0.5
                if mask.sum() < 20:
                    continue
                found = True
                color = np.asarray([
                    (0, 140, 255), (255, 120, 0),
                    (0, 220, 80), (230, 0, 220),
                ][int(classes[j]) % 4], dtype=np.float32)
                overlay[mask] = (
                    overlay[mask].astype(np.float32) * 0.55 + color * 0.45
                ).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, color.astype(np.uint8).tolist(), 1)
                metrics = _compute_metrics(mask, gray, cloud)
                rows.append({'image': stem, 'gt_class': gt, 'pred_class': int(classes[j]),
                             'conf': float(confs[j]), **metrics})
        missed += not found
        cv2.imwrite(os.path.join(overlay_dir, stem + '.png'), overlay)
        if i % 100 == 0 or i == len(names):
            print(f'{i}/{len(names)} images, {len(rows)} regions', flush=True)

    if not rows:
        raise SystemExit('No defect regions detected')
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nImages without any defect region: {missed}')
    print('Per predicted class (mean of depth metrics in the mask):')
    for cid in sorted({r['pred_class'] for r in rows}):
        sel = [r for r in rows if r['pred_class'] == cid and r['depth_mean_mm'] is not None]
        if sel:
            print(f"  {CLASS_NAMES_SHORT.get(cid, cid) if isinstance(CLASS_NAMES_SHORT, dict) else CLASS_NAMES_SHORT[cid]:>10} "
                  f"n={len(sel):4d} "
                  f"mean={np.mean([r['depth_mean_mm'] for r in sel]):.3f} "
                  f"p95={np.mean([r['depth_p95_mm'] for r in sel]):.3f} "
                  f"max={np.mean([r['depth_max_mm'] for r in sel]):.3f} "
                  f"vol={np.mean([r['depth_volume_mm3'] for r in sel]):.3f}")


if __name__ == '__main__':
    main()

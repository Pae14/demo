"""3D + 2D region metrics for the dataset_defectos folder tree.

Per fabric, the zero reference is the median depth of its `without_defect`
images. Each defect image is segmented with YOLO and the top-confidence
non-"No Defect" region is measured. Appends to a CSV so it can be resumed.
"""
import argparse
import csv
import os
import re

import cv2
import numpy as np

import rgb_to_depth as r
from config import YOLO12_PATH
from utils_vision import _compute_metrics

TYPES = {'abrasion': 'Abrasion', 'friction': 'Friction', 'cut': 'Crack', 'hole': 'Holes'}


def depth_of(recon, path):
    img = cv2.imread(path)
    if img.shape[:2] != (240, 320):
        img = cv2.resize(img, (320, 240))
    return img, recon.get_depthmap(img, False).astype(np.float32)


def list_pngs(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.png'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--ref-n', type=int, default=60)
    ap.add_argument('--only', default=None, help='text file with relative paths to process')
    ap.add_argument('--conf', type=float, default=0.25)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(YOLO12_PATH)
    recon = r._load_reconstructor(os.path.join(
        r.ROOT, '..', 'vb_tactile_sensors', 'vb_tactile_sensors', 'nnmini.pt'), 'cpu')

    only = set(open(args.only).read().split()) if args.only else None
    done = set()
    if os.path.isfile(args.out):
        with open(args.out) as f:
            done = {row['relative_path'] for row in csv.DictReader(f)}

    refs = {}
    wd = os.path.join(args.root, 'without_defect')
    for sub in sorted(os.listdir(wd)):
        fab = re.search(r'fabric\d+', sub).group()
        files = list_pngs(os.path.join(wd, sub))
        files = files[::max(1, len(files) // args.ref_n)][:args.ref_n]
        refs[fab] = np.median(np.stack([depth_of(recon, p)[1] for p in files]), axis=0)
        print('reference', fab, len(files), flush=True)

    for folder in sorted(os.listdir(args.root)):
        if folder == 'without_defect':
            continue
        m = re.match(r'(?:(thin|thick)_)?(\w+)$', folder)
        width, kind = m.group(1) or '', TYPES[m.group(2)]
        for sub in sorted(os.listdir(os.path.join(args.root, folder))):
            fab = re.search(r'fabric\d+', sub).group()
            variant = width or ('simple' if fab == 'fabric01' else 'multiple')
            files = list_pngs(os.path.join(args.root, folder, sub))
            files = [p for p in files if os.path.relpath(p, args.root) in only] if only else files[::args.stride]
            for k, path in enumerate(files):
                rel = os.path.relpath(path, args.root)
                if rel in done:
                    continue
                bgr, depth = depth_of(recon, path)
                depth_mm = (depth - refs[fab]) * r.MPP_MM
                h, w = depth_mm.shape
                ys, xs = np.indices((h, w), dtype=np.float32)
                cloud = (np.stack((xs * r.MPP_MM, ys * r.MPP_MM, depth_mm), -1) / 1000.0
                         ).astype(np.float32)
                res = model.predict(bgr, conf=args.conf, imgsz=640, verbose=False)[0]
                row = {'relative_path': rel, 'defect_type': kind, 'variant': variant,
                       'fabric': fab, 'detected': 0, 'pred_class': -1, 'conf': ''}
                if res.masks is not None:
                    cls = res.boxes.cls.cpu().numpy().astype(int)
                    conf = res.boxes.conf.cpu().numpy()
                    for j in np.argsort(-conf):
                        if cls[j] == r.NO_DEFECT:
                            continue
                        mask = cv2.resize(res.masks.data[j].cpu().numpy(), (w, h)) > 0.5
                        if mask.sum() < 20:
                            continue
                        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                        row.update(detected=1, pred_class=int(cls[j]), conf=float(conf[j]),
                                   **_compute_metrics(mask, gray, cloud))
                        break
                rows_write(row, args.out)
            print(folder, sub, len(files), flush=True)


_writer = {}


def rows_write(row, out):
    exists = os.path.isfile(out) and os.path.getsize(out) > 0
    # Fixed header: first detected row defines the columns
    cols = _writer.get('cols')
    if cols is None:
        if exists:
            cols = next(csv.reader(open(out)))
        elif row['detected']:
            cols = list(row)
        else:
            _writer.setdefault('pending', []).append(row)
            return
        _writer['cols'] = cols
    with open(out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        if not exists:
            w.writeheader()
        for p in _writer.pop('pending', []):
            w.writerow(p)
        w.writerow(row)


if __name__ == '__main__':
    main()

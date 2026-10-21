"""Label and train the repairable / non-repairable classifier.

  python train_repairability.py --init-labels   # creates/extends labels.csv
  (fill the 'repairable' column with 1 or 0)
  python train_repairability.py                 # compares + saves the model
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

import config
from repairability import feature_vector, read_ascii_pcd, read_descriptors

ROOT = Path(config.SAVE_DIR) / 'pointclouds'
LABELS = Path(config.SAVE_DIR) / 'labels.csv'
MODEL = Path(getattr(config, 'REPAIR_MODEL_PATH', Path(config.SAVE_DIR) / 'repairability.joblib'))


def instances():
    for manifest in sorted(ROOT.glob('*/manifest.json')):
        data = json.loads(manifest.read_text(encoding='utf-8'))
        for inst in data['instances']:
            yield manifest.parent, inst


def init_labels():
    existing = {}
    if LABELS.is_file():
        with LABELS.open(newline='', encoding='utf-8') as f:
            existing = {(r['capture'], r['file_stem']): r for r in csv.DictReader(f)}
    with LABELS.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, ['capture', 'file_stem', 'class_name', 'piece', 'repairable'])
        w.writeheader()
        for d, inst in instances():
            key = (d.name, inst['file_stem'])
            old = existing.get(key, {})
            w.writerow({'capture': d.name, 'file_stem': inst['file_stem'],
                        'class_name': inst['class_name'],
                        'piece': old.get('piece') or d.name,
                        'repairable': old.get('repairable', '')})
    print(f'Wrote {LABELS}. Fill "repairable" (1/0); use "piece" to group captures of the same piece.')


def load_dataset():
    with LABELS.open(newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['repairable'].strip() in ('0', '1')]
    feats = {'geometry': [], 'descriptors': [], 'both': []}
    y, groups = [], []
    for r in rows:
        d = ROOT / r['capture']
        points = read_ascii_pcd(d / f"{r['file_stem']}.pcd")
        desc = read_descriptors(d / f"{r['file_stem']}_pfh_shot.csv")
        for k in feats:
            feats[k].append(feature_vector(points, desc, k))
        y.append(int(r['repairable'])); groups.append(r['piece'])
    return {k: np.array(v) for k, v in feats.items()}, np.array(y), np.array(groups)


def make_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=300, class_weight='balanced', random_state=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init-labels', action='store_true')
    ap.add_argument('--use', choices=('geometry', 'descriptors', 'both'), default=None,
                    help='feature set to save (default: best by CV F1)')
    args = ap.parse_args()
    if args.init_labels:
        return init_labels()

    import joblib
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    X, y, groups = load_dataset()
    if len(y) < 10 or len(set(y)) < 2:
        raise SystemExit('Need >=10 labelled instances with both classes (0 and 1).')
    n_splits = min(5, len(set(groups)))
    if n_splits < 2:
        raise SystemExit('Need at least 2 different "piece" groups.')
    scores = {}
    for kind in ('geometry', 'descriptors', 'both'):
        pred = cross_val_predict(make_model(), X[kind], y,
                                 groups=groups, cv=GroupKFold(n_splits))
        scores[kind] = f1_score(y, pred, pos_label=0)
        print(f'\n=== {kind} (F1 on non-repairable = {scores[kind]:.3f}) ===')
        print(confusion_matrix(y, pred, labels=[0, 1]), '(rows=true [0,1], cols=pred)')
        print(classification_report(y, pred, target_names=['non-repairable', 'repairable'], zero_division=0))
    kind = args.use or max(scores, key=scores.get)
    model = make_model().fit(X[kind], y)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'kind': kind, 'threshold': 0.5}, MODEL)
    print(f'Saved "{kind}" model to {MODEL}')


if __name__ == '__main__':
    main()

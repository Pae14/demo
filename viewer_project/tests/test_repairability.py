import numpy as np

from repairability import PFH_DIM, SHOT_DIM, predict_repairable, feature_vector


def test_feature_vector_and_prediction():
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(0)

    def sample(depth):
        pts = rng.random((30, 3)).astype(np.float32)
        pts[:, 2] *= depth
        return pts, rng.random((30, PFH_DIM + SHOT_DIM))

    xs, ys = [], []
    for i in range(20):
        label = i % 2
        pts, desc = sample(0.001 if label else 0.1)
        xs.append(feature_vector(pts, desc)); ys.append(label)
    bundle = {'model': RandomForestClassifier(random_state=0).fit(xs, ys),
              'kind': 'both'}
    pts, desc = sample(0.001)
    ok, p = predict_repairable(bundle, pts, desc)
    assert ok and p > 0.5

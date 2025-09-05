import numpy as np
from era.csi import combine_csi, CSIWeights
from era.metrics import symmetric_kl, w1, cosine_distance

def test_csi_monotonicity():
    w = CSIWeights(embedding=0.5, prob=0.5)
    assert combine_csi(0.1, 0.1, w) < combine_csi(0.2, 0.1, w)
    assert combine_csi(0.1, 0.1, w) < combine_csi(0.1, 0.2, w)

def test_cosine_distance_bounds():
    a = np.array([[1.0, 0.0], [1.0, 0.0]])
    b = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert abs(cosine_distance(a,b)) < 1e-6

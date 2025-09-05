import numpy as np
from numpy.linalg import norm
from scipy.stats import entropy
from scipy.stats import wasserstein_distance

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 2: a = a.mean(axis=0)
    if b.ndim == 2: b = b.mean(axis=0)
    if norm(a) == 0 or norm(b) == 0:
        return 0.0
    a = a / (norm(a) + 1e-8)
    b = b / (norm(b) + 1e-8)
    return float(1.0 - np.dot(a, b))

def symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = p + eps; q = q + eps
    p = p / p.sum(); q = q / q.sum()
    return float(entropy(p, q) + entropy(q, p)) * 0.5

def w1(p: np.ndarray, q: np.ndarray) -> float:
    xs = np.arange(len(p))
    p = p / p.sum(); q = q / q.sum()
    return float(wasserstein_distance(xs, xs, p, q))

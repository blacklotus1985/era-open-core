from dataclasses import dataclass

@dataclass
class CSIWeights:
    embedding: float = 0.5
    prob: float = 0.5

def squash(x: float) -> float:
    return x / (1.0 + x) if x >= 0 else 0.0

def combine_csi(d_emb: float, d_prob: float, w: CSIWeights) -> float:
    e = squash(d_emb)
    p = squash(d_prob)
    return w.embedding * e + w.prob * p

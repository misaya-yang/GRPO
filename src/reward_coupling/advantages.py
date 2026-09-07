"""On-policy advantages; last axis is a complete, retained actor group."""

import numpy as np


def advantages(rewards, kind="standardized", epsilon=1e-6):
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim < 1 or r.shape[-1] < 2 or not np.isfinite(r).all():
        raise ValueError("Need finite rewards, K >= 2")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    centered = r - r.mean(axis=-1, keepdims=True)
    if kind == "rloo":
        return centered * r.shape[-1] / (r.shape[-1] - 1)
    if kind == "mean_only":
        return centered
    if kind != "standardized":
        raise ValueError("Unknown advantage")
    sd = np.sqrt(np.mean(centered**2, axis=-1, keepdims=True))
    nonconstant = np.ptp(r, axis=-1, keepdims=True) > 0
    return np.divide(centered, sd + epsilon, out=np.zeros_like(r), where=(sd > 0) & nonconstant)

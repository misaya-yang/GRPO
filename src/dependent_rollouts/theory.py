"""Finite-policy operators with explicit marginal and score checks."""

import numpy as np


def pair_operator(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.ndim != 1 or np.any(p <= 0) or not np.isclose(p.sum(), 1):
        raise ValueError("Expected positive probability vector")
    if q.shape != (len(p), len(p)) or np.any(q < 0):
        raise ValueError("Invalid pair probability matrix")
    if not (np.allclose(q, q.T) and np.allclose(q.sum(1), p)):
        raise ValueError("Pair law must be exchangeable with the declared marginal")
    return q / p[:, None]


def geometry(p, scores, reward, q):
    p, s, r = np.asarray(p, float), np.asarray(scores, float), np.asarray(reward, float)
    if s.ndim == 1:
        s = s[:, None]
    if not np.allclose(p @ s, 0, atol=1e-10):
        raise ValueError("Marginal scores must be centered")
    t = pair_operator(p, q)
    fisher = s.T @ (p[:, None] * s)
    inverse = np.linalg.pinv(fisher, hermitian=True)
    projection = s @ inverse @ (s.T * p)
    r0 = r - p @ r
    v = projection @ r0
    residual = r0 - v
    g = s.T @ (p * r)
    gq = s.T @ (p * (r - t @ r))
    positive = v @ (p * (v - t @ v))
    interference = v @ (p * (t @ residual))
    return {
        "g": g,
        "gq": gq,
        "fisher": fisher,
        "projection": projection,
        "slope": float(g @ inverse @ gq),
        "positive": float(positive),
        "interference": float(interference),
        "eta": float((v @ (p * v)) / (r0 @ (p * r0))) if np.any(r0) else None,
    }


def count_counterexample():
    p = np.full(4, 0.25)
    qa = np.zeros((4, 4))
    qb = np.zeros((4, 4))
    qa[0, 2] = qa[2, 0] = qb[1, 2] = qb[2, 1] = 0.25
    qa[1, 3] = qa[3, 1] = qa[1, 1] = qa[3, 3] = 0.125
    qb[0, 3] = qb[3, 0] = qb[0, 0] = qb[3, 3] = 0.125
    return p, np.array([5.0, -7.0, 0.0, 2.0]), np.array([1.0, 1.0, 0.0, 0.0]), qa, qb


def stratified_pair(p, k):
    """Interval overlaps for categorical inverse CDF, random member labels."""
    p = np.asarray(p, float)
    if k < 2 or np.any(p <= 0) or not np.isclose(p.sum(), 1):
        raise ValueError("Invalid marginal or K")
    edges = np.r_[0.0, np.cumsum(p)]
    bins = np.arange(k + 1) / k
    mass = (
        np.maximum(
            0,
            np.minimum(edges[1:, None], bins[None, 1:])
            - np.maximum(edges[:-1, None], bins[None, :-1]),
        )
        * k
    )
    q = (np.outer(mass.sum(1), mass.sum(1)) - mass @ mass.T) / (k * (k - 1))
    return q, mass.T


def safety_bound(k, eta):
    if k < 2 or not 0 < eta <= 1:
        raise ValueError("Need K>=2 and 0<eta<=1")
    return (2 * k - 1 - eta**-0.5) / (2 * (k - 1))

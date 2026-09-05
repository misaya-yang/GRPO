"""Dossier (1), (18), (20), (21), (22) and the dose mixture of Section 8.5;
one prompt per array."""

import numpy as np


def binary_rewards(rewards):
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] < 1 or r.shape[1] < 2:
        raise ValueError("Expected [independent groups, K >= 2] for one prompt")
    if not np.isin(r, [0, 1]).all():
        raise ValueError("The count/identity experiment requires binary rewards")
    return r


def coefficient(n, k, kind="rloo", epsilon=1e-6):
    n = np.asarray(n, dtype=np.float64)
    if k < 2 or np.any((n < 0) | (n > k)):
        raise ValueError("Invalid group size or success count")
    if kind == "rloo":
        return n * (k - n) / (k * (k - 1))
    if kind != "standardized" or epsilon <= 0:
        raise ValueError("Use rloo or standardized with epsilon > 0")
    return n * (k - n) / (k * k * (np.sqrt(n / k * (1 - n / k)) + epsilon))


def advantages(rewards, arm="within", kind="rloo", epsilon=1e-6, lam=None):
    r = binary_rewards(rewards)
    b, k = r.shape
    coefficient(r.sum(axis=1), k, kind, epsilon)
    if arm == "dose":
        # A^(lam) = (1-lam) A^C + lam A^W, preregistered lam in {0, 1/2, 1}.
        # Linearity of the stored-data gradient in lam is algebra, not a result.
        if lam is None or not 0 <= lam <= 1:
            raise ValueError("Dose arm needs 0 <= lam <= 1")
        count = advantages(r, "count", kind, epsilon)
        within = advantages(r, "within", kind, epsilon)
        return (1 - lam) * count + lam * within
    if arm == "within":
        centered = r - r.mean(axis=1, keepdims=True)
        if kind == "rloo":
            return centered * k / (k - 1)
        return centered / (r.std(axis=1, keepdims=True, ddof=0) + epsilon)
    if arm == "count":
        p = r.mean()
        if p == 0 or p == 1:
            return np.zeros_like(r)
        a = coefficient(r.sum(axis=1), k, kind, epsilon).mean()
        return np.where(r == 1, a / p, -a / (1 - p))
    if arm == "cross":
        if kind != "rloo" or b < 2:
            raise ValueError("Cross baseline needs >=2 independent groups and unnormalized RLOO")
        other_mean = (r.sum() - r.sum(axis=1, keepdims=True)) / ((b - 1) * k)
        return r - other_mean
    raise ValueError(f"Unknown arm: {arm}")


def gradient(rewards, scores, **kwargs):
    a = advantages(rewards, **kwargs)
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 3 or s.shape[:2] != a.shape or not np.isfinite(s).all():
        raise ValueError("Scores must be finite [groups, K, parameters]")
    return np.einsum("bk,bkd->d", a, s) / a.size


def rewire_indices(rewards, rng):
    """Permute identities over fixed binary slots; return flat pool indices."""
    r = binary_rewards(rewards).reshape(-1)
    indices = np.arange(r.size)
    for label in (0, 1):
        slots = np.flatnonzero(r == label)
        indices[slots] = rng.permutation(slots)
    assert np.array_equal(r[indices], r)
    assert np.array_equal(np.sort(indices), np.arange(r.size))
    return indices

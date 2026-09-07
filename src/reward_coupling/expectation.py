"""Exact finite-bank expectations, without replacing E[A(R)] by A(E[R])."""

import itertools
import math

import numpy as np

from dependent_rollouts.prediction import poisson_binomial

from .advantages import advantages


def probability_vector(p, size=None):
    p = np.asarray(p, float)
    if (
        p.ndim != 1
        or not len(p)
        or (size is not None and len(p) != size)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.isclose(p.sum(), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Invalid probability vector")
    return p


def coefficients(k, epsilon=1e-6):
    if not isinstance(k, int) or k < 2 or not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("Invalid K or epsilon")
    m = np.arange(k, dtype=float)
    dp = np.sqrt((m + 1) * (k - 1 - m)) + k * epsilon
    dm = np.sqrt(m * (k - m)) + k * epsilon
    return (
        np.divide(k - 1 - m, dp, out=np.zeros(k), where=dp > 0),
        np.divide(m, dm, out=np.zeros(k), where=dm > 0),
    )


def binary_weight(p, k, epsilon=1e-6):
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Invalid pass probability")
    plus, minus = coefficients(k, epsilon)
    return sum(
        math.comb(k - 1, m) * p**m * (1 - p) ** (k - 1 - m) * (plus[m] + minus[m]) for m in range(k)
    )


def independent_binary(q, epsilon=1e-6):
    q = np.asarray(q, float)
    if q.ndim != 1 or len(q) < 2 or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("Invalid candidate pass probabilities")
    plus, minus = coefficients(len(q), epsilon)
    result = np.zeros(len(q))
    for i in range(len(q)):
        pmf = poisson_binomial(np.delete(q, i))
        result[i] = q[i] * (pmf @ plus) - (1 - q[i]) * (pmf @ minus)
    return result


def independent_grid(pmfs, scale, epsilon=1e-6, max_states=200000):
    """Exact prescribed v/scale law. Never rounds a continuous law to a grid."""
    p = np.asarray(pmfs, float)
    if p.ndim != 2 or p.shape[0] < 2 or not isinstance(scale, int) or scale < 1:
        raise ValueError("Invalid grid")
    for row in p:
        probability_vector(row)
    k = len(p)
    coefficients(k, epsilon)
    out = np.zeros(k)
    for i in range(k):
        state = {(0, 0): 1.0}
        for j in range(k):
            if i == j:
                continue
            nxt = {}
            for (total, square), mass in state.items():
                for v in np.flatnonzero(p[j]):
                    v = int(v)
                    key = (total + v, square + v * v)
                    nxt[key] = nxt.get(key, 0.0) + mass * p[j, v]
            if len(nxt) > max_states:
                raise ValueError("Exact DP state budget exceeded; no silent approximation")
            state = nxt
        for v in np.flatnonzero(p[i]):
            v = int(v)
            for (total, square), mass in state.items():
                total, square = total + v, square + v * v
                variance = k * square - total * total
                if variance > 0:
                    out[i] += (
                        p[i, v]
                        * mass
                        * (k * v - total)
                        / (math.sqrt(variance) + k * scale * epsilon)
                    )
    return out


def independent_finite(matrix, mu, kind="standardized", epsilon=1e-6, max_combinations=200000):
    """Bounded exact fallback for non-grid finite real reward supports."""
    matrix = np.asarray(matrix, float)
    k, c = matrix.shape
    mu = probability_vector(mu, c)
    if c**k > max_combinations:
        raise ValueError("Finite support enumeration budget exceeded")
    out = np.zeros(k)
    for context in itertools.product(range(c), repeat=k):
        out += np.prod(mu[list(context)]) * advantages(matrix[np.arange(k), context], kind, epsilon)
    return out


def expected_advantages(matrix, mu=None, kind="standardized", epsilon=1e-6, grid_scale=None):
    b = np.asarray(matrix, float)
    if b.ndim != 2 or b.shape[0] < 2 or b.shape[1] < 1 or not np.isfinite(b).all():
        raise ValueError("Need finite [K, contexts] matrix")
    mu = probability_vector(np.ones(b.shape[1]) / b.shape[1] if mu is None else mu, b.shape[1])
    shared = advantages(b.T, kind, epsilon).T @ mu
    q = b @ mu
    rloo = advantages(q, "rloo", epsilon)
    if kind in ("rloo", "mean_only"):
        independent = advantages(q, kind, epsilon)
    elif np.isin(b, [0, 1]).all():
        independent = independent_binary(q, epsilon)
    elif grid_scale is not None:
        if not isinstance(grid_scale, int) or grid_scale < 1:
            raise ValueError("Invalid grid scale")
        values = b * grid_scale
        if np.any(values < 0) or not np.allclose(values, np.rint(values), atol=1e-12, rtol=0):
            raise ValueError("Rewards do not belong to the declared exact grid")
        values = np.rint(values).astype(int)
        p = np.zeros((len(b), int(values.max()) + 1))
        for i in range(len(b)):
            np.add.at(p[i], values[i], mu)
        independent = independent_grid(p, grid_scale, epsilon)
    else:
        independent = independent_finite(b, mu, kind, epsilon)
    return {
        "shared": shared,
        "independent": independent,
        "rloo": rloo,
        "difference": shared - independent,
        "average": q,
    }


def rubric_matrix(verdicts, keep):
    """Uniform fixed-size, equal-weight masks; cache verdicts once per response."""
    b = np.asarray(verdicts, float)
    if (
        b.ndim != 2
        or not np.isin(b, [0, 1]).all()
        or not isinstance(keep, int)
        or not 1 <= keep <= b.shape[1]
    ):
        raise ValueError("Need binary criterion verdicts and valid keep count")
    if math.comb(b.shape[1], keep) > 100000:
        raise ValueError("Mask enumeration budget exceeded")
    masks = list(itertools.combinations(range(b.shape[1]), keep))
    return np.stack([b[:, mask].mean(axis=1) for mask in masks], axis=1), masks


def sample_advantages(matrix, mu, coupling, rng, kind="standardized", epsilon=1e-6):
    b = np.asarray(matrix, float)
    mu = probability_vector(mu, b.shape[1])
    if coupling not in ("shared", "independent"):
        raise ValueError("Unknown coupling")
    indices = rng.choice(b.shape[1], size=1 if coupling == "shared" else len(b), p=mu)
    rewards = b[np.arange(len(b)), indices]
    return advantages(rewards, kind, epsilon), indices

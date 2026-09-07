"""Finite-space structural checks; numerical tests are not general proofs."""

import itertools

import numpy as np

from .advantages import advantages
from .expectation import binary_weight, probability_vector


def additive_decomposition(table):
    """Uniform-product ANOVA projection for a scalar table on R^K.

    Marginal tuples need not have equal coordinate distributions. The reference
    product measure defines a decomposition, not a universal effect size.
    """
    a = np.asarray(table, float)
    if a.ndim < 2 or not np.isfinite(a).all():
        raise ValueError("Need a finite table with >=2 coordinates")
    mean = a.mean()
    marginal = np.full_like(a, mean)
    for axis in range(a.ndim):
        other = tuple(j for j in range(a.ndim) if j != axis)
        marginal += a.mean(axis=other, keepdims=True) - mean
    return marginal, a - marginal


def coupling_envelope(support, marginals, signed_scores, kind="standardized", epsilon=1e-6):
    """Scalar Frechet LP; fixed-bank envelope, not a deployable exogenous law."""
    from scipy.optimize import linprog

    support, p, z = (
        np.asarray(support, float),
        np.asarray(marginals, float),
        np.asarray(signed_scores, float),
    )
    if support.ndim != 1 or len(set(support)) != len(support) or not np.isfinite(support).all():
        raise ValueError("Invalid support")
    if (
        p.ndim != 2
        or p.shape[1] != len(support)
        or z.shape != (len(p),)
        or not np.isfinite(z).all()
    ):
        raise ValueError("Invalid marginals or score projection")
    for row in p:
        probability_vector(row, len(support))
    if len(support) ** len(p) > 100000:
        raise ValueError("LP size limit")
    states = np.array(list(itertools.product(range(len(support)), repeat=len(p))))
    cost = advantages(support[states], kind, epsilon) @ z / len(p)
    constraints = [np.ones(len(states))]
    rhs = [1.0]
    for i in range(len(p)):
        for value in range(len(support) - 1):
            constraints.append(states[:, i] == value)
            rhs.append(p[i, value])
    extrema = []
    for sign in (1, -1):
        fit = linprog(
            sign * cost, A_eq=np.array(constraints), b_eq=rhs, bounds=(0, None), method="highs"
        )
        if not fit.success:
            raise RuntimeError(f"LP failed: {fit.message}")
        extrema.append(float(cost @ fit.x))
    return {
        "lower": extrema[0],
        "upper": extrema[1],
        "width": extrema[1] - extrema[0],
        "status": "floating_point_fixed_bank_envelope",
    }


def binary_population(matrix, pi, mu, k, epsilon=1e-6):
    b = np.asarray(matrix, float)
    pi, mu = probability_vector(pi, len(b)), probability_vector(mu, b.shape[1])
    if not np.isfinite(b).all() or np.any((b < 0) | (b > 1)):
        raise ValueError("Expected conditional pass probabilities")
    jacobian = np.diag(pi) - np.outer(pi, pi)
    p, g = pi @ b, jacobian @ b
    shared = g @ (mu * np.array([binary_weight(x, k, epsilon) for x in p]))
    independent = binary_weight(float(p @ mu), k, epsilon) * (g @ mu)
    return shared, independent

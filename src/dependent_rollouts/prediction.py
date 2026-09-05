"""IID-only stratified forecasts; scores may be functional coordinates."""

import numpy as np


def poisson_binomial(probabilities):
    probabilities = np.asarray(probabilities, float)
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Invalid Bernoulli probabilities")
    pmf = np.array([1.0])
    for p in probabilities:
        pmf = np.convolve(pmf, [1 - p, p])
    return pmf


def standardized_forecast(p, positive, negative, epsilon=1e-6):
    """(23). Plug-in means are not asserted to be population-unbiased."""
    p, pos, neg = np.asarray(p, float), np.asarray(positive, float), np.asarray(negative, float)
    k = len(p)
    if k < 2 or epsilon <= 0 or pos.shape != neg.shape or pos.shape[0] != k:
        raise ValueError("Invalid stratum statistics")
    if not np.isfinite(pos).all() or not np.isfinite(neg).all():
        raise ValueError("Use zero means for zero-mass classes")
    result = np.zeros(pos.shape[1:])
    counts = np.arange(k)
    for j in range(k):
        pmf = poisson_binomial(np.delete(p, j))
        dplus = np.sqrt((1 + counts) / k * (1 - (1 + counts) / k)) + epsilon
        dminus = np.sqrt(counts / k * (1 - counts / k)) + epsilon
        uplus = pmf @ ((k - 1 - counts) / (k * dplus))
        uminus = pmf @ (counts / (k * dminus))
        result += (p[j] * uplus * pos[j] - (1 - p[j]) * uminus * neg[j]) / k
    return result


def iid_forecast_weights(rewards, bins, k, kind="rloo", epsilon=1e-6):
    """Integrated virtual groups, one IID-bank item per bin.

    Sum of weighted scores is the empirical-product forecast. Empty bins fail;
    missing classes have zero mass. Original IID groups own uncertainty.
    """
    r, bins = np.asarray(rewards, float), np.asarray(bins, int)
    if r.ndim != 1 or r.shape != bins.shape or not np.isin(r, [0, 1]).all():
        raise ValueError("Expected flat binary rewards and bin labels")
    if k < 2 or np.any((bins < 0) | (bins >= k)):
        raise ValueError("Invalid bins")
    counts = np.bincount(bins, minlength=k)
    if np.any(counts == 0):
        raise ValueError("IID calibration has empty strata; collect more data")
    p = np.bincount(bins, weights=r, minlength=k) / counts
    if kind == "rloo":
        a = r - (p.sum() - p[bins]) / (k - 1)
    elif kind == "standardized" and epsilon > 0:
        a = np.zeros_like(r)
        m = np.arange(k)
        for j in range(k):
            pmf = poisson_binomial(np.delete(p, j))
            for label in (0, 1):
                n = m + label
                d = np.sqrt(n / k * (1 - n / k)) + epsilon
                a[(bins == j) & (r == label)] = pmf @ ((label - n / k) / d)
    else:
        raise ValueError("Invalid estimator")
    return a / (k * counts[bins])

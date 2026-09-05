"""Explicit conditional diagnostics, paired group bootstrap, unclipped IS."""

import numpy as np

from .estimators import gradient


def functional_distortion(w, c, x, scalar=None, denominator_floor=1e-12):
    """Inputs are directional scores on the same independent IID eval bank.

    With scalar=None this is the descriptive best positive scalar, not a
    discovery-frozen predictive comparator and not a noise-corrected effect.
    """
    w, c, x = [np.asarray(v, float) for v in (w, c, x)]
    if w.shape != c.shape or w.shape != x.shape or w.ndim != 1:
        raise ValueError("Expected matching directional-score vectors")
    denom = float(np.mean(x * x))
    if denom <= denominator_floor:
        return {"status": "unresolved_reference", "delta": None, "scalar": None}
    cc = float(np.mean(c * c))
    a = max(0.0, float(np.mean(w * c)) / cc) if cc > denominator_floor else 0.0
    if scalar is not None:
        if scalar < 0:
            raise ValueError("Scalar must be nonnegative")
        a = scalar
    return {
        "status": "descriptive_only",
        "delta": float(np.sqrt(np.mean((w - a * c) ** 2) / denom)),
        "scalar": a,
    }


def bootstrap_contrast(rewards, scores, replicates=1000, seed=0, kind="rloo", alpha=0.05):
    """Resample original independent groups and recompute C on each pool."""
    r, s = np.asarray(rewards), np.asarray(scores)
    if len(r) < 2 or replicates < 2 or not 0 < alpha < 1:
        raise ValueError("Need independent groups, bootstrap replicates and 0<alpha<1")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(replicates):
        idx = rng.integers(len(r), size=len(r))
        estimates.append(
            gradient(r[idx], s[idx], arm="within", kind=kind)
            - gradient(r[idx], s[idx], arm="count", kind=kind)
        )
    estimates = np.asarray(estimates)
    return {
        "unit": "original_group",
        "conditional_on_prompt": True,
        "lower": np.quantile(estimates, alpha / 2, axis=0),
        "upper": np.quantile(estimates, 1 - alpha / 2, axis=0),
    }


def importance_reward(rewards, old_logp, new_logp):
    r, old, new = [np.asarray(v, float) for v in (rewards, old_logp, new_logp)]
    if r.ndim != 1 or r.size < 2 or old.shape != r.shape or new.shape != r.shape:
        raise ValueError("Expected matching IID evaluation vectors with >=2 observations")
    logw = new - old
    if not np.isfinite(logw).all() or np.max(logw) > 700:
        return {"status": "unresolved_overlap"}
    weights = np.exp(logw)
    if weights.sum() == 0:
        return {"status": "unresolved_overlap"}
    # Unnormalized likelihood-ratio estimate. No clipping or self-normalization.
    weighted = r * weights
    ess = float(weights.sum() ** 2 / (weights @ weights))
    return {
        "status": "estimate_requires_overlap_review",
        "reward": float(weighted.mean()),
        "paired_change": float((weighted - r).mean()),
        "paired_se": float((weighted - r).std(ddof=1) / np.sqrt(len(r))),
        "ess": ess,
        "ess_fraction": ess / len(r),
        "max_weight": float(weights.max()),
        "weight_quantiles": np.quantile(weights, [0.5, 0.9, 0.99, 1.0]),
        "mean_weight": float(weights.mean()),
        "sequence_kl_estimate": float(-logw.mean()),
    }

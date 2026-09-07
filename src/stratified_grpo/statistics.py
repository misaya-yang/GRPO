"""Prompt and independent macro-bank inference; no virtual-group sample count."""

import numpy as np


def centered_gram(vectors):
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Need at least two finite macro gradients")
    mean = x.mean(axis=0)
    centered = x - mean
    return centered @ centered.T, mean


def trace_variance(gram, counts=None):
    gram = np.asarray(gram, dtype=float)
    n = len(gram)
    if n < 2 or gram.shape != (n, n):
        raise ValueError("Invalid macro-bank Gram matrix")
    c = np.ones(n) if counts is None else np.asarray(counts, dtype=float)
    if c.shape != (n,) or np.any(c < 0) or not np.isclose(c.sum(), n):
        raise ValueError("Invalid resampling multiplicity")
    return max(0.0, float((c @ np.diag(gram) - c @ gram @ c / n) / (n - 1)))


def compare(records, threshold=0.85, replicates=2000, seed=20260907):
    """Hierarchical percentile intervals, exploratory at small prompt counts.

    Each record contains centered macro Gram matrices and per-macro costs for
    both independent arms. Resample prompts jointly and macro-banks independently.
    """
    if not records or not 0 < threshold < 1 or replicates < 100:
        raise ValueError("Invalid inference specification")
    rng = np.random.default_rng(seed)
    point = {
        arm: np.mean([trace_variance(r[arm + "_gram"]) for r in records])
        for arm in ("iid", "stratified")
    }
    costs = {
        arm: np.mean([np.mean(r[arm + "_costs"]) for r in records]) for arm in ("iid", "stratified")
    }
    if point["iid"] <= 0 or min(costs.values()) <= 0:
        return {
            "decision": "INCONCLUSIVE",
            "reason": "zero reference variance or invalid cost",
            "variances": point,
            "rho": None,
            "rho_time": None,
        }
    rho = point["stratified"] / point["iid"]
    q = costs["stratified"] / costs["iid"]
    draws = []
    for _ in range(replicates):
        totals = {"iid": [0.0, 0.0], "stratified": [0.0, 0.0]}
        for i in rng.integers(len(records), size=len(records)):
            r = records[int(i)]
            for arm in totals:
                n = len(r[arm + "_gram"])
                counts = rng.multinomial(n, np.full(n, 1 / n))
                totals[arm][0] += trace_variance(r[arm + "_gram"], counts)
                totals[arm][1] += float(counts @ np.asarray(r[arm + "_costs"]) / n)
        if totals["iid"][0] > 0 and totals["iid"][1] > 0:
            ratio = totals["stratified"][0] / totals["iid"][0]
            draws.append((ratio, ratio * totals["stratified"][1] / totals["iid"][1]))
    valid = len(draws)
    ci = np.quantile(draws, [0.025, 0.975], axis=0).T.tolist() if valid else [[None, None]] * 2
    decision = "INCONCLUSIVE"
    # This is a screening decision, never an online-training GO by itself.
    if len(records) >= 8 and valid >= replicates * 0.95:
        if ci[1][1] < threshold:
            decision = "SUPPORT_CONTINUATION"
        elif ci[1][0] >= threshold:
            decision = "EXCLUDES_PRACTICAL_GAIN"
    return {
        "decision": decision,
        "rho": float(rho),
        "cost_ratio_q": float(q),
        "rho_time": float(q * rho),
        "rho_interval": ci[0],
        "rho_time_interval": ci[1],
        "variances": point,
        "mean_macro_cost_seconds": costs,
        "practical_threshold": threshold,
        "prompts": len(records),
        "bootstrap_replicates": replicates,
        "valid_replicates": valid,
        "interval_method": "hierarchical_prompt_and_independent_macro_percentile_95",
        "limitations": "Small-sample exploratory interval; no automatic expansion or training GO",
        "uncertainty_units": ["prompt", "independent_macro_bank"],
    }

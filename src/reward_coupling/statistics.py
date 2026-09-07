"""Prompt units, independent C/D banks, and signed preregistered endpoints."""

import numpy as np


def signed_prompt_interval(values, seed=0, replicates=2000, alpha=0.05, comparisons=1):
    v = np.asarray(values, float)
    if v.ndim != 1 or len(v) < 2 or not np.isfinite(v).all():
        raise ValueError("Need >=2 independent prompt scalars")
    if replicates < 2 or not 0 < alpha < 1 or comparisons < 1:
        raise ValueError("Invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    draws = np.array([v[rng.integers(len(v), size=len(v))].mean() for _ in range(replicates)])
    tail = alpha / (2 * comparisons)
    return {
        "estimate": float(v.mean()),
        "lower": float(np.quantile(draws, tail)),
        "upper": float(np.quantile(draws, 1 - tail)),
        "unit": "prompt",
        "method": "percentile_bootstrap_signed_fixed_endpoint_bonferroni",
        "n": len(v),
        "comparisons": comparisons,
        "seed": seed,
    }


def two_bank_bootstrap(h, seed=0, replicates=2000, alpha=0.05):
    """Optional full H: independent row and column prompt resampling."""
    h = np.asarray(h, float)
    if h.ndim != 2 or min(h.shape) < 2 or not np.isfinite(h).all():
        raise ValueError("Need H[D prompts, C prompts], each >=2")
    if replicates < 2 or not 0 < alpha < 1:
        raise ValueError("Invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        d = rng.integers(h.shape[0], size=h.shape[0])
        c = rng.integers(h.shape[1], size=h.shape[1])
        draws.append(h[np.ix_(d, c)].mean())
    return {
        "estimate": float(h.mean()),
        "lower": float(np.quantile(draws, alpha / 2)),
        "upper": float(np.quantile(draws, 1 - alpha / 2)),
        "unit": "independent_C_D_prompts",
    }


def projected_two_bank_variance(d_projection, c_projection, interaction=None):
    """Low-storage summary; omit interaction => explicitly incomplete uncertainty.

    Projections use the OTHER empirical mean. interaction estimates
    tr(Sigma_h M Sigma_delta M^T). An unbiased sample-covariance contraction
    corrects double counting: Var(a)/nD + Var(b)/nC - interaction/(nD*nC).
    """
    d, c = np.asarray(d_projection, float), np.asarray(c_projection, float)
    if (
        d.ndim != 1
        or c.ndim != 1
        or min(len(d), len(c)) < 2
        or not np.isfinite(d).all()
        or not np.isfinite(c).all()
    ):
        raise ValueError("Need finite prompt projections from both banks")
    if not np.isclose(d.mean(), c.mean(), atol=1e-8, rtol=1e-6):
        raise ValueError("Projections do not share an aggregate estimand")
    first = float(d.var(ddof=1) / len(d) + c.var(ddof=1) / len(c))
    if interaction is None:
        return {
            "estimate": float(d.mean()),
            "plug_in_first_order_variance": first,
            "status": "interaction_unresolved_no_full_CI",
        }
    if not np.isfinite(interaction) or interaction < 0:
        raise ValueError("Invalid interaction contraction")
    return {
        "estimate": float(d.mean()),
        "variance_unbiased_estimate": first - interaction / (len(d) * len(c)),
        "status": "moment_estimate_not_automatic_confidence_interval",
    }


def positive_scale(shared, independent, floor=1e-12):
    s, i = np.asarray(shared, float), np.asarray(independent, float)
    if s.shape != i.shape or s.ndim != 1 or not np.isfinite(s).all() or not np.isfinite(i).all():
        raise ValueError("Need matching finite functional vectors")
    norm2 = float(i @ i)
    if norm2 <= floor:
        return {"status": "unresolved_reference", "signed_scale": None, "residual": None}
    signed = float(s @ i / norm2)
    return {
        "status": "descriptive_not_a_significance_test",
        "signed_scale": signed,
        "positive_scale": max(0, signed),
        "residual": float(np.linalg.norm(s - max(0, signed) * i)),
    }


def pilot_decision(evidence):
    """Evidence-bearing decision, never a config boolean masquerading as GO."""
    required = ("implementation", "credit", "non_scalar", "local_steps")
    gates = evidence.get("gates", {})
    for key in required:
        gate = gates.get(key, {})
        if gate.get("status") == "PASS" and not gate.get("evidence"):
            raise ValueError(f"Missing evidence for {key}")
    if any(gates.get(key, {}).get("status") == "FAIL" for key in required):
        return "STOP"
    if all(gates.get(key, {}).get("status") == "PASS" for key in required):
        return "GO"
    return "INCONCLUSIVE"

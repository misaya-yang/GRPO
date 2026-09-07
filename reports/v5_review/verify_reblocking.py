"""Finite K=B=m=2 example: random reblocking of independent strata, CPU only."""

import itertools
import json
from pathlib import Path

import numpy as np

EPSILON = 1e-6


def kernel(u, v):
    # Original Bernoulli(1/2) score, not a stratum-conditional score.
    return (u - v) * ((u - 0.5) - (v - 0.5)) / (2 * (1 + 2 * EPSILON))


result = {
    "status": "CPU_FINITE_EXAMPLE_ONLY",
    "K": 2,
    "B": 2,
    "m": 2,
    "epsilon": EPSILON,
    "target_mean": 0.25 / (1 + 2 * EPSILON),
    "cases": [],
}
for probabilities in ((0.2, 0.8), (0.5, 0.5)):
    values, masses = [], []
    for bits in itertools.product((0, 1), repeat=4):
        z = np.array(bits).reshape(2, 2)
        masses.append(
            np.prod(
                [
                    probabilities[j] if z[b, j] else 1 - probabilities[j]
                    for b in range(2)
                    for j in range(2)
                ]
            )
        )
        cross = np.mean([kernel(z[0, j], z[1, k]) for j in range(2) for k in range(2)])
        swapped = z.copy()
        swapped[:, 1] = z[::-1, 1]
        other = np.mean([kernel(swapped[0, j], swapped[1, k]) for j in range(2) for k in range(2)])
        complete = (
            0.25 * kernel(z[0, 0], z[1, 0])
            + 0.25 * kernel(z[0, 1], z[1, 1])
            + 0.5 * np.mean([kernel(z[b, 0], z[c, 1]) for b in range(2) for c in range(2)])
        )
        assert abs(complete - (cross + other) / 2) < 1e-14
        values.append((cross, complete))
    values, masses = np.array(values), np.array(masses)
    means = masses @ values
    variances = masses @ (values - means) ** 2
    assert np.max(abs(means - result["target_mean"])) < 1e-14
    assert variances[1] < variances[0]
    result["cases"].append(
        {
            "stratum_success_probabilities": probabilities,
            "means": means.tolist(),
            "variance_cross": float(variances[0]),
            "variance_reblocked": float(variances[1]),
            "variance_ratio": float(variances[1] / variances[0]),
            "enumerated_banks": len(masses),
        }
    )
iid_values = [
    np.mean([kernel(bits[i], bits[j]) for i, j in itertools.combinations(range(4), 2)])
    for bits in itertools.product((0, 1), repeat=4)
]
result["iid_all_variance"] = float(np.var(iid_values))
for case in result["cases"]:
    case["reblocked_over_iid_all"] = case["variance_reblocked"] / result["iid_all_variance"]
result["caveat"] = "K2 finite example, not verification of v5 K4 oracle or LLM benefit."
Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))

"""Executable finite-policy audit. Results are CPU evidence only."""

import itertools

import numpy as np

from .artifacts import provenance, write_json
from .estimators import advantages
from .prediction import standardized_forecast
from .theory import count_counterexample, geometry, safety_bound, stratified_pair


def coverage(q, reward):
    """Dossier coverage object Pr(max_i r_i = 1) for a pair law."""
    reward = np.asarray(reward, float)
    failures = np.flatnonzero(reward == 0)
    return float(1 - q[np.ix_(failures, failures)].sum())


def run_exact(output):
    p, s, r, qa, qb = count_counterexample()
    ga, gb = geometry(p, s, r, qa), geometry(p, s, r, qb)
    np.testing.assert_allclose([ga["g"][0], ga["gq"][0], gb["gq"][0]], [-0.5, 0.125, -1.375])
    # Dossier Section 7.4: both laws beat IID coverage 3/4 while sharing counts.
    iid_coverage = float(1 - (1 - p @ r) ** 2)
    np.testing.assert_allclose(
        [coverage(qa, r), coverage(qb, r), iid_coverage], [7 / 8, 7 / 8, 3 / 4]
    )
    qstrat, _ = stratified_pair(p, 2)
    reward_one = [1.0, 0.0, 0.0, 0.0]
    strat = geometry(p, [1.0, -4.0, 3.0, 0.0], reward_one, qstrat)
    np.testing.assert_allclose([strat["g"][0], strat["gq"][0]], [0.25, -0.125])
    # Dossier Section 7.5: coverage rises 7/16 -> 1/2 while the update reverses.
    strat_iid_coverage = float(1 - (1 - np.dot(p, reward_one)) ** 2)
    np.testing.assert_allclose([coverage(qstrat, reward_one), strat_iid_coverage], [0.5, 7 / 16])
    rng = np.random.default_rng(2027)
    errors = []
    for k in (2, 4, 8):
        prob = rng.uniform(0.05, 0.95, size=k)
        pos, neg = rng.normal(size=(2, k, 3))
        brute = np.zeros(3)
        for labels in itertools.product((0, 1), repeat=k):
            labels = np.array(labels)
            mass = np.prod(np.where(labels, prob, 1 - prob))
            scores = np.where(labels[:, None], pos, neg)
            a = advantages(labels[None, :], kind="standardized")[0]
            brute += mass * np.mean(a[:, None] * scores, axis=0)
        errors.append(float(np.max(np.abs(brute - standardized_forecast(prob, pos, neg)))))
    assert max(errors) < 1e-12
    result = {
        "evidence_tier": "cpu_exact_enumeration",
        "provenance": provenance(),
        "count_example": {
            "true": ga["g"],
            "A": ga["gq"],
            "B": gb["gq"],
            "count_profile": [0.125, 0.75, 0.125],
            "coverage": {"A": coverage(qa, r), "B": coverage(qb, r), "iid": iid_coverage},
        },
        "stratified_example": {
            **strat,
            "coverage": {
                "stratified": coverage(qstrat, reward_one),
                "iid": strat_iid_coverage,
            },
        },
        "normalized_forecast_max_error": max(errors),
        "stratified_bound": safety_bound(2, strat["eta"]),
        "pretrained_model_experiment": "not_run",
    }
    write_json(output, result)
    return result

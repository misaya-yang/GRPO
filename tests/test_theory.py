import itertools

import numpy as np
import pytest

from dependent_rollouts.estimators import advantages, coefficient, gradient, rewire_indices
from dependent_rollouts.prediction import iid_forecast_weights, standardized_forecast
from dependent_rollouts.theory import (
    count_counterexample,
    geometry,
    pair_operator,
    safety_bound,
    stratified_pair,
)


def test_same_count_different_direction():
    p, scores, reward, qa, qb = count_counterexample()
    counts = reward[:, None] + reward[None, :]
    for q, expected in ((qa, 0.125), (qb, -1.375)):
        np.testing.assert_allclose(q.sum(0), p)
        np.testing.assert_allclose([q[counts == n].sum() for n in range(3)], [0.125, 0.75, 0.125])
        answer = geometry(p, scores, reward, q)
        assert answer["g"][0] == -0.5
        assert answer["gq"][0] == expected
        direct = 0.0
        for i, j in itertools.product(range(4), repeat=2):
            direct += q[i, j] * (scores[i] - scores[j]) * (reward[i] - reward[j]) / 2
        assert direct == expected
        assert answer["slope"] == pytest.approx(answer["positive"] - answer["interference"])


@pytest.mark.parametrize("kind", ["rloo", "standardized"])
@pytest.mark.parametrize("k", [2, 4, 8])
def test_binary_count_identity(kind, k):
    rng = np.random.default_rng(42)
    for n in range(k + 1):
        r = np.array([1] * n + [0] * (k - n))[None, :]
        scores = rng.normal(size=(1, k, 3))
        expected = (
            np.zeros(3)
            if n in (0, k)
            else coefficient(n, k, kind) * (scores[0, :n].mean(0) - scores[0, n:].mean(0))
        )
        np.testing.assert_allclose(gradient(r, scores, kind=kind), expected, atol=1e-14)


@pytest.mark.parametrize("kind", ["rloo", "standardized"])
def test_exhaustive_rewiring(kind):
    rewards = np.array([[1, 0], [1, 1], [0, 0]])
    scores = np.arange(12).reshape(3, 2, 2) ** 2
    flat = rewards.ravel()
    positives, negatives = np.flatnonzero(flat), np.flatnonzero(1 - flat)
    values = []
    for pos in itertools.permutations(positives):
        for neg in itertools.permutations(negatives):
            idx = np.arange(6)
            idx[positives], idx[negatives] = pos, neg
            values.append(gradient(rewards, scores.reshape(6, 2)[idx].reshape(3, 2, 2), kind=kind))
    np.testing.assert_allclose(
        np.mean(values, axis=0), gradient(rewards, scores, arm="count", kind=kind)
    )
    idx = rewire_indices(rewards, np.random.default_rng(0))
    np.testing.assert_array_equal(flat[idx], flat)
    np.testing.assert_array_equal(np.sort(idx), np.arange(6))
    lengths = np.array([1, 5, 3, 7, 2, 4])
    assert lengths[idx].sum() == lengths.sum()


def test_cross_baseline_exact_unbiased_and_guard():
    p, s, r, q, _ = count_counterexample()
    expected = 0.0
    for i, j, a, b in itertools.product(range(4), repeat=4):
        rr = r[np.array([[i, j], [a, b]])]
        ss = s[np.array([[i, j], [a, b]])][..., None]
        expected += q[i, j] * q[a, b] * gradient(rr, ss, arm="cross")[0]
    assert expected == pytest.approx(p @ (s * r))
    with pytest.raises(ValueError):
        advantages([[0, 1], [1, 0]], arm="cross", kind="standardized")
    with pytest.raises(ValueError):
        advantages([[0, 1]], arm="cross")


@pytest.mark.parametrize("k", [2, 4, 8])
def test_normalized_forecast_exhaustive(k):
    rng = np.random.default_rng(k)
    p = rng.uniform(0.01, 0.99, size=k)
    pos, neg = rng.normal(size=(2, k, 5))
    brute = np.zeros(5)
    for labels in itertools.product((0, 1), repeat=k):
        r = np.array(labels)
        prob = np.prod(np.where(r, p, 1 - p))
        s = np.where(r[:, None], pos, neg)
        brute += prob * gradient(r[None, :], s[None, :, :], kind="standardized")
    np.testing.assert_allclose(standardized_forecast(p, pos, neg), brute, atol=1e-13)


@pytest.mark.parametrize("kind", ["rloo", "standardized"])
def test_iid_only_virtual_forecast(kind):
    r = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    bins = np.array([0, 0, 1, 1, 2, 2])
    s = np.arange(18).reshape(6, 3) ** 2
    brute = np.zeros(3)
    for idx in itertools.product([0, 1], [2, 3], [4, 5]):
        idx = np.array(idx)
        brute += gradient(r[idx][None, :], s[idx][None, :, :], kind=kind) / 8
    weights = iid_forecast_weights(r, bins, 3, kind)
    np.testing.assert_allclose(weights @ s, brute, atol=1e-13)
    with pytest.raises(ValueError, match="empty strata"):
        iid_forecast_weights(r, bins, 4, kind)


def test_projection_saturation_and_spectral_boundary():
    rng = np.random.default_rng(2027)
    for n in (4, 7):
        p = rng.dirichlet(np.ones(n))
        q, _ = stratified_pair(p, 4)
        t = pair_operator(p, q)
        whitened = np.sqrt(p)[:, None] * t / np.sqrt(p)[None, :]
        eig = np.linalg.eigvalsh(whitened)
        assert eig.min() >= -1 / 3 - 1e-12
        assert eig.max() <= 1 + 1e-12
        saturated = np.eye(n) - p[None, :]
        for _ in range(20):
            r = rng.normal(size=n)
            full = geometry(p, saturated, r, q)
            assert full["slope"] >= -1e-12
            assert abs(full["interference"]) < 1e-12
            s = rng.normal(size=(n, 2))
            s -= p @ s
            limited = geometry(p, s, r, q)
            v = limited["projection"] @ (r - p @ r)
            assert limited["slope"] / (v @ (p * v)) >= safety_bound(4, limited["eta"]) - 1e-10


@pytest.mark.parametrize("reward", [0, 1])
def test_degenerate_pool(reward):
    np.testing.assert_array_equal(advantages(np.full((3, 4), reward), arm="count"), 0)


def test_iid_mixture_full_support_preserves_reversal():
    # Dossier Section 7.4: mixing each pair law with 10% IID gives full pair
    # support and preserves the reversal of Q_A.
    p, scores, reward, qa, qb = count_counterexample()
    iid = np.outer(p, p)
    for q, expected_sign in ((qa, 1), (qb, -1)):
        mixed = 0.9 * q + 0.1 * iid
        np.testing.assert_allclose(mixed.sum(1), p)
        assert np.all(mixed > 0)
        answer = geometry(p, scores, reward, mixed)
        assert answer["gq"][0] * expected_sign > 0


def test_dose_linearity_and_preregistered_values():
    # Dossier Section 8.5: A^(lam) is a convex mixture; gradient linearity in
    # lam is an algebraic identity, not an experimental discovery.
    rewards = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    rng = np.random.default_rng(9)
    scores = rng.normal(size=(4, 2, 3))
    within = gradient(rewards, scores, arm="within")
    count = gradient(rewards, scores, arm="count")
    for lam in (0.0, 0.5, 1.0):
        np.testing.assert_allclose(
            gradient(rewards, scores, arm="dose", lam=lam), (1 - lam) * count + lam * within
        )
    with pytest.raises(ValueError):
        advantages(rewards, arm="dose")
    with pytest.raises(ValueError):
        advantages(rewards, arm="dose", lam=1.5)

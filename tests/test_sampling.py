import itertools
from fractions import Fraction

import numpy as np
import pytest

from dependent_rollouts.sampling import (
    ArithmeticDecoder,
    cdf_from_probs,
    latent_group,
    sample_categorical,
)


@pytest.mark.parametrize("kind", ["iid", "stratified", "lattice"])
def test_member_marginals_and_group_permutation(kind):
    p = np.array([0.1, 0.2, 0.3, 0.4])
    samples = sample_categorical(p, kind, 4, 3000, 2027)
    for slot in range(4):
        np.testing.assert_allclose(
            np.bincount(samples[:, slot], minlength=4) / len(samples), p, atol=0.035
        )
    strata = [u.stratum(4) for u in latent_group(kind, 4, 93)]
    if kind != "iid":
        assert sorted(strata) == list(range(4))
        assert strata != list(range(4))


def test_long_prefix_does_not_exhaust_uniform():
    decoder = ArithmeticDecoder(latent_group("iid", 2, 10)[0])
    tokens = [decoder.step([0.0, 0.5, 1.0]) for _ in range(2048)]
    assert decoder.latent.source.bits >= 2048
    assert 0.45 < np.mean(tokens) < 0.55
    assert decoder.high - decoder.low == Fraction(1, 2**2048)


def test_tree_leaf_marginals():
    counts = {leaf: 0 for leaf in itertools.product(range(2), repeat=3)}
    for seed in range(2500):
        decoder = ArithmeticDecoder(latent_group("iid", 2, seed)[0])
        leaf = tuple(decoder.step([0.0, 0.25, 1.0]) for _ in range(3))
        counts[leaf] += 1
    for leaf, count in counts.items():
        expected = np.prod([0.25 if x == 0 else 0.75 for x in leaf])
        assert abs(count / 2500 - expected) < 0.03


def test_positive_tail_support_is_preserved():
    samples = sample_categorical([1.0, 1e-30], "iid", 2, 4, 7)
    assert np.all(samples == 0)


def test_invalid_probability_values_are_rejected():
    with pytest.raises(ValueError):
        cdf_from_probs([1.0, float("nan")])


def test_lattice_orbit_null_mode():
    # Reward has period 1/2 on four equiprobable leaves: every K=2 orbit agrees.
    samples = sample_categorical([0.25] * 4, "lattice", 2, 100, 44)
    rewards = np.array([0, 1, 0, 1])[samples]
    assert np.all(rewards[:, 0] == rewards[:, 1])


def test_stratified_pair_law_on_small_tree():
    # Dossier Section 11.2: pair-law check on a small autoregressive tree.
    # Uniform two-step binary tree; K=2 bins align with the leaf halves, so each
    # group holds exactly one leaf from {0,1} and one from {2,3}.
    samples = sample_categorical([0.25] * 4, "stratified", 2, 2000, 7)
    left = samples < 2
    assert np.all(left[:, 0] != left[:, 1])
    assert 0.4 < left[:, 0].mean() < 0.6  # random labeling of the slots
    frequencies = np.bincount(samples[:, 0] * 4 + samples[:, 1], minlength=16).reshape(4, 4)
    frequencies = frequencies / len(samples)
    expected = np.zeros((4, 4))
    expected[:2, 2:] = expected[2:, :2] = 1 / 8
    np.testing.assert_allclose(frequencies, expected, atol=0.03)


def test_lattice_pair_law_on_small_tree():
    # K=2 orbits pair each leaf with its first-token flip, uniformly over the
    # four ordered orbit pairs after random labeling.
    samples = sample_categorical([0.25] * 4, "lattice", 2, 2000, 8)
    assert np.all((samples[:, 1] - samples[:, 0]) % 4 == 2)
    frequencies = np.bincount(samples[:, 0] * 4 + samples[:, 1], minlength=16).reshape(4, 4)
    frequencies = frequencies / len(samples)
    expected = np.zeros((4, 4))
    expected[(0, 2, 1, 3), (2, 0, 3, 1)] = 0.25
    np.testing.assert_allclose(frequencies, expected, atol=0.03)

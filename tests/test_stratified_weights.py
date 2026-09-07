import itertools
import math

import numpy as np
import pytest

from stratified_grpo.weights import (
    cross_weights,
    full_stratified_grid_weights,
    full_stratified_weights,
    iid_weights,
)


def _compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in _compositions(total - first, parts - 1):
            yield (first, *tail)


def _advantage(focal, partner_successes, k, epsilon):
    successes = focal + partner_successes
    if successes in (0, k):
        return 0.0
    return (focal - successes / k) / (math.sqrt(successes * (k - successes)) / k + epsilon)


def _brute_full(rewards, k, epsilon):
    rewards = np.asarray(rewards, dtype=int)
    blocks, strata = rewards.shape
    result = np.zeros_like(rewards, dtype=float)
    for counts in _compositions(k, strata):
        label_probability = math.factorial(k) / strata**k
        for count in counts:
            label_probability /= math.factorial(count)
        choices = [list(itertools.combinations(range(blocks), count)) for count in counts]
        denominator = math.prod(math.comb(blocks, count) for count in counts)
        for subsets in itertools.product(*choices):
            selected = [
                (block, stratum) for stratum, subset in enumerate(subsets) for block in subset
            ]
            successes = sum(rewards[index] for index in selected)
            for index in selected:
                result[index] += (
                    label_probability
                    / denominator
                    / k
                    * _advantage(rewards[index], successes - rewards[index], k, epsilon)
                )
    return rewards.size * result


def _brute_cross(rewards, k, epsilon):
    rewards = np.asarray(rewards, dtype=int)
    blocks, strata = rewards.shape
    result = np.zeros_like(rewards, dtype=float)
    for focal_block in range(blocks):
        available = [block for block in range(blocks) if block != focal_block]
        cases = 0
        for partner_blocks in itertools.combinations(available, k - 1):
            for labels in itertools.product(range(strata), repeat=k - 1):
                partner_successes = sum(
                    rewards[block, label]
                    for block, label in zip(partner_blocks, labels, strict=True)
                )
                for stratum in range(strata):
                    result[focal_block, stratum] += _advantage(
                        rewards[focal_block, stratum], partner_successes, k, epsilon
                    )
                cases += 1
        result[focal_block] /= cases
    return result


def _brute_iid(rewards, k, epsilon):
    rewards = np.asarray(rewards, dtype=int)
    flat = rewards.reshape(-1)
    result = np.zeros(flat.shape)
    for focal_index, focal in enumerate(flat):
        available = [index for index in range(flat.size) if index != focal_index]
        cases = list(itertools.combinations(available, k - 1))
        result[focal_index] = np.mean(
            [
                _advantage(focal, sum(flat[index] for index in partners), k, epsilon)
                for partners in cases
            ]
        )
    return result.reshape(rewards.shape)


def _grid_advantage(focal, partner_sum, partner_square, k, scale, epsilon):
    total = focal + partner_sum
    square = focal * focal + partner_square
    discriminant = k * square - total * total
    if discriminant == 0:
        return 0.0
    return (focal - total / k) / (math.sqrt(discriminant) / k + scale * epsilon)


def _brute_grid(rewards, k, scale, epsilon):
    rewards = np.asarray(rewards, dtype=int)
    blocks, strata = rewards.shape
    result = np.zeros(rewards.shape + (3,))
    for counts in _compositions(k, strata):
        label_probability = math.factorial(k) / strata**k
        for count in counts:
            label_probability /= math.factorial(count)
        choices = [list(itertools.combinations(range(blocks), count)) for count in counts]
        denominator = math.prod(math.comb(blocks, count) for count in counts)
        for subsets in itertools.product(*choices):
            selected = [
                (block, stratum) for stratum, subset in enumerate(subsets) for block in subset
            ]
            total = sum(int(rewards[index]) for index in selected)
            square = sum(int(rewards[index]) ** 2 for index in selected)
            for index in selected:
                focal = int(rewards[index])
                advantage = _grid_advantage(
                    focal, total - focal, square - focal * focal, k, scale, epsilon
                )
                result[index] += (
                    rewards.size
                    * label_probability
                    / denominator
                    / k
                    * np.array([advantage, max(advantage, 0.0), min(advantage, 0.0)])
                )
    return result


@pytest.mark.parametrize("blocks,strata,k", [(2, 2, 2), (4, 2, 4), (4, 3, 3), (5, 2, 4)])
def test_binary_weights_match_independent_subset_enumeration(blocks, strata, k):
    rewards = np.random.default_rng(blocks * 10 + strata + k).integers(0, 2, size=(blocks, strata))
    epsilon = 3e-4
    np.testing.assert_allclose(
        full_stratified_weights(rewards, k, epsilon),
        _brute_full(rewards, k, epsilon),
        atol=2e-13,
    )
    np.testing.assert_allclose(
        cross_weights(rewards, k, epsilon),
        _brute_cross(rewards, k, epsilon),
        atol=2e-13,
    )
    np.testing.assert_allclose(
        iid_weights(rewards, k, epsilon),
        _brute_iid(rewards, k, epsilon),
        atol=2e-13,
    )


def test_weights_use_mean_over_all_n_responses():
    rewards = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    scores = np.arange(24, dtype=float).reshape(4, 2, 3) / 7
    weights = full_stratified_weights(rewards, 4)
    actual = (weights[..., None] * scores).mean(axis=(0, 1))
    brute = (_brute_full(rewards, 4, 1e-6)[..., None] * scores).mean(axis=(0, 1))
    np.testing.assert_allclose(actual, brute, atol=2e-13)
    assert weights.sum() == pytest.approx(0.0, abs=2e-13)


@pytest.mark.parametrize(
    "rewards,k,scale",
    [
        (np.array([[0, 1], [1, 0]]), 2, 1),
        (np.array([[0, 3], [1, 2], [2, 1], [3, 0]]), 4, 3),
        (np.array([[0, 2, 1], [2, 1, 0], [1, 0, 2]]), 3, 2),
    ],
)
def test_grid_positive_negative_weights_match_brute_subsets(rewards, k, scale):
    epsilon = 2e-5
    actual = full_stratified_grid_weights(rewards, k, scale, epsilon)
    expected = _brute_grid(rewards, k, scale, epsilon)
    np.testing.assert_allclose(actual, expected, atol=3e-13)
    np.testing.assert_array_equal(actual[..., 0], actual[..., 1] + actual[..., 2])
    assert np.all(actual[..., 1] >= 0)
    assert np.all(actual[..., 2] <= 0)


@pytest.mark.parametrize(
    "function,args",
    [
        (full_stratified_weights, ([[0, 2], [1, 0]], 2)),
        (full_stratified_weights, ([[0, 1]], 2)),
        (full_stratified_weights, ([[0, 1], [1, 0]], True)),
        (full_stratified_weights, ([[0, 1], [1, 0]], 2, -1)),
        (full_stratified_weights, ([[0, 1], [1, 0]], 2, float("nan"))),
        (full_stratified_weights, ([[0, 1], [1, 0]], 2, True)),
        (cross_weights, ([[0, 1]], 2)),
        (iid_weights, ([[0, 1]], 3)),
        (full_stratified_grid_weights, ([[0.0, 1.0], [1.0, 0.0]], 2, 1)),
        (full_stratified_grid_weights, ([[0, 2], [1, 0]], 2, 1)),
        (full_stratified_grid_weights, ([[0, 1], [1, 0]], 2, 0)),
    ],
)
def test_invalid_weight_inputs_are_rejected(function, args):
    with pytest.raises(ValueError):
        function(*args)


def test_grid_state_budget_fails_closed():
    rewards = np.array([[0, 3], [1, 2], [2, 1], [3, 0]])
    with pytest.raises(ValueError, match="state budget"):
        full_stratified_grid_weights(rewards, 4, 3, max_states=1)

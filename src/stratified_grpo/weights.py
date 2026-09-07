"""Exact fixed-bank weights for stratified and strong-IID GRPO estimators.

Every public function returns one weight per actual response.  The estimator
contract is ``mean(weights * scores)`` over all ``N = B * m`` responses; callers
must not divide by the virtual group size K again.

``full_stratified_weights`` requires all B*m responses to be mutually independent
conditional on frozen history. ``cross_weights`` instead requires independent
blocks (axis 0); responses inside one block may be dependent because a virtual
group uses at most one response from each selected block. ``iid_weights`` is the
strong-IID-all control and is valid only when every pooled response is IID.
"""

from math import comb, sqrt

import numpy as np


def _validate_epsilon(epsilon):
    if (
        isinstance(epsilon, (bool, np.bool_))
        or not isinstance(epsilon, (int, float, np.integer, np.floating))
        or not np.isfinite(epsilon)
        or epsilon < 0
    ):
        raise ValueError("epsilon must be finite and nonnegative")
    return float(epsilon)


def _validate_k(k, maximum):
    if type(k) is not int or not 2 <= k <= maximum:
        raise ValueError(f"Require integer 2 <= K <= {maximum}")


def _binary_bank(rewards):
    values = np.asarray(rewards)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("Expected a nonempty binary B x m array")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("Expected binary rewards")
    return values.astype(np.int8, copy=False)


def _hypergeom(population, successes, draws):
    if not 0 <= successes <= population or not 0 <= draws <= population:
        raise ValueError("Invalid hypergeometric parameters")
    denominator = comb(population, draws)
    return np.array(
        [
            comb(successes, count) * comb(population - successes, draws - count) / denominator
            if 0 <= count <= successes and 0 <= draws - count <= population - successes
            else 0.0
            for count in range(draws + 1)
        ]
    )


def _advantage(focal, partner_successes, k, epsilon):
    successes = focal + partner_successes
    if successes in (0, k):
        return 0.0
    return (focal - successes / k) / (sqrt(successes * (k - successes)) / k + epsilon)


def _label_count_probability(remaining, selected, categories):
    probability = 1.0 / categories
    return (
        comb(remaining, selected)
        * probability**selected
        * (1.0 - probability) ** (remaining - selected)
    )


def full_stratified_weights(rewards, k, epsilon=1e-6):
    """Return B x m binary Strat-full weights.

    B must be at least K. All samples within and across strata must be independent;
    this function must not be used for copied answers, shared random offsets, or
    otherwise dependent strata.
    """
    values = _binary_bank(rewards)
    epsilon = _validate_epsilon(epsilon)
    blocks, strata = values.shape
    _validate_k(k, blocks)
    successes = values.sum(axis=0, dtype=int)
    result = np.zeros(values.shape, dtype=float)

    for focal_stratum in range(strata):
        for focal in (0, 1):
            mask = values[:, focal_stratum] == focal
            if not mask.any():
                continue
            # State is (assigned partner labels, total partner successes).
            distribution = np.zeros((k, k), dtype=float)
            distribution[0, 0] = 1.0
            for stratum in range(strata):
                population = blocks - int(stratum == focal_stratum)
                available_successes = (
                    int(successes[stratum]) - int(stratum == focal_stratum) * focal
                )
                updated = np.zeros_like(distribution)
                categories = strata - stratum
                for used in range(k):
                    remaining = k - 1 - used
                    if remaining < 0:
                        continue
                    for selected in range(remaining + 1):
                        if selected > population:
                            continue
                        label_probability = _label_count_probability(
                            remaining, selected, categories
                        )
                        reward_probability = _hypergeom(population, available_successes, selected)
                        updated[
                            used + selected,
                            : used + selected + 1,
                        ] += label_probability * np.convolve(
                            distribution[used, : used + 1], reward_probability
                        )
                distribution = updated
            partner = distribution[k - 1]
            if not np.isclose(partner.sum(), 1.0, atol=2e-12, rtol=0):
                raise ArithmeticError("Stratified partner law lost probability mass")
            result[mask, focal_stratum] = sum(
                probability * _advantage(focal, count, k, epsilon)
                for count, probability in enumerate(partner)
            )
    return result


def iid_weights(rewards, k, epsilon=1e-6):
    """Return B x m weights for fixed-K IID-all groups drawn from the pooled bank."""
    values = _binary_bank(rewards)
    epsilon = _validate_epsilon(epsilon)
    _validate_k(k, values.size)
    total = int(values.sum())
    result = np.zeros(values.shape, dtype=float)
    for focal in (0, 1):
        mask = values == focal
        if not mask.any():
            continue
        partner = _hypergeom(values.size - 1, total - focal, k - 1)
        result[mask] = sum(
            probability * _advantage(focal, count, k, epsilon)
            for count, probability in enumerate(partner)
        )
    return result


def cross_weights(rewards, k, epsilon=1e-6):
    """Return B x m Strat-cross weights for independent blocks.

    Blocks on axis 0 must be independent. Dependence among the m responses within
    one block is allowed; each virtual K-group selects distinct blocks and at most
    one uniformly labelled response from each selected block.
    """
    values = _binary_bank(rewards)
    epsilon = _validate_epsilon(epsilon)
    blocks, strata = values.shape
    _validate_k(k, blocks)
    result = np.zeros(values.shape, dtype=float)
    for focal_block in range(blocks):
        # Coefficients count subsets of partner blocks; normalize at the end.
        counts = np.zeros((k, k), dtype=float)
        counts[0, 0] = 1.0
        for block in range(blocks):
            if block == focal_block:
                continue
            pass_probability = float(values[block].sum()) / strata
            updated = counts.copy()
            updated[1:] += (1.0 - pass_probability) * counts[:-1]
            updated[1:, 1:] += pass_probability * counts[:-1, :-1]
            counts = updated
        partner = counts[k - 1] / comb(blocks - 1, k - 1)
        if not np.isclose(partner.sum(), 1.0, atol=2e-12, rtol=0):
            raise ArithmeticError("Cross partner law lost probability mass")
        for stratum in range(strata):
            focal = int(values[focal_block, stratum])
            result[focal_block, stratum] = sum(
                probability * _advantage(focal, count, k, epsilon)
                for count, probability in enumerate(partner)
            )
    return result


def _grid_advantage(focal, partner_sum, partner_square_sum, k, scale, epsilon):
    total = focal + partner_sum
    square_total = focal * focal + partner_square_sum
    discriminant = k * square_total - total * total
    if discriminant < 0:
        raise ArithmeticError("Negative exact integer variance")
    if discriminant == 0:
        return 0.0
    return (focal - total / k) / (sqrt(discriminant) / k + scale * epsilon)


def _grid_subset_law(values, draws):
    states = {(0, 0, 0): 1}
    for value in values:
        updated = dict(states)
        for (used, total, square), count in states.items():
            if used < draws:
                key = (used + 1, total + value, square + value * value)
                updated[key] = updated.get(key, 0) + count
        states = updated
    denominator = comb(len(values), draws)
    return {
        (total, square): count / denominator
        for (used, total, square), count in states.items()
        if used == draws
    }


def full_stratified_grid_weights(rewards, k, scale, epsilon=1e-6, max_states=1_000_000):
    """Return B x m x 3 Strat-full weights: mean A, mean A+, mean A-.

    Rewards are exact integer grid units in [0, scale], representing real rewards
    ``value / scale``. Continuous rewards are deliberately rejected rather than
    quantized. The final axis is ordered ``(A, max(A,0), min(A,0))``.
    """
    values = np.asarray(rewards)
    if (
        values.ndim != 2
        or not values.shape[0]
        or not values.shape[1]
        or not np.issubdtype(values.dtype, np.integer)
        or type(scale) is not int
        or scale < 1
        or np.any(values < 0)
        or np.any(values > scale)
    ):
        raise ValueError("Expected integer B x m rewards in [0, scale]")
    if type(max_states) is not int or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    epsilon = _validate_epsilon(epsilon)
    blocks, strata = values.shape
    _validate_k(k, blocks)
    result = np.zeros(values.shape + (3,), dtype=float)

    for focal_stratum in range(strata):
        for focal_value in np.unique(values[:, focal_stratum]):
            focal = int(focal_value)
            distribution = {(0, 0, 0): 1.0}
            for stratum in range(strata):
                population = values[:, stratum].astype(int).tolist()
                if stratum == focal_stratum:
                    population.remove(focal)
                updated = {}
                categories = strata - stratum
                subset_laws = {
                    selected: _grid_subset_law(population, selected)
                    for selected in range(min(k - 1, len(population)) + 1)
                }
                for (used, total, square), mass in distribution.items():
                    remaining = k - 1 - used
                    for selected in range(remaining + 1):
                        if selected > len(population):
                            continue
                        label_probability = _label_count_probability(
                            remaining, selected, categories
                        )
                        for (part_total, part_square), probability in subset_laws[selected].items():
                            key = (
                                used + selected,
                                total + part_total,
                                square + part_square,
                            )
                            updated[key] = (
                                updated.get(key, 0.0) + mass * label_probability * probability
                            )
                if len(updated) > max_states:
                    raise ValueError("Exact grid DP state budget exceeded")
                distribution = updated
            weights = np.zeros(3)
            probability_mass = 0.0
            for (used, total, square), probability in distribution.items():
                if used != k - 1:
                    continue
                advantage = _grid_advantage(focal, total, square, k, scale, epsilon)
                weights += probability * np.array(
                    [advantage, max(advantage, 0.0), min(advantage, 0.0)]
                )
                probability_mass += probability
            if not np.isclose(probability_mass, 1.0, atol=2e-12, rtol=0):
                raise ArithmeticError("Grid partner law lost probability mass")
            # Preserve the algebraic decomposition exactly at storage precision.
            weights[0] = weights[1] + weights[2]
            result[values[:, focal_stratum] == focal, focal_stratum] = weights
    return result

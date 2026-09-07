from fractions import Fraction

import numpy as np
import pytest

from stratified_grpo.sampling import ConditionalInterval, stream_seed


def test_posterior_intersection_and_release_not_width():
    decoder = ConditionalInterval(20, 0, 2)
    token, _ = decoder.step([0.75, 0.25])
    assert token == 0
    assert (decoder.low, decoder.high) == (Fraction(0), Fraction(2, 3))
    assert decoder.release_step is None
    # A tiny interval is still not released merely for being tiny.
    decoder.high = Fraction(1, 10**100)
    decoder.step([0.75, 0.25])
    assert decoder.release_step is None
    decoder = ConditionalInterval(20, 0, 2)
    token, _ = decoder.step([0.25, 0.75])
    assert token in (0, 1)
    if token == 0:
        assert (decoder.low, decoder.high) == (0, 1)
        assert decoder.release_step == 1


def test_private_streams_and_reproducibility():
    seeds = [
        stream_seed(1, "dev", "x", r, arm, b, j)
        for r in range(2)
        for arm in ("iid_all", "stratified_full")
        for b in range(4)
        for j in range(2)
    ]
    assert len(seeds) == len(set(seeds))
    a, b = ConditionalInterval(seeds[0], 1, 2), ConditionalInterval(seeds[0], 1, 2)
    assert [a.step([0.2, 0.3, 0.5])[0] for _ in range(20)] == [
        b.step([0.2, 0.3, 0.5])[0] for _ in range(20)
    ]


def test_two_token_mixture_preserves_joint_law():
    # Parent-dependent second-token distributions exercise posterior updates.
    counts = np.zeros((2, 2))
    repetitions = 5000
    for j in range(2):
        for b in range(repetitions):
            d = ConditionalInterval(stream_seed(17, "cpu", "x", b, "s", 0, j), j, 2)
            first, _ = d.step([0.7, 0.3])
            second, _ = d.step([0.2, 0.8] if first == 0 else [0.6, 0.4])
            counts[first, second] += 1
    np.testing.assert_allclose(counts / (2 * repetitions), [[0.14, 0.56], [0.18, 0.12]], atol=0.015)


def test_invalid_lost_support_rejected():
    with pytest.raises(FloatingPointError):
        ConditionalInterval(1).step([1.0, 1e-100])
    with pytest.raises(ValueError):
        ConditionalInterval(1, 2, 2)

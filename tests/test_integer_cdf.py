import random
from fractions import Fraction

import numpy as np
import pytest

from dependent_rollouts.sampling import BitUniform, Latent, _cdf_boundary_fraction, cdf_from_probs


@pytest.mark.parametrize(
    "values",
    [
        [0.1, 1e-22, 0.9],
        [0.5, 0.5 - 1e-17, 1e-17],
        [1.0, 1e-100],
        [0.0, 0.2, 0.0, 0.8],
        [1.0, np.nextafter(0.0, 1.0)],
    ],
)
def test_actual_boundary_widths_preserve_input_ratios(values):
    cdf, tv = cdf_from_probs(values)
    exact = [Fraction(float(p)) for p in values]
    total = sum(exact)
    assert [cdf[i + 1] - cdf[i] for i in range(len(values))] == [p / total for p in exact]
    assert tv == 0


def test_longdouble_conversion_preserves_ratio():
    value = np.nextafter(np.longdouble(1), np.longdouble(0))
    assert _cdf_boundary_fraction(value) == Fraction(*value.as_integer_ratio())
    assert _cdf_boundary_fraction(value) < 1


def test_crossing_latent_interval_refines():
    source = BitUniform(random.Random(5), bits=1, numerator=0)
    assert Latent(source).stratum(3) in (0, 1)
    assert source.bits > 1


def test_vectorized_cdf_all_boundaries_and_draws_equal_integer_reference():
    from bisect import bisect_right
    from itertools import accumulate

    from dependent_rollouts.sampling import _cdf_bisect_right

    rng = np.random.default_rng(771)
    for size in (2, 31, 1001):
        p = np.exp(rng.uniform(-730, 600, size))
        p[0] = 0
        ratios = [float(v).as_integer_ratio() for v in p]
        exponent = max(d.bit_length() - 1 for _, d in ratios)
        cumulative = [0, *accumulate(n << (exponent - (d.bit_length() - 1)) for n, d in ratios)]
        expected = [Fraction(v, cumulative[-1]) for v in cumulative]
        actual, _ = cdf_from_probs(p)
        assert [actual[i] for i in range(len(actual))] == expected
        for numerator in rng.integers(0, 2**53, size=30):
            value = Fraction(int(numerator), 2**53)
            assert _cdf_bisect_right(actual, value) == bisect_right(expected, value)

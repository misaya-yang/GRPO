"""Lazy-bit sampling from exact integer CDFs of supplied binary64 probabilities.

No positive input weight is lost in accumulation; zero weights stay zero.
"""

import random
from bisect import bisect_right
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np


@dataclass
class BitUniform:
    rng: random.Random
    bits: int = 0
    numerator: int = 0

    def refine(self, count=64):
        self.numerator = (self.numerator << count) | self.rng.getrandbits(count)
        self.bits += count

    def interval(self):
        if self.bits == 0:
            self.refine()
        return (
            Fraction(self.numerator, 1 << self.bits),
            Fraction(self.numerator + 1, 1 << self.bits),
        )


@dataclass
class Latent:
    source: BitUniform
    scale: Fraction = Fraction(1)
    offset: Fraction = Fraction(0)
    bin_id: int | None = None

    def interval(self):
        low, high = self.source.interval()
        return self.offset + low * self.scale, self.offset + high * self.scale

    def stratum(self, k):
        while True:
            low, high = self.interval()
            j = int(low * k)
            if high <= Fraction(j + 1, k):
                return j
            self.source.refine()


def latent_group(kind, k, seed):
    if k < 2 or kind not in ("iid", "stratified", "lattice"):
        raise ValueError("Expected iid/stratified/lattice and K>=2")
    rng = random.Random(seed)
    shared = BitUniform(random.Random(rng.getrandbits(128)))
    members = []
    for j in range(k):
        source = shared if kind == "lattice" else BitUniform(random.Random(rng.getrandbits(128)))
        if kind == "iid":
            members.append(Latent(source))
        else:
            # Shared V in [0,1/K), plus a random labeling, is the same
            # unordered lattice law as (V+j/K) mod 1 for V uniform [0,1).
            members.append(Latent(source, Fraction(1, k), Fraction(j, k), j))
    rng.shuffle(members)
    return members


class _IntegerPrefixes:
    """Compact base-2**32 cumulative integers, reconstructed only when queried."""

    def __init__(self, limbs):
        self.limbs = limbs

    def __len__(self):
        return self.limbs.shape[1]

    def __getitem__(self, index):
        value = 0
        for limb in self.limbs[::-1, index]:
            value = (value << 32) | int(limb)
        return value


class IntegerCDF:
    """Exact binary64-weight CDF; vectorized integer limbs avoid Python per token."""

    def __init__(self, probabilities):
        p = np.ascontiguousarray(probabilities, dtype=np.float64)
        if len(p) >= 2**31:
            raise ValueError("Vocabulary exceeds uint64 accumulation capacity")
        bits = p.view(np.uint64)
        exponents = ((bits >> 52) & 2047).astype(np.int64)
        mantissa = bits & ((1 << 52) - 1)
        mantissa = mantissa | np.where(exponents > 0, np.uint64(1 << 52), np.uint64(0))
        effective = np.maximum(exponents, 1)
        shifts = effective - int(effective[p > 0].min())
        width = int(shifts[p > 0].max()) + 53 + len(p).bit_length()
        count = (width + 31) // 32
        limbs = np.zeros((count, len(p) + 1), dtype=np.uint32)
        carry = np.zeros(len(p), dtype=np.uint64)
        for limb in range(count):
            delta = shifts - 32 * limb
            left = np.clip(delta, 0, 64).astype(np.uint64)
            right = np.clip(-delta, 0, 64).astype(np.uint64)
            weights = np.where(delta >= 0, mantissa << left, mantissa >> right) & np.uint64(
                0xFFFFFFFF
            )
            sums = np.cumsum(weights, dtype=np.uint64) + carry
            limbs[limb, 1:] = (sums & np.uint64(0xFFFFFFFF)).astype(np.uint32)
            carry = sums >> np.uint64(32)
        if np.any(carry):
            raise OverflowError("Integer CDF limb allocation was insufficient")
        self.cumulative = _IntegerPrefixes(limbs)
        self.total = self.cumulative[-1]

    def __len__(self):
        return len(self.cumulative)

    def __getitem__(self, index):
        return Fraction(self.cumulative[index], self.total)


def _cdf_boundary_fraction(value):
    if hasattr(value, "as_integer_ratio"):
        return Fraction(*value.as_integer_ratio())
    return Fraction(value)


def _cdf_bisect_right(cdf, value):
    if isinstance(cdf, IntegerCDF):
        # All cumulative weights are integers; floor gives the exact search key.
        key = value.numerator * cdf.total // value.denominator
        return bisect_right(cdf.cumulative, key)
    return bisect_right(cdf, value)


def cdf_from_probs(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any(p < 0) or not np.any(p > 0):
        raise ValueError("Invalid token probabilities")
    # Binary64 numbers have power-of-two denominators. A common integer scale
    # preserves their ratios exactly without precision-dependent tail repairs.
    # TV=0 refers to normalized supplied binary64 weights, not ideal real softmax.
    return IntegerCDF(p), 0.0


@dataclass
class ArithmeticDecoder:
    latent: Latent
    low: Fraction = field(default_factory=lambda: Fraction(0))
    high: Fraction = field(default_factory=lambda: Fraction(1))

    def step(self, cdf):
        """Certify both latent interval endpoints select the same token."""
        if (
            len(cdf) < 2
            or _cdf_boundary_fraction(cdf[0]) != 0
            or _cdf_boundary_fraction(cdf[-1]) != 1
        ):
            raise ValueError("CDF must span [0,1]")
        width = self.high - self.low
        while True:
            ulow, uhigh = self.latent.interval()
            relative = ((ulow + uhigh) / 2 - self.low) / width
            token = max(0, min(len(cdf) - 2, _cdf_bisect_right(cdf, relative) - 1))
            # Certify the index against preserved boundaries, including when a
            # caller supplies a non-IntegerCDF representation.
            while token > 0 and relative < _cdf_boundary_fraction(cdf[token]):
                token -= 1
            while token < len(cdf) - 2 and relative >= _cdf_boundary_fraction(cdf[token + 1]):
                token += 1
            left = self.low + width * _cdf_boundary_fraction(cdf[token])
            right = self.low + width * _cdf_boundary_fraction(cdf[token + 1])
            if left <= ulow and uhigh <= right and left < right:
                self.low, self.high = left, right
                return token
            self.latent.source.refine()


def sample_categorical(p, kind, k, groups, seed):
    cdf, _ = cdf_from_probs(p)
    rng = random.Random(seed)
    return np.array(
        [
            [ArithmeticDecoder(u).step(cdf) for u in latent_group(kind, k, rng.getrandbits(128))]
            for _ in range(groups)
        ]
    )

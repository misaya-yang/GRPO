"""Lazy-bit, rational-interval reference sequence arithmetic sampler.

Token CDFs approximate softmax in float64; measure their TV discrepancy and
reject lost support. Latent randomness never runs out after 53 bits.
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


def cdf_from_probs(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("Invalid token probabilities")
    p = p / p.sum()
    cdf = np.r_[0.0, np.cumsum(p)]
    cdf[-1] = 1.0
    numerical = np.diff(cdf)
    if np.any(numerical < 0) or np.any((p > 0) & (numerical == 0)):
        raise FloatingPointError("Float64 token CDF lost positive support; no silent fallback")
    tv = float(np.abs(numerical - p).sum() / 2)
    return cdf, tv


@dataclass
class ArithmeticDecoder:
    latent: Latent
    low: Fraction = field(default_factory=lambda: Fraction(0))
    high: Fraction = field(default_factory=lambda: Fraction(1))

    def step(self, cdf):
        """Certify both latent interval endpoints select the same token."""
        if len(cdf) < 2 or cdf[0] != 0 or cdf[-1] != 1:
            raise ValueError("CDF must span [0,1]")
        width = self.high - self.low
        while True:
            ulow, uhigh = self.latent.interval()
            relative = ((ulow + uhigh) / 2 - self.low) / width
            token = max(0, min(len(cdf) - 2, bisect_right(cdf, float(relative)) - 1))
            # Float conversion may round across a boundary; certify the index
            # using rational comparisons to avoid indefinite refinement there.
            while token > 0 and relative < Fraction(float(cdf[token])):
                token -= 1
            while token < len(cdf) - 2 and relative >= Fraction(float(cdf[token + 1])):
                token += 1
            left = self.low + width * Fraction(float(cdf[token]))
            right = self.low + width * Fraction(float(cdf[token + 1]))
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

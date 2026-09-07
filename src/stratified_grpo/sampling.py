"""Independent conditional-interval sampling with fresh bits at every token.

The interval is an exact rational posterior of the stratum event under the
recorded float64 token CDF. No narrow-interval release or shared prefix RNG.
"""

import random
from bisect import bisect_right
from fractions import Fraction

import numpy as np

from dependent_rollouts.sampling import BitUniform, cdf_from_probs
from reward_coupling.bank import digest


def stream_seed(seed, stage, prompt_id, macro, arm, block, stratum):
    return int(digest([seed, stage, prompt_id, macro, arm, block, stratum]), 16)


class ConditionalInterval:
    def __init__(self, seed, stratum=None, strata=2):
        if type(strata) is not int or strata < 1:
            raise ValueError("Positive integer strata required")
        if stratum is not None and (type(stratum) is not int or not 0 <= stratum < strata):
            raise ValueError("Invalid stratum")
        self.rng = random.Random(seed)
        self.low = Fraction(0 if stratum is None else stratum, 1 if stratum is None else strata)
        self.high = Fraction(
            1 if stratum is None else stratum + 1, 1 if stratum is None else strata
        )
        self.steps = 0
        self.release_step = 0 if stratum is None or strata == 1 else None
        self.bits_used = 0

    def step(self, probabilities):
        cdf, tv = cdf_from_probs(probabilities)
        source = BitUniform(self.rng)
        width = self.high - self.low
        while True:
            a, b = source.interval()
            left, right = self.low + width * a, self.low + width * b
            midpoint = (left + right) / 2
            token = min(len(cdf) - 2, max(0, bisect_right(cdf, float(midpoint)) - 1))
            while token > 0 and midpoint < Fraction(float(cdf[token])):
                token -= 1
            while token < len(cdf) - 2 and midpoint >= Fraction(float(cdf[token + 1])):
                token += 1
            c, d = Fraction(float(cdf[token])), Fraction(float(cdf[token + 1]))
            if c <= left and right <= d and d > c:
                break
            source.refine()
        self.low = (max(self.low, c) - c) / (d - c)
        self.high = (min(self.high, d) - c) / (d - c)
        if not 0 <= self.low < self.high <= 1:
            raise FloatingPointError("Invalid posterior interval; no sampler fallback")
        self.steps += 1
        self.bits_used += source.bits
        if self.low == 0 and self.high == 1 and self.release_step is None:
            self.release_step = self.steps
        return token, tv


def generate(
    model,
    prompt_ids,
    eos_ids,
    seed,
    max_new_tokens,
    stratum=None,
    strata=2,
    onset=0,
    use_cache=True,
    deadline=None,
):
    import time

    import torch

    if not prompt_ids or not eos_ids or max_new_tokens < 1 or onset < 0:
        raise ValueError("Invalid generation contract")
    model.eval()
    device = model.get_input_embeddings().weight.device
    decoder = ConditionalInterval(seed, None, strata)
    conditional = ConditionalInterval(stream_seed(seed, "onset", "", 0, "", 0, 0), stratum, strata)
    ids = torch.tensor([prompt_ids], device=device)
    response, logps, tvs = [], [], []
    cache = None
    with torch.no_grad():
        for step in range(max_new_tokens):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Generation deadline")
            output = model(input_ids=ids, past_key_values=cache, use_cache=use_cache)
            logits = output.logits[0, -1].double()
            logp = logits.log_softmax(-1)
            probs = logp.exp().cpu().numpy()
            if not np.isfinite(probs).all() or np.any(probs <= 0):
                raise FloatingPointError("Softmax lost full support")
            engine = decoder if step < onset else conditional
            token, tv = engine.step(probs)
            response.append(token)
            logps.append(float(logp[token]))
            tvs.append(tv)
            if token in eos_ids:
                break
            cache = output.past_key_values if use_cache else None
            ids = torch.tensor([[token]] if use_cache else [prompt_ids + response], device=device)
    return {
        "response_ids": response,
        "sampling_token_logp": logps,
        "old_logp": sum(logps),
        "finish_reason": "eos" if response[-1] in eos_ids else "length",
        "response_mask": [1] * len(response),
        "eos_index_or_null": len(response) - 1 if response[-1] in eos_ids else None,
        "rng_seed": seed,
        "stratum": stratum,
        "onset": onset,
        "release_step": conditional.release_step,
        "release_observed": conditional.release_step is not None,
        "cdf_tv_sum": sum(tvs),
        "cdf_tv_max": max(tvs),
        "fresh_bits_used": decoder.bits_used + conditional.bits_used,
        "interval_numeric_law": "exact_rational_posterior_of_float64_CDF",
    }

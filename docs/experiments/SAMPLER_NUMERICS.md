# Sequence sampler numerical contract

`sampling.py` implements the dossier's single-latent-uniform sequence decoder. IID draws
independent bit streams; stratification independently draws within K bins; lattice uses
one common within-bin offset. All group member labels are randomly permuted. A new group
seed defines each independent original randomization block.

Each latent uniform is represented by a dyadic interval. If that interval straddles a token
boundary, the same latent is refined with more random bits. The selected sequence-prefix
interval uses Python arbitrary-precision `Fraction`. There is no 53-bit exhaustion, interval
underflow, or mid-sequence switch to independent token sampling. PRNG independence is the
ordinary computational sampling assumption. K=8 subsampling is never relabeled native K=4.

Model softmax and CDF are computed in float64, in fixed ascending token-ID order. Therefore
the practical marginal is the **numerical CDF policy**, an approximation to mathematical
softmax. Every step records half the L1 discrepancy between normalized probabilities and
CDF interval widths; `cdf_tv_sum` accumulates the encountered discrepancies. This observed
sum is a diagnostic along sampled prefixes, not a uniform all-prefix theorem bound.
Positive mass rounded to a zero-width bin raises an error. There is no clipping/top-k fix.

Generation uses cached logits; weighted gradients use teacher-forced logits. Stored logp is
checked before every aggregate gradient/update with absolute sequence tolerance 0.005.
This is a practical numerical tolerance, not a proof of exact LLM marginals. Pilot Stage B
must inspect both discrepancies, sampler marginals and throughput on the real checkpoint.

Tests include small-tree leaf probabilities, member marginals and random labeling, lattice
null reward modes, and 2,048-token binary prefixes. Exact finite-policy identities are
verified separately. The implementation is intended as a correctness-oriented reference;
optimize only after end-to-end timing identifies an actual cost blocker.

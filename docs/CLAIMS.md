# Claims table (dossier Section 18)

Statuses: **proven** = derivation checked in dossier/code; **cpu-verified** = exact
enumeration or numerical linear algebra on CPU; **plumbing** = random-model or small-tree
implementation checks only; **untested** = planned pretrained-model evidence, not obtained.
Do not promote a claim across columns without a new receipt under `artifacts/validation/`.

| Claim | Status | Evidence location |
|---|---|---|
| Pairwise operator identity, Eqs. (3)-(4) | proven + cpu-verified | `theory.geometry`; `tests/test_theory.py::test_same_count_different_direction` |
| Score-subspace decomposition Eq. (9) | proven + cpu-verified | `theory.geometry`; slope = positive − interference assertions |
| Theorem 1 (universal local compatibility) | proven | dossier §5.2; block-operator argument reproduced in paper |
| Spectral bound Eq. (6), safety bound Eqs. (14)-(16) | proven + cpu-verified | `theory.safety_bound`; `test_projection_saturation_and_spectral_boundary` |
| Stratified/lattice operators Eqs. (11), (17) | proven | dossier §6; appendix derivations |
| Same-count counterexample, Eqs. in §7.4 (−1/2, 1/8, −11/8; coverage 7/8 vs 3/4; 10% IID mixture keeps reversal) | proven + cpu-verified | `toy.run_exact`; `test_iid_mixture_full_support_preserves_reversal` |
| Stratified variance-warning example §7.5 (1/4, −1/8; coverage 7/16 → 1/2) | proven + cpu-verified | `toy.run_exact` coverage assertions |
| Binary count/identity decomposition Eqs. (18)-(20) | proven + cpu-verified | `estimators.coefficient`; `test_binary_count_identity` |
| Rewiring identity Eqs. (21)-(22) | proven + cpu-verified | exhaustive permutations in `test_exhaustive_rewiring` |
| Normalized IID-only forecast Eq. (23) | proven + cpu-verified | exhaustive 2^K enumeration through K=8 (`toy.run_exact`, `test_normalized_forecast_exhaustive`) |
| Cross-group baseline exact unbiasedness (unnormalized RLOO) | proven + cpu-verified | `test_cross_baseline_exact_unbiased_and_guard` |
| Dose linearity in λ (algebra, not a discovery) | proven + cpu-verified | `test_dose_linearity_and_preregistered_values` |
| Sampler marginals, pair laws on small trees, lazy-bit numerics | plumbing | `tests/test_sampling.py`; random-model `tests/test_llm.py` |
| Pretrained-model marginal/score-point validity | untested | Stage B on the real checkpoint |
| Mechanism effect δ_func > 0.10, prediction gate, consequence gate | untested | pilot Stages C-E |
| Downstream training differences (six arms × three seeds) | untested | Stage F, only after pilot GO |
| GPU throughput / wall-clock budgets | untested | dossier §13.1: benchmark before the matrix |

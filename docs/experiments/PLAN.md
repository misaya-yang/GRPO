# Dossier-derived experimental plan

## Scientific target

Characterize reward-independent positive natural preconditioning versus score-residual
interference; predict sampler-specific updates from IID data; isolate identity selection
with fixed trajectories and complete reward-count profiles. Cross-group baselines are an
identification instrument. Neither known RLOO bias nor a low raw cosine is the main claim.

## Fixed contract

For each prompt and generating checkpoint: temperature 1, full support, disabled dropout,
fixed token-id ordering, identical EOS/cap (160), parameter-independent binary verifier.
The score is the sum of response-token gradients, including emitted EOS. Cap outcomes are
retained and scored as failures. No PPO clipping, off-policy reuse, response-length averaging,
or KL-in-reward in the primary diagnostic. Standardized extensions use population std + 1e-6.
Cross-group correction is only unnormalized RLOO.

W: original group advantages. C: exact conditional mean over binary-slot rewiring.
X: baseline from other independently randomized groups for the same prompt.
W-C identifies finite-pool identity assignment; C-X also contains count/prompt weighting.
All prompts retain equal weight. Finite-pool ratios are not called unbiased population effects.

## Stages and implementation map

| Stage | Code / artifact | Required evidence |
|---|---|---|
| A: CPU exact | `theory.py`, `toy.py`, `tests/test_theory.py` | identities, counterexamples, saturation, bound, exhaustive normalized forecast and rewiring |
| B: sampler/loss | `sampling.py`, `llm.py`, `test_sampling.py`, `test_llm.py` | marginal and pair-law tree checks, long-bit refinement, response/EOS sum, same-checkpoint score agreement |
| C: frozen W/C/X | `collect`, `audit` | unrestricted responses, full-parameter aggregate gradients, raw banks/weights/hash receipts |
| D: independent prediction | IID `audit --arm forecast` | forecast receipt frozen before coupled bank inspection; held-out prompt IID calibration; scalar/count comparators fixed on discovery |
| E: consequences | `branch`, `audit --arm dose`, `statistics.py` | independent IID evaluation; preregistered doses λ∈{0,1/2,1}; verifier reward, KL, ESS/tails; step halving; fixed SGD; uncertainty |
| F: downstream | `train`, `make_training_matrix.py` | only after pilot GO; six arms x three seeds, common IID evaluation |

## Bounded pilot

- Model: Qwen2.5-0.5B-Instruct, full parameters, K=4, IID/stratified/lattice.
- 48 prompts: 16 discovery (8 generated arithmetic + 8 SVAMP), 32 confirmation
  (16 arithmetic + 16 SVAMP). 16 independent groups per prompt per sampler.
- Main bank: 9,216 responses, cap 160, at most 1,474,560 response tokens.
- Calibration/discovery/confirmation/train/evaluation use distinct seeds and partitions.
  Generated tasks also split construction templates. Fixed data deduplicates normalized text.
- Before the first GPU collection, freeze `MANIFEST.md` (dossier Section 18). Before
  confirmation, register the behavioral quantities of dossier Section 9.3: held-out
  verifier reward, probability of a valid final-answer format, and a prespecified
  partition of arithmetic solution/failure types; forecast their changes with
  Δη-style linear predictions from independent IID evaluation banks.
- The implicit training distributions of dossier Section 7.3 (weights w_+, w_- and
  P_Q^+/P_Q^-) are a deferred analysis target: response features must be fixed before
  examining their association with gradient changes, and hand-picked examples never
  substitute for the full-gradient test.
- For each confirmation prompt, forecast from its own IID calibration bank at this checkpoint.
  No claim that one prompt's bin means transfer unchanged to another prompt or checkpoint.
- 16 GPU-hour plan: implementation/throughput 2; collection 6; gradients/prediction 4;
  functional/consequence analysis 4. Costs in dossier are historical assumptions.

## Pilot decision (all four required)

1. Implementation: correct marginal/score point, IID sham compatible with zero, resolved X.
2. Mechanism: prespecified unrestricted-response condition has multiplicity-adjusted lower
   95% confidence bound for non-scalar functional distortion >0.10 after C and scalar control.
3. Prediction: independent functional forecast error improves at least 25% over the
   discovery-frozen scalar/count comparator, with uncertainty supporting improvement.
4. Consequence: pre-predicted verifier-reward direction confirmed, effect at least 10%
   of resolved X reference slope at matched KL; survives half-step/fixed-map check.

Unresolved at cap means underpowered/measurement-limited, not GO. Upper mechanism limit
below 0.10 kills the strong-paper claim in this scope. Only toy/restricted-subspace effects
do not establish an LLM mechanism. Training config stays `pilot_decision=NOT_RUN` until a
real decision is recorded; changing that value alone is not scientific evidence.

## Full study, if gates pass

80-hour total including pilot, allocated as dossier: 16/8/27/8/8/5/8 hours.
Primary six arms = {IID, stratified, lattice} x {within RLOO, cross RLOO}, three seeds.
Each arm: two independent groups of four for every prompt, four prompts/update, starting
proposal 100 updates. Learning rate and common affordable update budget require discovery
calibration on the slowest sampler; template values are smoke starting values.
Then focused generated-task four arms/two seeds, Qwen2.5-1.5B replication, and
SmolLM2-1.7B separate-family frozen audit. Model size does not justify changing only one arm
to LoRA. All downstream metrics use common IID evaluation and record actual total tokens/time.

## Inference and implementation boundaries

Original groups are resampling units within prompts; prompt bootstrap supports prompt
generalization. Rewired and virtual groups add no independent observations. Recompute C
within each bootstrap. Functional norms from estimated gradients need independent bank
replicates/cross-bank products; a descriptive positive norm is not itself a significance test.
The provided component bootstrap handles conditional W-C coordinates. Final multilevel,
multiplicity-adjusted four-gate adjudication remains an analysis step on real pilot data.

Use independent calibration for KL step choice, then freeze it for confirmation. Equal-KL
deterministic comparisons complement common-noise mean shifts. The branch runner reports
actual per-prompt KL/IS overlap and directional finite-difference convergence; it does not
automatically certify overlap, select a favorable dose, or declare a pilot result.

# Initial delivery status

## Completed

- Local Git repository initialized on `main`; no remote repository, commit or push created.
- Original dossier copied byte-for-byte and SHA-256 checked.
- 13 Python source modules: exact theory, lazy sequence sampling, W/C/X losses,
  IID forecast, rational verifier, tasks, immutable receipts, statistical components,
  Transformers rollout/gradient runner, branch intervention, online training, CLI.
- Pilot JSON configurations, two-group six-arm/three-seed matrix generator, SVAMP
  local import and split script, 24 generated arithmetic example prompts.
- Conda wrapper uses `/root/miniconda3/bin/python` directly. Observed dependency
  versions are pinned in the local optional environment; remote packages unchanged.
- Substantive LaTeX first draft in official ICLR 2027 style, with theoretical proofs,
  CPU counterexample table and planned empirical sections. No invented LLM results.

## Verified locally

- `uv run --extra llm pytest -q`: **35 passed** (2.48 seconds for the final full suite).
- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: 29 files formatted.
- Exact gradients reproduced: `(-0.5, 0.125, -1.375)` and `(0.25, -0.125)`.
- Standardized IID-only forecast matched exhaustive enumeration through K=8;
  maximum error `3.3306690738754696e-16` in this run.
- Six tests use a random-weight, tiny Qwen2 model on the **local CPU**, with
  PyTorch 2.8.0 / Transformers 5.15.1. They check plumbing, not pretrained capability.
- CLI help and structural-split task generation exercised.
- LaTeX compiled with bundled Tectonic; PDF: `paper/build/main.pdf`.

Receipts: `artifacts/validation/cpu_exact_final.json`,
`artifacts/validation/server_environment.json`, `artifacts/validation/local_tests.json`.

## Deliberately deferred for the user's real-machine testing

- No remote source deployment, model download, GPU tensor allocation, training or
  pretrained generation was performed. SSH was used only for environment inventory.
- The current server GPU is occupied by other experiments; do not run GPU tests now.
- Before real collection: resolve exact model revision, supply the SVAMP JSON,
  freeze prompt partitions, inspect full-support CDF numerics, and benchmark the
  reference pipeline on the chosen GPU. Config revisions remain explicit placeholders.
- No throughput, memory fit, pilot GO, real-model mechanism effect, causal reward
  consequence, or downstream training claim is established.
- Statistical helpers are components, not an automatic pilot verdict: independent
  bank replication, prompt-level/multiplicity-adjusted analysis, scalar comparator
  freezing and KL calibration must be completed with real pilot data.
- The dossier's additional common-noise mean-shift control and richer analysis/report
  automation are follow-up work. Initial branch code supports deterministic SGD
  directions and full/half steps. Training checkpoints are model-only, without resume.

Start future work with `docs/experiments/RUNBOOK.md`; preserve the current theory and
contract rather than repeating literature exploration.

## Review hardening (2026-09-05)

A dossier-aligned review found no errors in the existing implementation; all changes
below are additive. Existing logic, sampler internals, configs and the frozen dossier
are byte-unchanged.

- Implemented the preregistered dose arm A^(λ)=(1−λ)A^C+λA^W (dossier Section 8.5):
  `estimators.advantages(arm="dose", lam=...)`, `llm.audit` and CLI `audit --arm dose`.
  Preregistered λ ∈ {0, 1/2, 1}; linearity in λ is algebra, not a finding.
- Added coverage diagnostics to the CPU counterexamples (dossier Sections 7.4/7.5):
  count example 7/8 vs IID 3/4; stratified example 1/2 vs IID 7/16.
- Five new tests (35 → 40 passed): full-support IID mixture preserves the Section 7.4
  reversal; dose linearity and input validation; stratified and lattice pair laws on a
  K=2 tree; bank reader rejects subsampled groups (dossier Section 11.2).
- Created `docs/experiments/MANIFEST.md` (dossier Section 18 frozen pilot manifest
  template; items marked TO_FREEZE before first GPU collection, except the recorded
  server snapshot) and `docs/CLAIMS.md` (claim-to-evidence table; no status promotion
  without a new receipt under `artifacts/validation/`).
- PLAN.md: freeze the manifest before first GPU collection; register Section 9.3
  behavioral quantities and Section 7.3 implicit training distributions as deferred
  analysis targets with features fixed before examination.
- RUNBOOK.md: dose arm usage example with the preregistered-λ warning.
- paper/: nine bibliography entries and citation sentences crediting the prior-work
  items listed in dossier Section 19; stratified coverage numbers added to the
  experiments section. No results changed; the draft still contains no LLM results.

Receipts: `artifacts/validation/cpu_exact_review.json`,
`artifacts/validation/local_tests_review.json` (earlier receipts preserved).

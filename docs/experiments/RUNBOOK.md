# Runbook

## Existing server

```bash
ssh -p 27741 root@connect.westc.seetacloud.com
cd /path/to/ICLR2027
/root/miniconda3/bin/python scripts/remote_env_probe.py
bash scripts/conda_run.sh exact --output runs/preflight-exact.json
```

No environment activation is necessary. `RESEARCH_PYTHON=/other/env/bin/python` changes
the interpreter. Source is imported via PYTHONPATH; existing packages stay in place.
Observed baseline is recorded in `artifacts/validation/server_environment.json`.
Do not reinstall generic PyTorch wheels over the CUDA 12.8 build. If modules are absent
on another server, first record that environment, then install only missing compatible packages.

## Prepare tasks and model configuration

Local `uv run dependent-rollouts` and remote `bash scripts/conda_run.sh` are equivalent.
Examples below use remote entry points. Use NEW output paths on reruns.

```bash
bash scripts/conda_run.sh generate-tasks --split discovery --count 8 --seed 202701 --output data/processed/arithmetic-discovery.jsonl
bash scripts/conda_run.sh generate-tasks --split confirmation --count 16 --seed 202702 --output data/processed/arithmetic-confirmation.jsonl
PYTHONPATH=src /root/miniconda3/bin/python scripts/prepare_svamp.py /path/to/SVAMP.json data/processed/svamp
```

Combine corresponding arithmetic/SVAMP JSONL partitions after checking duplicate IDs/text;
`read_tasks` rejects duplicates. Do not filter by sampler disagreement. Calibration-only
intermediate-success filters, if used, must be frozen and accompanied by unfiltered reporting.

If revision is still a placeholder:

```bash
mkdir -p runs/configs
/root/miniconda3/bin/python scripts/pin_model.py configs/pilot_iid.json runs/configs/iid.json
```

Copy that exact revision to stratified and lattice configs, preserving their distinct bank seeds.
Pin once per checkpoint. Configs are deliberately small JSON files, readable without PyYAML.
Full-parameter float32 is the initial 0.5B diagnostic; benchmark throughput and memory before
committing the 48-prompt collection. This reference backend makes no GPU throughput guarantee.

## Forecast before coupled outcomes

```bash
bash scripts/conda_run.sh collect --config runs/configs/iid.json --tasks data/processed/confirmation.jsonl --output runs/confirm-iid
bash scripts/conda_run.sh audit --bank runs/confirm-iid --arm forecast --output runs/confirm-forecast
# Now collect the independently seeded coupled bank
bash scripts/conda_run.sh collect --config runs/configs/stratified.json --tasks data/processed/confirmation.jsonl --output runs/confirm-stratified
bash scripts/conda_run.sh audit --bank runs/confirm-stratified --arm within --output runs/strat-W
bash scripts/conda_run.sh audit --bank runs/confirm-stratified --arm count --output runs/strat-C
bash scripts/conda_run.sh audit --bank runs/confirm-stratified --arm cross --output runs/strat-X
```

Repeat W/C on IID for the sham and W/C/X on lattice. `--kind standardized --epsilon 1e-6`
works for within, count, forecast, and dose. It intentionally rejects cross+standardized.
Preregistered dose mixtures (dossier Section 8.5) reuse the same stored bank:

```bash
bash scripts/conda_run.sh audit --bank runs/confirm-stratified --arm dose --lam 0.5 --output runs/strat-dose-half
```

Use only the preregistered λ values {0, 0.5, 1}; linearity in λ is algebra, not a finding.
Missing IID bins stop forecast construction, rather than assigning invented means.
Each audit is one full aggregate gradient with exact per-trajectory weight receipts;
no per-trajectory parameter gradients are stored. Large outputs live under ignored `runs/`.

## Held-out intervention

Collect a separate IID evaluation bank with a distinct seed. Use a separate IID calibration
bank to choose a conservative step and matched KL sizes; freeze step choices before evaluating
confirmation reward. Defaults are not calibrated learning rates.

```bash
bash scripts/conda_run.sh branch --gradient runs/strat-W --evaluation-bank runs/eval-iid --step 0.00001 --fd-step 0.001 --output runs/strat-W-branch
```

Repeat C/X with the same evaluation trajectories. `directional_scores.jsonl` stores finite
differences at h and h/2; `receipt.json` stores per-prompt unclipped likelihood-ratio reward,
paired standard error, ESS, weight tails and observed sequence KL for a step and half step.
Finite-difference step sizes must show convergence. A low-precision model may quantize away
small perturbations: use float32 and treat failed step-halving as unresolved.

Use `statistics.functional_distortion` on aligned independent-evaluation directional scores.
It labels the output descriptive; it does not turn noisy gradient norms into a population
claim. `statistics.bootstrap_contrast` resamples original groups and recomputes finite-pool C.
Independent bank replication, prompt-level bootstrap, simultaneous comparisons and frozen
scalar comparator evaluation are required before filling the final mechanism table.

## Downstream matrix

```bash
/root/miniconda3/bin/python scripts/make_training_matrix.py configs/train_template.json runs/training-configs
bash scripts/conda_run.sh train --config runs/training-configs/iid-within-202711.json --tasks data/processed/train.jsonl --output runs/train-iid-within-202711
```

The last command refuses while `pilot_decision` is `NOT_RUN`. After a documented GO, freeze
the matched configs and run the matrix within the common token/update budget. AdamW is an
optional downstream optimizer, not the mechanism theorem's fixed update map. Model-only
checkpoints are saved; this initial runner deliberately starts fresh and does not implement
optimizer-state resume. Evaluate final checkpoints with a common IID policy; training
batch reward is not held-out pass@1. Do not compare incomplete runs as a completed matrix.

# Frozen pilot manifest (dossier Section 18)

Fill every entry BEFORE the first GPU collection, then commit this file as a new
version. Do not overwrite history: later changes append a dated revision block.
Every entry must name the exact file, hash, or value that defines it.

| Item | Where it is fixed | Frozen value |
|---|---|---|
| Model revisions (0.5B; later 1.5B, SmolLM2-1.7B) | `configs/pilot_*.json` `revision`, via `scripts/pin_model.py` | TO_FREEZE (40-hex commit SHA) |
| Tokenizer / chat template | receipt `chat_template_sha256` from the pinned revision | TO_FREEZE |
| Dataset splits (calibration/discovery/confirmation/train/evaluation) | `generate-tasks` seeds in RUNBOOK; `prepare_svamp.py` seed and SVAMP source SHA | TO_FREEZE |
| Prompt format | prompt strings in `tasks.py` / `prepare_svamp.py` (chat template applied verbatim) | TO_FREEZE |
| Verifier implementation | `VERIFIER_VERSION = final_tag_rational_ast_v1` in `tasks.py` | TO_FREEZE |
| Token cap and truncation interpretation | `max_new_tokens: 160`; capped responses scored as failures (`tasks.verify`) | TO_FREEZE |
| Sampler specification and token ordering | `sampling.py` lazy-bit Fraction decoder, ascending token-ID CDF order; `SAMPLER_NUMERICS.md` | TO_FREEZE |
| Independent-group definition | one fresh `group_seed` per `collect` group; rewired/virtual groups never independent | TO_FREEZE |
| Loss equation | detached advantage × SUM of response-token log-probs including emitted EOS; no clipping/KL/length-averaging | TO_FREEZE |
| Optimizer / update map | mechanism: fixed SGD map (`branch`); downstream: config `optimizer` | TO_FREEZE |
| Random seeds | config `seed` per bank/arm (IID 202701, stratified 202702, lattice 202703; training 202711-13) | TO_FREEZE |
| Preregistered dose values | `audit --arm dose --lam` with λ ∈ {0, 1/2, 1} | TO_FREEZE |
| Behavioral quantities registered before confirmation (dossier 9.3) | verifier reward; probability of valid final-answer format; prespecified arithmetic solution/failure-type partition | TO_FREEZE |
| Four pilot gates | `PLAN.md` "Pilot decision": implementation, mechanism >0.10, prediction ≥25%, consequence ≥10% | TO_FREEZE |
| Server environment snapshot | `artifacts/validation/server_environment.json` | recorded 2026-09-05 |

A pilot run is only auditable when this manifest is frozen and the rollout banks,
aggregate gradients, and forecast files carry receipts whose hashes match it.

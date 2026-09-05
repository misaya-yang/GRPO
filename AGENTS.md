# ICLR 2027 dependent-rollout research

Use `docs/theory/dependent_rollout_research_dossier.md` as the primary theory and
experiment contract. Preserve its original bytes. Implement this plan directly;
do not reopen broad literature searches unless a concrete correctness blocker appears.

- Preserve existing work; make the smallest sufficient change.
- Separate CPU identities, random-model plumbing, pretrained-model measurements,
  causal consequences, and downstream training evidence.
- Primary loss: on-policy, detached advantages times the **sum** of response-token
  log probabilities including emitted EOS. No length averaging, clipping, KL
  reward, or off-policy epochs in the mechanism study.
- Every prompt is a separate W/C/X pool. Independent original groups own
  uncertainty. Never treat rewired or virtual groups as independent samples.
- No invented experimental results, model revisions, GPU throughput, or pilot GO.
- Match the observed server Conda environment; do not replace its CUDA/PyTorch
  installation. The local `uv` environment is for development.
- Remote training and the full study are separate from preparing code. Preserve
  the dossier's four pilot gates and report unresolved outcomes honestly.

Checks: `make test`, `make lint`, `make exact`; optional LLM plumbing:
`uv sync --extra llm && uv run pytest tests/test_llm.py -q`.

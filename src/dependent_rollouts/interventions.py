"""Deterministic full-parameter branch and independent-IID likelihood-ratio audit."""

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .artifacts import provenance, sha256, write_json, write_jsonl
from .llm import load_model, read_bank, sequence_logp
from .statistics import importance_reward


@contextmanager
def perturb(model, direction, scale):
    """Restore exact original tensor values even if evaluation fails."""
    import torch

    saved = {}
    try:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in direction:
                    if direction[name].shape != parameter.shape:
                        raise ValueError(f"Direction shape mismatch: {name}")
                    saved[name] = parameter.detach().cpu().clone()
                    parameter.add_(
                        direction[name].to(parameter.device, parameter.dtype), alpha=scale
                    )
        yield
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in saved:
                    parameter.copy_(saved[name])


def evaluate_logps(model, rows):
    import torch

    with torch.no_grad():
        return np.array(
            [sequence_logp(model, r["prompt_ids"], r["response_ids"]).item() for r in rows]
        )


def branch(gradient_path, evaluation_bank, output, step, fd_step=1e-3):
    from safetensors.torch import load_file

    direction_path = Path(gradient_path)
    receipt = json.loads((direction_path / "receipt.json").read_text())
    if sha256(direction_path / "gradient.safetensors") != receipt["gradient_sha256"]:
        raise ValueError("Gradient changed since its receipt")
    manifest, pools = read_bank(evaluation_bank)
    config = manifest["config"]
    if config["sampler"] != "iid" or step <= 0 or fd_step <= 0:
        raise ValueError("Need independent IID evaluation and positive steps")
    if any(config[key] != receipt["config"][key] for key in ("model_id", "revision", "dtype")):
        raise ValueError("Direction/evaluation checkpoints or dtypes differ")
    if sha256(Path(evaluation_bank) / "rollouts.jsonl") == receipt["bank_sha256"]:
        raise ValueError("Treatment and evaluation banks must be independent")
    training_manifest, treatment_pools = read_bank(receipt["bank"])
    treatment_seeds = {r["group_seed"] for rows in treatment_pools.values() for r in rows}
    rows = [row for pool in pools.values() for row in pool]
    if any(r["group_seed"] in treatment_seeds for r in rows):
        raise ValueError("Evaluation reuses treatment randomization")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    direction = load_file(str(direction_path / "gradient.safetensors"))
    model, _ = load_model(config)
    baseline = evaluate_logps(model, rows)
    if np.max(np.abs(baseline - [r["old_logp"] for r in rows])) > 5e-3:
        raise ValueError("Evaluation score-point mismatch")
    derivatives = []
    for h in (fd_step, fd_step / 2):
        with perturb(model, direction, h):
            plus = evaluate_logps(model, rows)
        with perturb(model, direction, -h):
            minus = evaluate_logps(model, rows)
        derivatives.append((plus - minus) / (2 * h))
    summaries, values = {}, {}
    for size in (step, step / 2):
        with perturb(model, direction, size):
            new = evaluate_logps(model, rows)
        values[str(size)] = new
        # Keep prompt-level estimates separate. Flat SE is conditional on the
        # fixed prompt allocation; it is not a prompt-generalization interval.
        per_prompt = {}
        for prompt in pools:
            idx = [i for i, row in enumerate(rows) if row["prompt_id"] == prompt]
            per_prompt[prompt] = importance_reward(
                [rows[i]["reward"] for i in idx], baseline[idx], new[idx]
            )
        summaries[str(size)] = per_prompt
    write_jsonl(
        out / "directional_scores.jsonl",
        [
            {
                "trajectory_id": r["trajectory_id"],
                "prompt_id": r["prompt_id"],
                "group_id": r["group_id"],
                "reward": r["reward"],
                "old_logp": baseline[i],
                "directional_score": derivatives[1][i],
                "directional_score_coarse": derivatives[0][i],
                "new_logps": {key: value[i] for key, value in values.items()},
            }
            for i, r in enumerate(rows)
        ],
    )
    result = {
        "status": "conditional_IID_importance_diagnostic",
        "step": step,
        "fixed_update_map": "SGD",
        "fd_step": fd_step,
        "fd_halving_relative_error": float(
            np.linalg.norm(derivatives[0] - derivatives[1])
            / max(np.linalg.norm(derivatives[1]), 1e-12)
        ),
        "per_prompt": summaries,
        "gradient_receipt": receipt,
        "evaluation_sha256": sha256(Path(evaluation_bank) / "rollouts.jsonl"),
        "provenance": provenance(),
    }
    write_json(out / "receipt.json", result)
    return result

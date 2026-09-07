"""Same frozen candidates, exact weights, full-parameter ascent gradients."""

import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from dependent_rollouts.artifacts import provenance, sha256, write_json, write_jsonl

from .bank import group_weights, read_bank, seal
from .sample import load_model, token_logps


def aggregate_gradient(model, weighted, tolerance=1e-5):
    import torch

    model.eval().zero_grad(set_to_none=True)
    if any(not p.requires_grad for p in model.parameters()):
        raise ValueError("Primary gradient scope must contain all actor parameters")
    gradient, hooks = {}, []
    if getattr(model, "_feedback_offload_gradients", False):
        for name, parameter in model.named_parameters():

            def accumulate(p, name=name):
                value = p.grad.detach().float().cpu()
                if not torch.isfinite(value).all():
                    raise FloatingPointError("Nonfinite gradient")
                if name in gradient:
                    gradient[name].add_(value)
                else:
                    gradient[name] = value.clone() if p.device.type == "cpu" else value
                p.grad = None

            hooks.append(parameter.register_post_accumulate_grad_hook(accumulate))
    error = 0.0
    parameter_storage = {p.untyped_storage().data_ptr() for p in model.parameters()}

    def pack(tensor):
        if tensor.device.type == "cpu" or tensor.untyped_storage().data_ptr() in parameter_storage:
            return tensor
        return tensor.device, tensor.to("cpu")

    def unpack(saved):
        return saved[1].to(saved[0]) if isinstance(saved, tuple) else saved

    try:
        for number, (row, weight) in enumerate(weighted):
            if not np.isfinite(weight):
                raise ValueError("Nonfinite detached advantage")
            if getattr(model, "_feedback_progress", False):
                print(
                    json.dumps(
                        {
                            "row": number + 1,
                            "trajectory_id": row.get("trajectory_id"),
                            "backward": bool(weight),
                        }
                    ),
                    flush=True,
                )
            context = (
                torch.autograd.graph.saved_tensors_hooks(pack, unpack)
                if getattr(model, "_feedback_offload_activations", False)
                else nullcontext()
            )
            with context, torch.set_grad_enabled(bool(weight)):
                lp = token_logps(model, row["prompt_ids"], row["response_ids"])
            delta = float(np.max(np.abs(lp.detach().cpu().numpy() - row["sampling_token_logp"])))
            error = max(error, delta)
            if error > tolerance:
                raise ValueError("Per-token score point mismatch")
            sequence_tolerance = getattr(model, "_feedback_sequence_tolerance", None)
            if (
                sequence_tolerance is not None
                and abs(float(lp.detach().sum()) - sum(row["sampling_token_logp"]))
                > sequence_tolerance
            ):
                raise ValueError("Sequence score point mismatch")
            if weight:
                (lp.sum() * float(weight)).backward()
            del lp
    finally:
        for hook in hooks:
            hook.remove()
    for name, p in model.named_parameters():
        if name in gradient:
            continue
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise FloatingPointError("Nonfinite gradient")
        # Explicit zeros make zero-credit banks valid safetensors artifacts.
        gradient[name] = (
            torch.zeros_like(p, device="cpu")
            if p.grad is None
            else p.grad.detach().float().cpu().clone()
        ).contiguous()
    model.zero_grad(set_to_none=True)
    return gradient, error


def dot(left, right):
    if set(left) != set(right) or any(left[n].shape != right[n].shape for n in left):
        raise ValueError("Parameter manifest mismatch")
    return sum(float((left[n].double() * right[n].double()).sum()) for n in left)


def load_gradient(directory):
    from safetensors.torch import load_file

    path = Path(directory)
    receipt = json.loads((path / "receipt.json").read_text())
    if (
        receipt["status"] != "complete"
        or receipt["file"] != "gradient.safetensors"
        or sha256(path / "gradient.safetensors") != receipt["sha256"]
        or sha256(path / "manifest.json") != receipt["manifest_sha256"]
    ):
        raise ValueError("Modified/incomplete gradient")
    return json.loads((path / "manifest.json").read_text()), load_file(
        str(path / "gradient.safetensors")
    )


def audit(bank, output, arm, readouts=()):
    from safetensors.torch import save_file

    if arm not in (
        "shared",
        "independent",
        "rloo",
        "difference",
        "average",
        "suite",
        "full_rubric",
    ):
        raise ValueError("Unknown arm")
    manifest, groups = read_bank(bank)
    config = manifest["config"]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "manifest.json",
        {
            "config": config,
            "bank": str(Path(bank).resolve()),
            "bank_sha256": sha256(Path(bank) / "rows.jsonl"),
            "arm": arm,
            "gradient_convention": "ascent",
            "fixed_map": "identity",
            "scope": "all_actor_parameters",
            "provenance": provenance(config),
        },
    )
    model, _ = load_model(config)
    by_prompt = {}
    for (prompt, _), group in groups.items():
        by_prompt.setdefault(prompt, []).append(group)
    total, weights, projected, max_error = {}, [], [], 0.0
    start = time.monotonic()
    if not readouts and not config.get("save_prompt_gradients", False):
        weighted = []
        for prompt_groups in by_prompt.values():
            for group in prompt_groups:
                a = group_weights(group, config)[arm] / (
                    len(by_prompt) * len(prompt_groups) * len(group)
                )
                weighted.extend(zip(group, a, strict=True))
        model._feedback_progress = True
        model._feedback_offload_activations = config.get("offload_gradients", False)
        total, max_error = aggregate_gradient(model, weighted, config["token_logp_tolerance"])
        weights = [
            {"trajectory_id": r["trajectory_id"], "prompt_id": r["prompt_id"], "weight": float(w)}
            for r, w in weighted
        ]
    else:
        for prompt, prompt_groups in by_prompt.items():
            print(
                json.dumps(
                    {
                        "stage": "gradient",
                        "prompt_id": prompt,
                        "arm": arm,
                        "groups": len(prompt_groups),
                    }
                ),
                flush=True,
            )
            weighted = []
            for group in prompt_groups:
                a = group_weights(group, config)[arm] / (len(prompt_groups) * len(group))
                weighted.extend(zip(group, a, strict=True))
            gradient, error = aggregate_gradient(model, weighted, config["token_logp_tolerance"])
            max_error = max(max_error, error)
            for name, value in gradient.items():
                if name not in total:
                    total[name] = value / len(by_prompt)
                else:
                    total[name].add_(value, alpha=1 / len(by_prompt))
            for directory in readouts:
                rm, rg = load_gradient(directory)
                for key in ("model_id", "model_revision", "tokenizer_revision", "dtype"):
                    if rm["config"][key] != config[key]:
                        raise ValueError("Readout checkpoint mismatch")
                projected.append(
                    {
                        "prompt_id": prompt,
                        "readout": str(directory),
                        "value": dot(rg, gradient),
                        "readout_sha256": sha256(Path(directory) / "gradient.safetensors"),
                        "uncertainty": "conditional_on_frozen_readout",
                    }
                )
                del rg
            if config.get("save_prompt_gradients", False):
                prompt_file = f"prompt-{len(weights):06d}.safetensors"
                save_file(gradient, str(out / prompt_file))
                write_json(
                    out / (prompt_file + ".json"),
                    {"prompt_id": prompt, "sha256": sha256(out / prompt_file)},
                )
            weights.extend(
                {
                    "trajectory_id": r["trajectory_id"],
                    "prompt_id": prompt,
                    "weight": float(w) / len(by_prompt),
                }
                for r, w in weighted
            )
            del gradient
    save_file(total, str(out / "gradient.safetensors"))
    write_jsonl(out / "weights.jsonl", weights)
    write_jsonl(out / "projections.jsonl", projected)
    write_json(
        out / "audit.json",
        {
            "max_token_error": max_error,
            "elapsed_seconds": time.monotonic() - start,
            "gradient_l2": float(np.sqrt(dot(total, total))),
            "prompts": len(by_prompt),
            "activation_storage": "CPU"
            if getattr(model, "_feedback_offload_activations", False)
            else "device",
            "aggregate_storage": "single_global_gradient"
            if not readouts and not config.get("save_prompt_gradients", False)
            else "per_prompt_and_global",
        },
    )
    return seal(out, "gradient.safetensors")


def freeze_readout(config, definitions, output):
    """A signed linear combination of fixed response log likelihoods, from Dev."""
    from safetensors.torch import save_file

    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    model, _ = load_model(config)
    rows = json.loads(Path(definitions).read_text())
    if not rows or any(r.get("split") != "Dev" for r in rows):
        raise ValueError("Readout selection must be frozen on Dev")
    for row in rows:
        row["sampling_token_logp"] = (
            token_logps(model, row["prompt_ids"], row["response_ids"]).detach().cpu().tolist()
        )
    gradient, _ = aggregate_gradient(model, [(r, r["coefficient"]) for r in rows])
    write_json(
        out / "manifest.json",
        {
            "config": config,
            "definitions": rows,
            "definitions_sha256": sha256(definitions),
            "scope": "all_actor_parameters",
            "kind": "Dev_frozen_function",
        },
    )
    save_file(gradient, str(out / "gradient.safetensors"))
    return seal(out, "gradient.safetensors")

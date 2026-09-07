"""Common initial parameters, common eta and eta/2, exact autograd predictions."""

import json
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np

from dependent_rollouts.artifacts import sha256, write_json, write_jsonl
from dependent_rollouts.interventions import perturb
from dependent_rollouts.statistics import importance_reward

from .bank import assert_independent, read_bank
from .sample import load_model, token_logps


def _gradient_metadata(directory):
    path = Path(directory)
    receipt = json.loads((path / "receipt.json").read_text())
    gradient = path / "gradient.safetensors"
    manifest = path / "manifest.json"
    if (
        receipt["status"] != "complete"
        or receipt["file"] != gradient.name
        or sha256(gradient) != receipt["sha256"]
        or sha256(manifest) != receipt["manifest_sha256"]
    ):
        raise ValueError("Modified/incomplete gradient")
    return json.loads(manifest.read_text()), gradient


def _checkpoint_restore_plan(model, config):
    """Map every model parameter to its already pinned local safetensors shard."""
    from safetensors import safe_open

    source = Path(config.get("model_path", ""))
    manifest_path = Path(config.get("checkpoint_manifest", ""))
    if not source.is_dir() or not manifest_path.is_file():
        return None
    if sha256(manifest_path) != config.get("checkpoint_manifest_sha256"):
        raise ValueError("Checkpoint manifest changed")
    files = json.loads(manifest_path.read_text()).get("files", {})
    declared = {name for name in files if name.endswith(".safetensors")}
    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        if index_path.name not in files:
            raise ValueError("Checkpoint index is absent from the pinned manifest")
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
    else:
        if len(declared) != 1:
            raise ValueError("Need one safetensors file or a pinned shard index")
        filename = next(iter(declared))
        with safe_open(source / filename, framework="pt", device="cpu") as handle:
            weight_map = {name: filename for name in handle.keys()}
    parameters = dict(model.named_parameters())
    if set(parameters) != set(weight_map):
        missing = sorted(set(parameters) - set(weight_map))
        extra = sorted(set(weight_map) - set(parameters))
        raise ValueError(
            f"Checkpoint parameter identity mismatch: missing={missing}, extra={extra}"
        )
    by_shard = defaultdict(list)
    for name, filename in weight_map.items():
        if filename not in declared:
            raise ValueError(f"Unpinned checkpoint shard: {filename}")
        by_shard[filename].append(name)
    for filename, names in by_shard.items():
        with safe_open(source / filename, framework="pt", device="cpu") as handle:
            if any(
                tuple(handle.get_slice(name).get_shape()) != tuple(parameters[name].shape)
                for name in names
            ):
                raise ValueError(f"Checkpoint parameter shape mismatch in {filename}")
    return {filename: sorted(names) for filename, names in by_shard.items()}


def restore_from_checkpoint(model, config, restore_plan=None):
    """Restore every parameter from the pinned safetensors checkpoint in place."""
    import torch
    from safetensors import safe_open

    plan = restore_plan or _checkpoint_restore_plan(model, config)
    if plan is None:
        raise ValueError("Checkpoint restoration requires model_path and checkpoint_manifest")
    parameters = dict(model.named_parameters())
    source = Path(config["model_path"])
    with torch.no_grad():
        for filename, names in plan.items():
            with safe_open(source / filename, framework="pt", device="cpu") as handle:
                for name in names:
                    initial = handle.get_tensor(name)
                    if tuple(initial.shape) != tuple(parameters[name].shape):
                        raise ValueError(f"Checkpoint parameter shape changed: {name}")
                    parameters[name].copy_(
                        initial.to(parameters[name].device, parameters[name].dtype)
                    )
    return plan


def checkpoint_parameter_status(model, config, restore_plan=None):
    """Compare live parameters with the pinned checkpoint without keeping a dense copy."""
    import torch
    from safetensors import safe_open

    plan = restore_plan or _checkpoint_restore_plan(model, config)
    if plan is None:
        raise ValueError("Checkpoint comparison requires model_path and checkpoint_manifest")
    parameters = dict(model.named_parameters())
    changed_elements = 0
    changed_tensors = 0
    source = Path(config["model_path"])
    with torch.no_grad():
        for filename, names in plan.items():
            with safe_open(source / filename, framework="pt", device="cpu") as handle:
                for name in names:
                    expected = handle.get_tensor(name).to(
                        parameters[name].device, parameters[name].dtype
                    )
                    changed = int(torch.count_nonzero(parameters[name] != expected))
                    changed_elements += changed
                    changed_tensors += int(changed > 0)
    return {
        "exact": changed_elements == 0,
        "changed_parameter_elements": changed_elements,
        "changed_parameter_tensors": changed_tensors,
        "parameter_elements": sum(parameter.numel() for parameter in parameters.values()),
        "parameter_tensors": len(parameters),
    }


@contextmanager
def _checkpoint_perturb(model, direction, scale, config, restore_plan):
    """Perturb theta0 and restore it from pinned safetensors without a full copy."""
    import torch

    parameters = dict(model.named_parameters())
    if set(parameters) != set(direction) or any(
        tuple(direction[name].shape) != tuple(parameter.shape)
        for name, parameter in parameters.items()
    ):
        raise ValueError("Direction parameter identity/shape mismatch")
    if not np.isfinite(scale):
        raise ValueError("Nonfinite perturbation scale")
    try:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.add_(direction[name].to(parameter.device, parameter.dtype), alpha=scale)
        yield
    finally:
        restore_from_checkpoint(model, config, restore_plan)


def _perturb(model, direction, scale, config, restore_plan):
    return (
        _checkpoint_perturb(model, direction, scale, config, restore_plan)
        if restore_plan is not None
        else perturb(model, direction, scale)
    )


@contextmanager
def perturb_from_checkpoint(model, direction, scale, config):
    """Public exact-restoration step helper for local pinned safetensors checkpoints."""
    restore_plan = _checkpoint_restore_plan(model, config)
    if restore_plan is None:
        raise ValueError(
            "Checkpoint-backed perturbation requires model_path and checkpoint_manifest"
        )
    with _checkpoint_perturb(model, direction, scale, config, restore_plan):
        yield


def _chunked_dot(left, right, chunk_size=1_048_576):
    if left.shape != right.shape:
        raise ValueError("Parameter shape mismatch")
    left, right = left.reshape(-1), right.reshape(-1)
    return sum(
        float(
            (
                left[offset : offset + chunk_size].double()
                * right[offset : offset + chunk_size].double()
            ).sum()
        )
        for offset in range(0, left.numel(), chunk_size)
    )


def _projected_gradient(model, weighted, direction, tolerance):
    """Compute <direction, gradient> while discarding each parameter gradient immediately."""
    import torch

    parameters = dict(model.named_parameters())
    if set(parameters) != set(direction) or any(
        tuple(direction[name].shape) != tuple(parameter.shape)
        for name, parameter in parameters.items()
    ):
        raise ValueError("Direction parameter identity/shape mismatch")
    model.eval().zero_grad(set_to_none=True)
    projection, hooks, error = 0.0, [], 0.0
    parameter_storage = {p.untyped_storage().data_ptr() for p in parameters.values()}

    def pack(tensor):
        if tensor.device.type == "cpu" or tensor.untyped_storage().data_ptr() in parameter_storage:
            return tensor
        return tensor.device, tensor.to("cpu")

    def unpack(saved):
        return saved[1].to(saved[0]) if isinstance(saved, tuple) else saved

    for name, parameter in parameters.items():

        def project(p, name=name):
            nonlocal projection
            value = p.grad.detach().float().cpu()
            if not torch.isfinite(value).all():
                raise FloatingPointError("Nonfinite gradient")
            projection += _chunked_dot(value, direction[name])
            p.grad = None

        hooks.append(parameter.register_post_accumulate_grad_hook(project))
    try:
        for row, weight in weighted:
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
        model.zero_grad(set_to_none=True)
    return projection, error


def _dot_gradient_file(direction, path):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(direction):
            raise ValueError("Parameter manifest mismatch")
        return sum(_chunked_dot(handle.get_tensor(name), direction[name]) for name in direction)


def evaluate(model, rows):
    import torch

    with torch.no_grad():
        return np.array(
            [float(token_logps(model, r["prompt_ids"], r["response_ids"]).sum()) for r in rows]
        )


def branch(shared_path, independent_path, evaluation_bank, output, step, readouts=()):
    from safetensors.torch import load_file

    if not np.isfinite(step) or step <= 0:
        raise ValueError("Freeze a positive step on Dev")
    sm, shared_gradient = _gradient_metadata(shared_path)
    im, independent_gradient = _gradient_metadata(independent_path)
    manifest, groups = read_bank(evaluation_bank)
    if (
        sm["arm"] != "shared"
        or im["arm"] != "independent"
        or sm["bank_sha256"] != im["bank_sha256"]
    ):
        raise ValueError("Need S/I from identical candidate bank")
    if sm["config"] != im["config"] or sm["config"] != manifest["config"]:
        raise ValueError("Checkpoint contract mismatch")
    _, treatment = read_bank(sm["bank"])
    if sha256(Path(sm["bank"]) / "rows.jsonl") != sm["bank_sha256"]:
        raise ValueError("Treatment bank changed")
    assert_independent(treatment, groups)
    treatment_rows = [r for group in treatment.values() for r in group]
    if any(r["split"] != "C" for r in treatment_rows):
        raise ValueError("Gradient directions require the C split")
    rows = [r for group in groups.values() for r in group]
    if any(r["split"] != "D" for r in rows):
        raise ValueError("Local consequences require independent D split")
    model, _ = load_model(manifest["config"])
    restore_plan = _checkpoint_restore_plan(model, manifest["config"])
    model._feedback_offload_activations = manifest["config"].get("offload_gradients", False)
    base = evaluate(model, rows)
    sequence_tolerance = manifest["config"].get("sequence_logp_tolerance")
    if sequence_tolerance is None:
        sequence_tolerance = manifest["config"]["token_logp_tolerance"] * max(
            len(r["response_ids"]) for r in rows
        )
    if (
        max(abs(base - np.array([sum(r["sampling_token_logp"]) for r in rows])))
        > sequence_tolerance
    ):
        raise ValueError("D bank off checkpoint")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    results, values = {}, {}
    prompt_ids = sorted({r["prompt_id"] for r in rows})
    for arm, direction_path in (
        ("shared", shared_gradient),
        ("independent", independent_gradient),
    ):
        direction = load_file(str(direction_path))
        predictions = []
        for readout in readouts:
            rm, readout_gradient = _gradient_metadata(readout)
            if rm["config"] != manifest["config"] or rm.get("kind") != "Dev_frozen_function":
                raise ValueError("Primary readout must be Dev-frozen at the same contract")
            definitions = rm["definitions"]
            baseline = sum(
                r["coefficient"] * lp
                for r, lp in zip(definitions, evaluate(model, definitions), strict=True)
            )
            slope = _dot_gradient_file(direction, readout_gradient)
            for eta in (step, step / 2):
                with _perturb(model, direction, eta, manifest["config"], restore_plan):
                    new = sum(
                        r["coefficient"] * lp
                        for r, lp in zip(definitions, evaluate(model, definitions), strict=True)
                    )
                predictions.append(
                    {
                        "readout": str(readout),
                        "step": eta,
                        "slope": slope,
                        "predicted_change": eta * slope,
                        "actual_change": new - baseline,
                        "residual": new - baseline - eta * slope,
                    }
                )
        targets = {}
        for target in ("average", "suite"):
            weights = []
            for prompt in prompt_ids:
                pr = [r for r in rows if r["prompt_id"] == prompt]
                for row in pr:
                    reward = (
                        np.mean(row["test_verdicts"])
                        if target == "average"
                        else np.prod(row["test_verdicts"])
                    )
                    weights.append((row, float(reward) / (len(pr) * len(prompt_ids))))
            slope, _ = _projected_gradient(
                model, weights, direction, manifest["config"]["token_logp_tolerance"]
            )
            targets[target] = {
                "aggregate_slope": slope,
                "uncertainty": "D_estimated_C_frozen_no_joint_CI",
            }
        for eta in (step, step / 2):
            with _perturb(model, direction, eta, manifest["config"], restore_plan):
                new = evaluate(model, rows)
            values[f"{arm}:{eta}"] = new
            summaries = {}
            for prompt in prompt_ids:
                idx = [i for i, r in enumerate(rows) if r["prompt_id"] == prompt]
                summaries[prompt] = {
                    target: importance_reward(
                        [
                            float(
                                np.mean(rows[i]["test_verdicts"])
                                if target == "average"
                                else np.prod(rows[i]["test_verdicts"])
                            )
                            for i in idx
                        ],
                        base[idx],
                        new[idx],
                    )
                    for target in ("average", "suite")
                }
            results[f"{arm}:{eta}"] = summaries
        write_json(
            out / f"{arm}_predictions.json", {"fixed_function": predictions, "D_targets": targets}
        )
        del direction
    write_jsonl(
        out / "evaluation.jsonl",
        [
            {
                "trajectory_id": r["trajectory_id"],
                "prompt_id": r["prompt_id"],
                "old_logp": float(base[i]),
                "new_logps": {k: float(v[i]) for k, v in values.items()},
            }
            for i, r in enumerate(rows)
        ],
    )
    report = {
        "status": "local_function_and_importance_diagnostic_not_pilot_GO",
        "step": step,
        "independent_generation_after_step": False,
        "per_prompt": results,
        "shared_gradient": sm,
        "independent_gradient": im,
        "evaluation_sha256": sha256(Path(evaluation_bank) / "rows.jsonl"),
    }
    write_json(out / "receipt.json", report)
    return report

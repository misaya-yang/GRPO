"""Single-run Stage-E warmup and online training; never launches a seed matrix."""

import json
import time
from pathlib import Path

import numpy as np

from dependent_rollouts.artifacts import sha256, write_json, write_jsonl
from dependent_rollouts.tasks import read_tasks
from reward_coupling.bank import digest

from .model import aggregate_gradient, load_model, parameter_identity, trainable
from .weights import full_stratified_weights, iid_weights


def _stage_d_decision(config, decision_path):
    path = Path(decision_path)
    expected = config.get("stage_d_decision_sha256")
    if not expected or sha256(path) != expected:
        raise ValueError("Stage-D decision hash mismatch")
    decision = json.loads(path.read_text())
    if decision.get("stage") != "D" or decision.get("decision") != "SUPPORT_CONTINUATION":
        raise ValueError("Stage D does not support continuation")
    evidence = decision.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Stage-D continuation requires evidence artifacts")
    for item in evidence:
        evidence_path = Path(item.get("path", ""))
        if not evidence_path.is_file() or sha256(evidence_path) != item.get("sha256"):
            raise ValueError("Stage-D evidence changed or is missing")
    return decision


def _positive_integer(config, key):
    value = config.get(key)
    if type(value) is not int or value < 1:
        raise ValueError(f"Invalid {key}")
    return value


def _validate(config, mode):
    for key in (
        "K",
        "B",
        "m",
        "updates",
        "prompts_per_update",
        "macros_per_prompt",
        "evaluation_macros_per_prompt",
        "stage_max_seconds",
    ):
        _positive_integer(config, key)
    if config["B"] < config["K"]:
        raise ValueError("Stratified online runs require B >= K")
    if mode == "common_warmup":
        arm = "iid_all"
    else:
        arm = config.get("arm")
        if arm not in ("iid_all", "stratified_full"):
            raise ValueError("Unknown online arm")
        adapter = Path(config.get("adapter_path", ""))
        if not adapter.is_file() or sha256(adapter) != config.get("adapter_sha256"):
            raise ValueError("Online arms require the identical frozen common adapter")
    if config.get("on_policy") is not True or config.get("epochs_per_bank") != 1:
        raise ValueError("Online training requires one on-policy epoch per fresh bank")
    if config.get("clip_policy_ratio") is not False:
        raise ValueError("Off-policy ratio clipping is outside this online contract")
    if config.get("loss_reduction") != "sum_tokens_then_mean_responses":
        raise ValueError("Unexpected loss reduction")
    if config.get("optimizer") not in ("sgd", "adamw"):
        raise ValueError("Optimizer must be sgd or adamw")
    learning_rate = config.get("learning_rate")
    if (
        not isinstance(learning_rate, (int, float))
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("Positive finite learning_rate required")
    for key in ("seed", "evaluation_seed"):
        if type(config.get(key)) is not int:
            raise ValueError(f"Invalid {key}")
    if config["seed"] == config["evaluation_seed"]:
        raise ValueError("Training and evaluation RNG identities must differ")
    if not config.get("shared_initialization_id"):
        raise ValueError("Missing shared initialization identity")
    return arm


def _task_sets(config, tasks_path):
    if sha256(tasks_path) != config.get("tasks_sha256"):
        raise ValueError("Tasks file changed")
    rows = read_tasks(tasks_path)
    lookup = {row["prompt_id"]: row for row in rows}
    train_ids = config.get("train_prompt_ids", [])
    evaluation_ids = config.get("evaluation_prompt_ids", [])
    if (
        not train_ids
        or not evaluation_ids
        or len(set(train_ids)) != len(train_ids)
        or len(set(evaluation_ids)) != len(evaluation_ids)
        or set(train_ids) & set(evaluation_ids)
        or not set(train_ids + evaluation_ids) <= lookup.keys()
    ):
        raise ValueError("Need disjoint frozen training and evaluation prompts")
    if len(train_ids) < config["prompts_per_update"]:
        raise ValueError("Insufficient training prompts per update")
    return [lookup[key] for key in train_ids], [lookup[key] for key in evaluation_ids]


def _binary_reward(row):
    reward = row.get("reward", row.get("suite_result"))
    if type(reward) not in (int, bool) or int(reward) not in (0, 1):
        raise ValueError("Online training requires complete binary rewards")
    return int(reward)


def _macro_weights(rows, arm, config):
    if arm == "iid_all":
        if len(rows) != config["K"]:
            raise ValueError("IID-all split-budget macro must contain exactly fixed K responses")
        rewards = np.array([_binary_reward(row) for row in rows]).reshape(config["K"], 1)
        return iid_weights(rewards, config["K"], config["epsilon"]).reshape(-1)
    expected = config["B"] * config["m"]
    if len(rows) != expected:
        raise ValueError("Strat-full macro must contain B*m responses")
    positions = [(row.get("block"), row.get("stratum")) for row in rows]
    required = [(block, stratum) for block in range(config["B"]) for stratum in range(config["m"])]
    if sorted(positions) != required:
        raise ValueError("Strat-full macro has incomplete block/stratum identity")
    order = np.argsort([block * config["m"] + stratum for block, stratum in positions])
    matrix = np.empty((config["B"], config["m"]), dtype=int)
    for row in rows:
        matrix[row["block"], row["stratum"]] = _binary_reward(row)
    ordered_weights = full_stratified_weights(matrix, config["K"], config["epsilon"]).reshape(-1)
    inverse = np.empty(expected, dtype=int)
    inverse[order] = np.arange(expected)
    return ordered_weights[inverse]


def _optimizer(model, config):
    import torch

    parameters = list(trainable(model).values())
    if not parameters:
        raise ValueError("No trainable LoRA parameters")
    if config["optimizer"] == "sgd":
        return torch.optim.SGD(parameters, lr=config["learning_rate"])
    return torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0)


def _save_adapter(model, path):
    from safetensors.torch import save_file

    values = {
        name: parameter.detach().cpu().contiguous() for name, parameter in trainable(model).items()
    }
    save_file(values, str(path))
    return sha256(path)


def _collect(collector, model, tokenizer, config, task, macro, arm, deadline):
    rows, costs = collector(model, tokenizer, config, task, macro, arm, deadline)
    if not isinstance(rows, list) or not rows or not isinstance(costs, dict):
        raise ValueError("collect_macro returned an invalid result")
    return rows, costs


def _run(config_path, tasks_path, decision_path, output, mode, collector=None):
    import torch

    config = json.loads(Path(config_path).read_text())
    arm = _validate(config, mode)
    decision = _stage_d_decision(config, decision_path)
    training_tasks, evaluation_tasks = _task_sets(config, tasks_path)
    if collector is None:
        from .pipeline import collect_macro

        collector = collect_macro
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load_model(config)
    initial_identity = parameter_identity(model)
    optimizer = _optimizer(model, config)
    optimizer_contract = {
        "name": config["optimizer"],
        "learning_rate": config["learning_rate"],
        "weight_decay": 0,
        "epochs_per_bank": 1,
    }
    write_json(
        out / "manifest.json",
        {
            "mode": mode,
            "arm": arm,
            "config": config,
            "config_sha256": sha256(config_path),
            "tasks_sha256": sha256(tasks_path),
            "stage_d_decision_sha256": sha256(decision_path),
            "stage_d_evidence": decision["evidence"],
            "initial_parameters": initial_identity,
            "optimizer": optimizer_contract,
            "aggregation": "mean_over_actual_responses_N_fixed_K",
            "on_policy": True,
        },
    )
    rng = np.random.default_rng(config["seed"])
    started = time.monotonic()
    deadline = started + config["stage_max_seconds"]
    seen_trajectories, seen_rng, update_summaries = set(), set(), []
    macro_number = 0
    for update in range(config["updates"]):
        chosen = rng.choice(len(training_tasks), size=config["prompts_per_update"], replace=False)
        update_rows, update_weights, costs = [], [], []
        for task_index in chosen:
            task = training_tasks[int(task_index)]
            for _ in range(config["macros_per_prompt"]):
                rows, macro_costs = _collect(
                    collector, model, tokenizer, config, task, macro_number, arm, deadline
                )
                weights = _macro_weights(rows, arm, config)
                row_hashes = [digest(row) for row in rows]
                bank_hash = digest(row_hashes)
                for row, weight, row_hash in zip(rows, weights, row_hashes, strict=True):
                    trajectory = row.get("trajectory_id")
                    rng_seed = row.get("rng_seed")
                    if not trajectory or trajectory in seen_trajectories:
                        raise ValueError("Online updates require fresh trajectory identities")
                    if rng_seed is None or rng_seed in seen_rng:
                        raise ValueError("Online updates require fresh RNG streams")
                    seen_trajectories.add(trajectory)
                    seen_rng.add(rng_seed)
                    audit = dict(row)
                    audit.update(
                        {
                            "update": update,
                            "macro": macro_number,
                            "arm": arm,
                            "weight": float(weight),
                            "row_hash": row_hash,
                            "fresh_bank_hash": bank_hash,
                        }
                    )
                    update_rows.append(audit)
                    update_weights.append(float(weight))
                costs.append(macro_costs)
                macro_number += 1
        gradients, score_errors = aggregate_gradient(
            model,
            update_rows,
            np.asarray(update_weights),
            config["token_logp_tolerance"],
            config["sequence_logp_tolerance"],
        )
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in trainable(model).items():
            if name not in gradients or gradients[name].shape != parameter.shape:
                raise ValueError("Gradient parameter identity mismatch")
            parameter.grad = -gradients[name].to(parameter.device, parameter.dtype)
        if time.monotonic() >= deadline:
            raise TimeoutError("Online budget exhausted before update")
        optimizer.step()
        write_jsonl(out / f"update-{update:04d}.jsonl", update_rows)
        summary = {
            "update": update,
            "responses": len(update_rows),
            "fresh_bank_hashes": sorted({row["fresh_bank_hash"] for row in update_rows}),
            "score_errors": score_errors,
            "costs": costs,
            "parameter_identity_after": parameter_identity(model),
        }
        write_json(out / f"update-{update:04d}.json", summary)
        update_summaries.append(summary)

    adapter_path = out / "adapter.safetensors"
    adapter_sha256 = _save_adapter(model, adapter_path)
    with (out / "optimizer.pt").open("xb") as handle:
        torch.save({"state_dict": optimizer.state_dict(), "config": optimizer_contract}, handle)
    evaluation_config = {**config, "seed": config["evaluation_seed"], "stage": "online_evaluation"}
    evaluation_rows, evaluation_costs = [], []
    evaluation_macro = 0
    for task in evaluation_tasks:
        for _ in range(config["evaluation_macros_per_prompt"]):
            rows, costs = _collect(
                collector,
                model,
                tokenizer,
                evaluation_config,
                task,
                evaluation_macro,
                "iid_all",
                deadline,
            )
            if len(rows) != config["K"]:
                raise ValueError("Independent evaluation macro must contain fixed K responses")
            bank_hash = digest([digest(row) for row in rows])
            for row in rows:
                audit = dict(row)
                audit.update(
                    {
                        "evaluation_macro": evaluation_macro,
                        "arm": "iid_all",
                        "row_hash": digest(row),
                        "fresh_bank_hash": bank_hash,
                    }
                )
                evaluation_rows.append(audit)
            evaluation_costs.append(costs)
            evaluation_macro += 1
    write_jsonl(out / "evaluation.jsonl", evaluation_rows)
    evaluation_summary = {
        prompt_id: {
            "responses": sum(row["prompt_id"] == prompt_id for row in evaluation_rows),
            "mean_reward": float(
                np.mean(
                    [
                        _binary_reward(row)
                        for row in evaluation_rows
                        if row["prompt_id"] == prompt_id
                    ]
                )
            ),
        }
        for prompt_id in config["evaluation_prompt_ids"]
    }
    result = {
        "status": "common_warmup_complete" if mode == "common_warmup" else "online_run_complete",
        "mode": mode,
        "arm": arm,
        "seed": config["seed"],
        "updates": len(update_summaries),
        "adapter": str(adapter_path),
        "adapter_sha256": adapter_sha256,
        "optimizer_state_sha256": sha256(out / "optimizer.pt"),
        "initial_parameter_space_hash": initial_identity["parameter_space_hash"],
        "initial_adapter_values_hash": initial_identity["adapter_values_hash"],
        "final_parameters": parameter_identity(model),
        "evaluation": evaluation_summary,
        "evaluation_costs": evaluation_costs,
        "evaluation_sha256": sha256(out / "evaluation.jsonl"),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(out / "receipt.json", result)
    return result


def run_common_warmup(config_path, tasks_path, decision_path, output, collector=None):
    """Run one evidence-gated common IID warmup and independent evaluation."""
    return _run(config_path, tasks_path, decision_path, output, "common_warmup", collector)


def run_online(config_path, tasks_path, decision_path, output, collector=None):
    """Run one arm and one seed from a frozen common adapter, then evaluate."""
    return _run(config_path, tasks_path, decision_path, output, "online", collector)

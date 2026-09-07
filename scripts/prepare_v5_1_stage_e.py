#!/usr/bin/env python
"""Compile Stage-E configs from real Stage-D evidence; never run a training matrix."""

import argparse
import json
from pathlib import Path

from dependent_rollouts.artifacts import sha256, write_json, write_jsonl
from dependent_rollouts.tasks import read_tasks
from reward_coupling.bank import digest
from stratified_grpo.online import _stage_d_decision


def prepare(screen, data, output, updates, warmup_updates, learning_rate, seeds, common=None):
    screen, data, output = Path(screen).resolve(), Path(data), Path(output)
    manifest = json.loads((screen / "manifest.json").read_text())
    base = manifest["config"]
    decision_path = screen / "decision.json"
    identity = manifest["parameter_identity"]
    train = read_tasks(data / "train.jsonl")
    evaluation = read_tasks(data / "evaluation.jsonl")
    if (
        type(updates) is not int
        or updates < 1
        or warmup_updates < 1
        or not learning_rate > 0
        or len(set(seeds)) != len(seeds)
        or not seeds
    ):
        raise ValueError("Freeze positive update counts/lr and distinct seeds")
    if set(r["prompt_id"] for r in train + evaluation) & set(base["prompt_ids"]):
        raise ValueError("Stage-E data overlaps Stage-D confirmation")
    config = {
        **base,
        "stage": "online",
        "updates": updates,
        "prompts_per_update": 1,
        "macros_per_prompt": 1,
        "evaluation_macros_per_prompt": 1,
        "on_policy": True,
        "epochs_per_bank": 1,
        "clip_policy_ratio": False,
        "loss_reduction": "sum_tokens_then_mean_responses",
        "optimizer": "sgd",
        "learning_rate": learning_rate,
        "stage_d_decision": str(decision_path),
        "stage_d_decision_sha256": sha256(decision_path),
        "stage_d_method_hash": manifest["method_hash"],
        "stage_d_parameter_space_hash": identity["parameter_space_hash"],
        "shared_initialization_id": digest(
            [identity, sha256(decision_path), warmup_updates, learning_rate]
        ),
        "train_prompt_ids": [r["prompt_id"] for r in train],
        "evaluation_prompt_ids": [r["prompt_id"] for r in evaluation],
    }
    _stage_d_decision(config, decision_path)
    output.mkdir(parents=True, exist_ok=False)
    task_path = output / "tasks.jsonl"
    write_jsonl(task_path, train + evaluation)
    config["tasks_sha256"] = sha256(task_path)
    warm = {
        **config,
        "updates": warmup_updates,
        "seed": 601001,
        "evaluation_seed": 602001,
        "iid_budget_mode": "same_prompt",
    }
    write_json(output / "warmup.json", warm)
    files = ["warmup.json"]
    if common is not None:
        common = Path(common).resolve()
        receipt = json.loads((common / "receipt.json").read_text())
        if (
            receipt["status"] != "common_warmup_complete"
            or receipt["shared_initialization_id"] != config["shared_initialization_id"]
        ):
            raise ValueError("Common checkpoint is not from this frozen warmup")
        adapter = common / "adapter.safetensors"
        if sha256(adapter) != receipt["adapter_sha256"]:
            raise ValueError("Common adapter changed")
        config.update(
            adapter_path=str(adapter),
            adapter_sha256=sha256(adapter),
            common_checkpoint_receipt=str(common / "receipt.json"),
            common_checkpoint_receipt_sha256=sha256(common / "receipt.json"),
        )
        for seed in seeds:
            for arm, mode in (
                ("iid_all", "same_prompt"),
                ("stratified_full", "same_prompt"),
                ("iid_all", "more_prompts"),
            ):
                item = {
                    **config,
                    "seed": seed,
                    "evaluation_seed": seed + 1000000,
                    "arm": arm,
                    "iid_budget_mode": mode,
                }
                name = f"{arm}_{mode}_seed{seed}.json"
                write_json(output / name, item)
                files.append(name)
        # New checkpoint requires its own Dev numerical/method freeze before same-point audit.
        for name, split in (("mid_dev", "calibration"), ("mid_screen", "confirmation")):
            original = Path(
                "configs/v5_1/dev.json" if name == "mid_dev" else "configs/v5_1/screen.json"
            )
            template = json.loads(original.read_text())
            mid = {
                **base,
                **{
                    k: template[k]
                    for k in (
                        "stage",
                        "macro_repeats",
                        "prompt_ids",
                        "cross_diagnostic_macros",
                        "seed",
                    )
                },
            }
            mid.pop("dev_freeze", None)
            mid.pop("dev_freeze_sha256", None)
            mid.update(
                adapter_path=str(adapter),
                adapter_sha256=sha256(adapter),
                seed=mid["seed"] + 2000000,
            )
            mid["tasks_sha256"] = sha256(data / (split + ".jsonl"))
            write_json(output / (name + ".json"), mid)
            files.append(name + ".json")
    write_json(
        output / "manifest.json",
        {
            "status": "CONFIGS_ONLY_NOT_RUN",
            "source_screen": str(screen),
            "updates": updates,
            "warmup_updates": warmup_updates,
            "seeds": seeds,
            "files": {n: sha256(output / n) for n in files},
            "note": "Use cumulative budget supervisor; do not run full matrix before a costed plan",
        },
    )
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen", required=True)
    p.add_argument("--data", default="data/v5_1/svamp")
    p.add_argument("--output", required=True)
    p.add_argument("--updates", type=int, required=True)
    p.add_argument("--warmup-updates", type=int, required=True)
    p.add_argument("--learning-rate", type=float, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--common")
    args = p.parse_args()
    print(
        json.dumps(
            prepare(
                args.screen,
                args.data,
                args.output,
                args.updates,
                args.warmup_updates,
                args.learning_rate,
                args.seeds,
                args.common,
            )
        )
    )


if __name__ == "__main__":
    main()

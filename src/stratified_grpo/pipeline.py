"""One-process paired-bank collection and fixed-point LoRA gradient audit."""

import json
import time
from pathlib import Path

import numpy as np

from dependent_rollouts.artifacts import provenance, sha256, write_json
from dependent_rollouts.tasks import verify
from reward_coupling.bank import digest
from reward_coupling.sample import encode_prompt

from .contract import method_contract, require_dev_freeze, selected_tasks, validate
from .model import aggregate_gradient, load_model, parameter_identity
from .sampling import generate, stream_seed
from .statistics import compare, trace_variance
from .weights import cross_weights, full_stratified_weights, iid_weights


def collect_macro(model, tokenizer, config, task, macro, arm, deadline):
    if arm not in ("iid_all", "stratified_full"):
        raise ValueError("Unknown bank arm")
    prompt = encode_prompt(tokenizer, task["prompt"])
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    eos = eos or [tokenizer.eos_token_id]
    if None in eos:
        raise ValueError("Missing EOS token")
    rows = []
    costs = {"generation_seconds": 0.0, "scoring_seconds": 0.0}
    for block in range(config["B"]):
        for stratum in range(config["m"]):
            seed = stream_seed(
                config["seed"], config["stage"], task["prompt_id"], macro, arm, block, stratum
            )
            started = time.monotonic()
            row = generate(
                model,
                prompt,
                eos,
                seed,
                config["max_new_tokens"],
                stratum if arm == "stratified_full" else None,
                config["m"],
                config["onset"],
                config["use_cache"],
                deadline,
            )
            costs["generation_seconds"] += time.monotonic() - started
            row.update(
                {
                    "prompt_id": task["prompt_id"],
                    "prompt_ids": prompt,
                    "macro": macro,
                    "arm": arm,
                    "block": block,
                    "slot": stratum,
                    "response_id": f"{task['prompt_id']}:{macro}:{arm}:{block}:{stratum}",
                }
            )
            started = time.monotonic()
            row["text"] = tokenizer.decode(row["response_ids"], skip_special_tokens=True)
            row.update(verify(row["text"], task, row["finish_reason"]))
            costs["scoring_seconds"] += time.monotonic() - started
            if row["cdf_tv_sum"] > config["cdf_tv_tolerance"]:
                raise FloatingPointError("CDF accumulated discrepancy exceeds frozen tolerance")
            rows.append(row)
    return rows, costs


def compiled_weights(rows, config, arm):
    rewards = np.array([r["reward"] for r in rows]).reshape(config["B"], config["m"])
    if arm == "iid_all":
        return iid_weights(rewards, config["K"], config["epsilon"]).reshape(-1)
    if arm == "stratified_full":
        return full_stratified_weights(rewards, config["K"], config["epsilon"]).reshape(-1)
    if arm == "stratified_cross":
        return cross_weights(rewards, config["K"], config["epsilon"]).reshape(-1)
    raise ValueError("Unknown estimator")


def run(config_path, tasks_path, output):
    import torch
    from safetensors.torch import save_file

    config = validate(json.loads(Path(config_path).read_text()))
    tasks = selected_tasks(config, tasks_path)
    require_dev_freeze(config)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "STARTED",
        "config": config,
        "config_hash": digest(config),
        "tasks_hash": sha256(tasks_path),
        "method_hash": digest(method_contract(config)),
        "provenance": provenance(config),
        "independence": "private fresh RNG for every actual response",
        "evidence_level": "pretrained_fixed_point_LoRA_only",
    }
    write_json(out / "manifest.json", manifest)
    started = time.monotonic()
    deadline = started + config["stage_max_seconds"]
    try:
        model, tokenizer = load_model(config)
        load_seconds = time.monotonic() - started
        identity = parameter_identity(model)
        manifest["parameter_identity"] = identity
        manifest["template_hash"] = digest(tokenizer.chat_template)
        write_json(out / "manifest.json", manifest)
        entries = []
        with (
            (out / "rows.jsonl").open("x") as bank_file,
            (out / "progress.jsonl").open("x") as progress,
        ):
            for task_no, task in enumerate(tasks):
                for macro in range(config["macro_repeats"]):
                    # Balance order to reduce systematic warm-cache/thermal cost confounding.
                    arms = config["arms"] if (task_no + macro) % 2 == 0 else config["arms"][::-1]
                    for arm in arms:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Stage deadline reached")
                        rows, costs = collect_macro(
                            model, tokenizer, config, task, macro, arm, deadline
                        )
                        for row in rows:
                            bank_file.write(json.dumps(row, allow_nan=False) + "\n")
                        bank_file.flush()
                        at = time.monotonic()
                        weights = compiled_weights(rows, config, arm)
                        costs["weight_seconds"] = time.monotonic() - at
                        at = time.monotonic()
                        gradients, errors = aggregate_gradient(
                            model,
                            rows,
                            weights,
                            config["token_logp_tolerance"],
                            config["sequence_logp_tolerance"],
                        )
                        costs["replay_seconds"] = time.monotonic() - at
                        filename = f"gradient-{task_no:03d}-{macro:03d}-{arm}.safetensors"
                        save_file(gradients, str(out / filename))
                        # A deterministic linear functional in the declared LoRA tangent space.
                        projected = []
                        for projection in range(config.get("functional_projections", 4)):
                            rng = np.random.default_rng(config["projection_seed"] + projection)
                            value = 0.0
                            for g in gradients.values():
                                signs = (
                                    rng.integers(0, 2, g.numel(), dtype=np.int8).astype(np.float64)
                                    * 2
                                    - 1
                                )
                                value += float(signs @ g.reshape(-1).double().numpy()) / np.sqrt(
                                    identity["trainable_count"]
                                )
                            projected.append(value)
                        record = {
                            "prompt_id": task["prompt_id"],
                            "macro": macro,
                            "arm": arm,
                            "gradient_file": filename,
                            "gradient_sha256": sha256(out / filename),
                            "parameter_space_hash": identity["parameter_space_hash"],
                            "adapter_values_hash": identity["adapter_values_hash"],
                            "rows_hash": digest(rows),
                            "weights": weights.tolist(),
                            "costs": costs,
                            "total_macro_seconds": sum(costs.values()),
                            "reward_mean": float(np.mean([r["reward"] for r in rows])),
                            "all_equal_rewards": len({r["reward"] for r in rows}) == 1,
                            "cdf_tv_sum_max": max(r["cdf_tv_sum"] for r in rows),
                            "unreleased_count": sum(not r["release_observed"] for r in rows),
                            "functional_projections": projected,
                            **errors,
                        }
                        entries.append(record)
                        progress.write(json.dumps(record, allow_nan=False) + "\n")
                        progress.flush()
                        print(
                            json.dumps(
                                {
                                    "completed_macros": len(entries),
                                    "total_macros": len(tasks) * config["macro_repeats"] * 2,
                                    "prompt": task["prompt_id"],
                                    "arm": arm,
                                    "reward_mean": record["reward_mean"],
                                }
                            ),
                            flush=True,
                        )
                        del gradients
                        if (
                            config.get("cross_diagnostic_macros", 0) > macro
                            and task_no == 0
                            and arm == "stratified_full"
                        ):
                            at = time.monotonic()
                            cross, error = aggregate_gradient(
                                model,
                                rows,
                                compiled_weights(rows, config, "stratified_cross"),
                                config["token_logp_tolerance"],
                                config["sequence_logp_tolerance"],
                            )
                            name = f"diagnostic-cross-{macro:03d}.safetensors"
                            save_file(cross, str(out / name))
                            write_json(
                                out / (name + ".json"),
                                {
                                    "sha256": sha256(out / name),
                                    "rows_hash": digest(rows),
                                    "audit_only_seconds": time.monotonic() - at,
                                    **error,
                                },
                            )
                            del cross
        if parameter_identity(model) != identity:
            raise ValueError("Fixed-point model parameters changed")
        write_json(
            out / "receipt.json",
            {
                "status": "COMPLETE",
                "macros": len(entries),
                "rows": len(entries) * config["B"] * config["m"],
                "model_load_seconds": load_seconds,
                "elapsed_seconds": time.monotonic() - started,
                "rows_sha256": sha256(out / "rows.jsonl"),
                "progress_sha256": sha256(out / "progress.jsonl"),
                "manifest_sha256": sha256(out / "manifest.json"),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else None,
            },
        )
        return analyze(out)
    except BaseException as error:
        write_json(
            out / "failure.json",
            {
                "status": "FAILED_OR_INTERRUPTED",
                "error": repr(error),
                "elapsed_seconds": time.monotonic() - started,
                "partial_banks_are_not_complete": True,
            },
        )
        raise


def analyze(directory):
    from safetensors.torch import load_file

    out = Path(directory)
    receipt = json.loads((out / "receipt.json").read_text())
    for name in ("rows", "progress", "manifest"):
        filename = name + (".json" if name == "manifest" else ".jsonl")
        if sha256(out / filename) != receipt[name + "_sha256"]:
            raise ValueError("Sealed experiment changed")
    if receipt["status"] != "COMPLETE":
        raise ValueError("Incomplete experiment")
    config = json.loads((out / "manifest.json").read_text())["config"]
    entries = [json.loads(s) for s in (out / "progress.jsonl").read_text().splitlines()]
    records = []
    for prompt in config["prompt_ids"]:
        record = {"prompt_id": prompt}
        means = {}
        for arm, label in (("iid_all", "iid"), ("stratified_full", "stratified")):
            chosen = sorted(
                [e for e in entries if e["prompt_id"] == prompt and e["arm"] == arm],
                key=lambda e: e["macro"],
            )
            if [e["macro"] for e in chosen] != list(range(config["macro_repeats"])):
                raise ValueError("Incomplete or duplicated macro identities")
            # Accumulate Gram matrices parameter by parameter, avoiding full per-response scores.
            states = []
            for e in chosen:
                if sha256(out / e["gradient_file"]) != e["gradient_sha256"]:
                    raise ValueError("Gradient changed")
                states.append(load_file(str(out / e["gradient_file"])))
            gram = np.zeros((len(states), len(states)))
            mean_parts = []
            for name in states[0]:
                values = np.stack([s[name].reshape(-1).double().numpy() for s in states])
                mean = values.mean(0)
                mean_parts.append(mean)
                values -= mean
                gram += values @ values.T
            means[label] = mean_parts
            record[label + "_gram"] = gram.tolist()
            record[label + "_costs"] = [e["total_macro_seconds"] for e in chosen]
            del states
        record["mean_gradient_difference_l2"] = float(
            np.sqrt(
                sum(
                    np.square(s - i).sum()
                    for s, i in zip(means["stratified"], means["iid"], strict=True)
                )
            )
        )
        record["estimated_sampling_noise_rms"] = float(
            np.sqrt(
                sum(
                    trace_variance(record[a + "_gram"]) / config["macro_repeats"]
                    for a in ("iid", "stratified")
                )
            )
        )
        records.append(record)
    result = compare(records, config["practical_threshold"], seed=config["analysis_seed"])
    if config["stage"] == "dev":
        result["decision"] = "DEV_DIAGNOSTIC_ONLY"
    result.update(
        {
            "stage": config["stage"],
            "metric": config["metric"],
            "K": config["K"],
            "N": config["B"] * config["m"],
            "all_equal_macros_retained": sum(e["all_equal_rewards"] for e in entries),
            "mean_reward": float(np.mean([e["reward_mean"] for e in entries])),
            "max_token_error": max(e["max_token_error"] for e in entries),
            "max_sequence_error": max(e["max_sequence_error"] for e in entries),
            "max_lora_B_gradient_l2": max(e["l2_B"] for e in entries),
            "cost_scope": "generation + deterministic reward + weight compilation + score-checked replay; model loading and cross audit separate",
            "receipt_sha256": sha256(out / "receipt.json"),
        }
    )
    write_json(out / "macro_statistics.json", records)
    write_json(out / "analysis.json", result)
    (out / "REPORT.md").write_text(
        "# v5.1 固定点实验报告\n\n"
        + f"状态：{result['decision']}。K={config['K']}，N={config['B'] * config['m']}；{len(records)} 题，每臂每题 {config['macro_repeats']} 个独立 macro-bank。\n\n"
        + f"方差比：{result.get('rho')}；时间归一化方差比：{result.get('rho_time')}。95% 层级 bootstrap 区间：{result.get('rho_time_interval')}。\n\n"
        + "范围：共同 checkpoint 的完整 LoRA 梯度 trace。不是全参数效果、任务收益或在线训练结论；全同奖励 bank 已保留。详细数值见 analysis.json，prompt/macro 统计见 macro_statistics.json。\n"
    )
    return result


def freeze_dev(directory, output):
    """Freeze passing Dev plumbing and measured resource forecast; no outcome tuning."""
    out = Path(directory)
    result = analyze(out)
    manifest = json.loads((out / "manifest.json").read_text())
    config = manifest["config"]
    if (
        config["stage"] != "dev"
        or not 0 < result["mean_reward"] < 1
        or result["max_lora_B_gradient_l2"] <= 0
    ):
        raise ValueError("Dev lacks reward variation or nonzero measured LoRA score signal")
    # Exact-match configuration prevents using prior FP32 tolerances for another path.
    entries = [json.loads(s) for s in (out / "progress.jsonl").read_text().splitlines()]
    if any(
        e["max_token_error"] > config["token_logp_tolerance"]
        or e["max_sequence_error"] > config["sequence_logp_tolerance"]
        for e in entries
    ):
        raise ValueError("Dev probability contract did not pass")
    receipt = json.loads((out / "receipt.json").read_text())
    estimate = float(np.mean([e["total_macro_seconds"] for e in entries]))
    freeze = {
        "status": "FROZEN_FOR_SCREENING",
        "method_hash": digest(method_contract(config)),
        "dev_prompt_ids": config["prompt_ids"],
        "dev_seed": config["seed"],
        "parameter_identity": manifest["parameter_identity"],
        "seconds_per_macro": estimate,
        "model_load_seconds": receipt["model_load_seconds"],
        "evidence": [
            {"path": str((out / n).resolve()), "sha256": sha256(out / n)}
            for n in ("receipt.json", "manifest.json", "analysis.json")
        ],
        "forecast_only": True,
        "selection": "one predeclared method; no benefit-based stratum search",
    }
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    write_json(path, freeze)
    return freeze

"""Ordinary ancestral IID generation, shared token scoring implementation."""

import json
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from dependent_rollouts.artifacts import provenance, sha256, write_json

from .bank import digest, read_tasks, seal
from .contract import check_task_hash, read_config


def encode_prompt(tokenizer, prompt):
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("Expected one prompt")
        encoded = encoded[0]
    if not encoded or any(type(token) is not int for token in encoded):
        raise ValueError("Chat template did not produce token IDs")
    return encoded


def load_model(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(config.get("cpu_threads", 4))

    source = config.get("model_path", config["model_id"])
    if config.get("model_path"):
        from dependent_rollouts.artifacts import sha256

        metadata = json.loads(Path(config["checkpoint_manifest"]).read_text())
        if sha256(config["checkpoint_manifest"]) != config["checkpoint_manifest_sha256"]:
            raise ValueError("Checkpoint manifest changed")
        for filename, expected in metadata["files"].items():
            path = Path(source) / filename
            if sha256(path) != expected:
                raise ValueError(f"Local checkpoint file changed: {filename}")
    model = AutoModelForCausalLM.from_pretrained(
        source,
        revision=config["model_revision"],
        torch_dtype=torch.float32,
        trust_remote_code=False,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        source, revision=config["tokenizer_revision"], trust_remote_code=False
    )
    model.to(config["device"]).eval().requires_grad_(True)
    if config.get("cpu_embeddings", False):
        embedding = model.get_input_embeddings().to("cpu")
        embedding.register_forward_hook(lambda module, args, output: output.to(config["device"]))
    model._feedback_offload_gradients = config.get("offload_gradients", False)
    model._feedback_sequence_tolerance = config.get("sequence_logp_tolerance")
    return model, tokenizer


def token_logps(model, prompt_ids, response_ids):
    import torch

    if not prompt_ids or not response_ids:
        raise ValueError("Nonempty prompt and response required")
    ids = torch.tensor([prompt_ids + response_ids], device=next(model.parameters()).device)
    logits = model(input_ids=ids, use_cache=False).logits[0, len(prompt_ids) - 1 : -1].double()
    targets = ids[0, len(prompt_ids) :, None].to(logits.device)
    return logits.log_softmax(-1).gather(1, targets).squeeze(1)


def generate(model, prompt_ids, eos_ids, seed, max_new_tokens, deadline=None, use_cache=False):
    import torch

    if not prompt_ids or not eos_ids or max_new_tokens < 1:
        raise ValueError("Invalid ancestral generation contract")
    model.eval()
    rng = torch.Generator(device="cpu").manual_seed(seed)
    response, scores = [], []
    cache = None
    ids = torch.tensor([prompt_ids], device=next(model.parameters()).device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Generation budget exhausted")
            output = model(
                input_ids=ids, past_key_values=cache, use_cache=use_cache, logits_to_keep=1
            )
            logits = output.logits[0, -1].double()
            cache = output.past_key_values if use_cache else None
            logp = logits.log_softmax(-1)
            probs = logp.exp().cpu()
            if not torch.isfinite(probs).all() or torch.any(probs <= 0):
                raise FloatingPointError("Numerical distribution lost full support")
            token = int(torch.multinomial(probs, 1, generator=rng).item())
            response.append(token)
            scores.append(float(logp[token]))
            if token in eos_ids:
                break
            ids = torch.tensor(
                [[token]] if use_cache else [prompt_ids + response],
                device=next(model.parameters()).device,
            )
        rescored = token_logps(model, prompt_ids, response).cpu().numpy()
    error = float(np.max(np.abs(rescored - scores)))
    return {
        "response_ids": response,
        "response_length": len(response),
        "sampling_token_logp": scores,
        "scoring_token_logp": rescored.tolist(),
        "old_logp": float(sum(scores)),
        "max_token_logp_error": error,
        "active_mask": [1] * len(response),
        "eos_index_or_null": len(response) - 1 if response[-1] in eos_ids else None,
        "truncated": response[-1] not in eos_ids,
        "actor_rng_stream": seed,
        "generation_cache": use_cache,
    }


def resume_candidates(config, split):
    if not config.get("resume_from"):
        return {}
    if split != "Dev":
        raise ValueError("Numeric-calibration resume is a Dev operation")
    parent = Path(config["resume_from"])
    required = {"manifest.json", "rows.jsonl"}
    if (parent / "score_point_failure.json").exists():
        required.add("score_point_failure.json")
    if set(config["resume_artifact_sha256"]) != required:
        raise ValueError("Freeze all partial-bank artifact hashes before resume")
    for name, expected in config["resume_artifact_sha256"].items():
        if sha256(parent / name) != expected:
            raise ValueError("Partial candidate artifact changed")
    previous = json.loads((parent / "manifest.json").read_text())
    allowed = {
        "token_logp_tolerance",
        "numeric_tolerance_basis",
        "resume_from",
        "resume_artifact_sha256",
        "stage_max_seconds",
        "dev_prompt_limit",
        "use_cache",
        "numeric_calibration_receipt",
        "numeric_calibration_receipt_sha256",
    }
    old = {k: v for k, v in previous["config"].items() if k not in allowed}
    new = {k: v for k, v in config.items() if k not in allowed}
    if old != new or previous["split"] != split:
        raise ValueError("Resume may not change actor/data/randomization contract")
    if config.get("dev_prompt_limit", 10**9) < previous["config"].get("dev_prompt_limit", 10**9):
        raise ValueError("Resume may extend the frozen Dev prefix, not shorten it")
    rows = [json.loads(s) for s in (parent / "rows.jsonl").read_text().splitlines()]
    failure = parent / "score_point_failure.json"
    if failure.exists():
        rows.append(json.loads(failure.read_text()))
    cache = {r["actor_rng_stream"]: r for r in rows}
    if len(cache) != len(rows):
        raise ValueError("Duplicate random stream in partial bank")
    for row in cache.values():
        row.setdefault("generation_cache", previous["config"].get("use_cache", False))
    return cache


def collect(config_path, tasks_path, output, split):
    config = read_config(config_path)
    check_task_hash(config, tasks_path)
    tasks = [t for t in read_tasks(tasks_path) if t["split"] == split]
    limit = config.get("prompt_limits", {}).get(split)
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise ValueError("Invalid predeclared prompt prefix")
        tasks = tasks[:limit]
    if split == "Dev" and config.get("dev_prompt_limit") is not None:
        if type(config["dev_prompt_limit"]) is not int or config["dev_prompt_limit"] < 1:
            raise ValueError("Invalid Dev prompt limit")
        tasks = tasks[: config["dev_prompt_limit"]]
    if not tasks:
        raise ValueError("Requested split is empty")
    groups_per_prompt = config.get("groups_per_prompt_by_split", {}).get(
        split, config["groups_per_prompt"]
    )
    if type(groups_per_prompt) is not int or groups_per_prompt < 1:
        raise ValueError("Invalid original group count")
    cached = resume_candidates(config, split)
    reused = len(cached)
    if split in ("C", "D"):
        from .contract import preflight

        report = preflight(config, tasks_path)
        if report["status"] == "BLOCKED":
            raise ValueError(str(report["blockers"]))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    deadline = start + config["stage_max_seconds"]
    model, tokenizer = load_model(config)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    eos = eos or [tokenizer.eos_token_id]
    if None in eos:
        raise ValueError("Missing EOS")
    manifest = {
        "config": config,
        "actor_sampling": "iid",
        "split": split,
        "selected_prompt_ids": [task["prompt_id"] for task in tasks],
        "effective_groups_per_prompt": groups_per_prompt,
        "scope": config.get("scope", "configured_stage"),
        "template_hash": digest(tokenizer.chat_template),
        "eos_ids": eos,
        "trainable_parameter_manifest": {n: list(p.shape) for n, p in model.named_parameters()},
        "numeric_law": "FP64_softmax_of_FP32_logits",
        "generation_cache": config.get("use_cache", False),
        "provenance": provenance(config),
    }
    write_json(out / "manifest.json", manifest)
    count, tokens = 0, 0
    with (out / "rows.jsonl").open("x") as handle:
        for task in tasks:
            prompt = encode_prompt(tokenizer, task["prompt"])
            for group in range(groups_per_prompt):
                group_seed = int(digest([config["seed"], split, task["prompt_id"], group])[:15], 16)
                for slot in range(config["group_size"]):
                    seed = int(digest([group_seed, slot])[:15], 16)
                    if seed in cached:
                        row = cached.pop(seed)
                        if row.get("prompt_ids", prompt) != prompt:
                            raise ValueError("Cached candidate prompt mismatch")
                    else:
                        row = generate(
                            model,
                            prompt,
                            eos,
                            seed,
                            config["max_new_tokens"],
                            deadline,
                            use_cache=config.get("use_cache", False),
                        )
                    if row["max_token_logp_error"] > config["token_logp_tolerance"]:
                        write_json(out / "score_point_failure.json", row)
                        raise ValueError(
                            f"Sampling/scoring token logp disagreement: {row['max_token_logp_error']}"
                        )
                    row["sequence_logp_error"] = abs(
                        sum(row["scoring_token_logp"]) - row["old_logp"]
                    )
                    if row["sequence_logp_error"] > config.get(
                        "sequence_logp_tolerance", float("inf")
                    ):
                        write_json(out / "score_point_failure.json", row)
                        raise ValueError(
                            f"Sequence score point disagreement: {row['sequence_logp_error']}"
                        )
                    row.update(
                        {
                            "prompt_id": task["prompt_id"],
                            "prompt_ids": prompt,
                            "prompt_hash": digest(prompt),
                            "split": split,
                            "group_id": group,
                            "slot_id": slot,
                            "group_seed": group_seed,
                            "trajectory_id": f"{task['prompt_id']}:{group}:{slot}",
                            "text": tokenizer.decode(row["response_ids"], skip_special_tokens=True),
                        }
                    )
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    handle.flush()
                    count += 1
                    tokens += row["response_length"]
    if cached:
        raise ValueError("Partial bank contains candidates outside the frozen request")
    write_json(
        out / "collection.json",
        {
            "rows": count,
            "tokens": tokens,
            "reused_candidates": reused,
            "elapsed_seconds_this_invocation": time.monotonic() - start,
        },
    )
    return seal(out, "rows.jsonl")

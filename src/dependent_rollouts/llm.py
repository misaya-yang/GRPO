"""Transformers reference backend; optional imports keep CPU theory lightweight."""

import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np

from .artifacts import provenance, sha256, write_json, write_jsonl
from .estimators import advantages
from .prediction import iid_forecast_weights
from .sampling import ArithmeticDecoder, cdf_from_probs, latent_group
from .tasks import VERIFIER_VERSION, read_tasks, verify


def load_model(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id, revision = config["model_id"], config["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pin an exact 40-character model commit in config.revision")
    dtype = getattr(torch, config.get("dtype", "float32"))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        trust_remote_code=False,
        attn_implementation=config.get("attention", "eager"),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    model.to(config.get("device", "cuda"))
    model.eval()
    model.requires_grad_(True)
    return model, tokenizer


def sequence_logp(model, prompt_ids, response_ids):
    """SUM of response log-probs, includes emitted EOS; prompt is excluded."""
    import torch

    if not prompt_ids or not response_ids:
        raise ValueError("Prompt and response must be nonempty")
    ids = torch.tensor([prompt_ids + response_ids], device=next(model.parameters()).device)
    logits = model(input_ids=ids, use_cache=False).logits[0, len(prompt_ids) - 1 : -1]
    targets = ids[0, len(prompt_ids) :]
    return logits.float().log_softmax(-1).gather(1, targets[:, None]).sum()


def generate_group(model, prompt_ids, eos_ids, kind, k, seed, max_new_tokens):
    """Batched KV-cache generation. Complete groups, full support, temperature 1."""
    import torch

    if max_new_tokens < 1 or not prompt_ids or not eos_ids:
        raise ValueError("Need nonempty prompt, EOS set and positive horizon")
    model.eval()
    latents = latent_group(kind, k, seed)
    bins = [u.stratum(k) for u in latents]
    decoders = [ArithmeticDecoder(u) for u in latents]
    responses = [[] for _ in range(k)]
    logps = [0.0] * k
    tvs = [0.0] * k
    active = [True] * k
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids] * k, device=device)
    cache = None
    with torch.no_grad():
        for _ in range(max_new_tokens):
            output = model(input_ids=ids, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            logits = output.logits[:, -1].double()
            probs = logits.softmax(-1).cpu().numpy()
            log_probs = logits.log_softmax(-1).cpu().numpy()
            tokens = []
            for i in range(k):
                if not active[i]:
                    tokens.append(eos_ids[0])
                    continue
                cdf, tv = cdf_from_probs(probs[i])
                token = decoders[i].step(cdf)
                responses[i].append(token)
                logps[i] += float(log_probs[i, token])
                tvs[i] += tv
                active[i] = token not in eos_ids
                tokens.append(token)
            if not any(active):
                break
            ids = torch.tensor(tokens, device=device)[:, None]
    return [
        {
            "response_ids": response,
            "old_logp": lp,
            "latent_bin": bin_id,
            "latent_bits_used": latent.source.bits,
            "cdf_tv_sum": tv,
            "finish_reason": "length" if is_active else "eos",
        }
        for response, lp, bin_id, latent, tv, is_active in zip(
            responses, logps, bins, latents, tvs, active, strict=True
        )
    ]


def collect(config_path, tasks_path, output):
    config = json.loads(Path(config_path).read_text())
    tasks = read_tasks(tasks_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = provenance(config)
    manifest.update(
        {
            "stage": "collection",
            "status": "started",
            "tasks_sha256": sha256(tasks_path),
            "verifier_version": VERIFIER_VERSION,
            "loss": "detached_advantage_times_sum_response_logp_including_EOS",
            "sampler_numeric_law": "integer_CDF_binary64_weights_with_lazy_rational_sequence_uniform",
        }
    )
    write_json(out / "manifest.json", manifest)
    model, tokenizer = load_model(config)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    if not eos:
        eos = [tokenizer.eos_token_id]
    if None in eos:
        raise ValueError("Model/tokenizer has no EOS")
    rng = random.Random(config["seed"])
    rows, start = [], time.monotonic()
    max_seconds = config.get("max_wall_seconds", 6 * 3600)
    for task in tasks:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": task["prompt"]}], tokenize=True, add_generation_prompt=True
        )
        for group_id in range(config["groups_per_prompt"]):
            if time.monotonic() - start > max_seconds:
                write_jsonl(out / "partial_rollouts.jsonl", rows)
                raise TimeoutError(
                    "Collection time cap reached; partial bank is not a complete result"
                )
            seed = rng.getrandbits(128)
            generated = generate_group(
                model,
                prompt_ids,
                eos,
                config["sampler"],
                config["k"],
                seed,
                config["max_new_tokens"],
            )
            for member, row in enumerate(generated):
                text = tokenizer.decode(row["response_ids"], skip_special_tokens=True)
                row.update(
                    {
                        "prompt_id": task["prompt_id"],
                        "split": task["split"],
                        "group_id": group_id,
                        "member": member,
                        "block_id": f"{task['prompt_id']}:{group_id}",
                        "group_seed": seed,
                        "trajectory_id": f"{task['prompt_id']}:{group_id}:{member}",
                        "prompt_ids": prompt_ids,
                        "text": text,
                        "task": task,
                        **verify(text, task, row["finish_reason"]),
                    }
                )
                rows.append(row)
    write_jsonl(out / "rollouts.jsonl", rows)
    receipt = {
        "status": "complete",
        "rows": len(rows),
        "generated_tokens": sum(len(r["response_ids"]) for r in rows),
        "elapsed_seconds": time.monotonic() - start,
        "rollouts_sha256": sha256(out / "rollouts.jsonl"),
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        "eos_ids": eos,
        "max_cdf_tv_sum": max(r["cdf_tv_sum"] for r in rows),
    }
    write_json(out / "receipt.json", receipt)
    return receipt


def read_bank(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    receipt = json.loads((path / "receipt.json").read_text())
    if (
        receipt["status"] != "complete"
        or sha256(path / "rollouts.jsonl") != receipt["rollouts_sha256"]
    ):
        raise ValueError("Incomplete or modified rollout bank")
    rows = [json.loads(line) for line in (path / "rollouts.jsonl").read_text().splitlines()]
    k = manifest["config"]["k"]
    grouped = {}
    seen = set()
    for row in rows:
        if row["trajectory_id"] in seen or row["reward"] not in (0, 1):
            raise ValueError("Duplicate identity or nonbinary reward")
        seen.add(row["trajectory_id"])
        grouped.setdefault(row["prompt_id"], {}).setdefault(row["group_id"], []).append(row)
    result = {}
    for prompt, groups in grouped.items():
        ordered = []
        for group_id in sorted(groups):
            group = sorted(groups[group_id], key=lambda row: row["member"])
            if len(group) != k or [r["member"] for r in group] != list(range(k)):
                raise ValueError("Incomplete group; never subsample a larger-K sampler")
            ordered.extend(group)
        result[prompt] = ordered
    return manifest, result


def bank_weights(manifest, pools, arm, kind="rloo", epsilon=1e-6, lam=None):
    weighted = []
    k = manifest["config"]["k"]
    for rows in pools.values():
        r = np.array([row["reward"] for row in rows])
        if arm == "forecast":
            if manifest["config"]["sampler"] != "iid":
                raise ValueError("Forecasts must use an IID bank")
            if lam is not None:
                raise ValueError("Forecast arm takes no dose mixture")
            bins = [row["latent_bin"] for row in rows]
            weights = iid_forecast_weights(r, bins, k, kind, epsilon)
        else:
            weights = advantages(r.reshape(-1, k), arm, kind, epsilon, lam).reshape(-1) / len(rows)
        weighted.extend(
            (row, float(weight) / len(pools)) for row, weight in zip(rows, weights, strict=True)
        )
    return weighted


def aggregate_gradient(model, weighted_rows, score_tolerance=5e-3):
    """One aggregate full-parameter ascent gradient, no per-response gradients."""
    model.eval()
    model.zero_grad(set_to_none=True)
    max_error = 0.0
    for row, weight in weighted_rows:
        lp = sequence_logp(model, row["prompt_ids"], row["response_ids"])
        error = abs(lp.detach().item() - row["old_logp"])
        max_error = max(error, max_error)
        if error > score_tolerance:
            raise ValueError(f"Off-checkpoint or numerical score mismatch: {error:.6g}")
        if weight:
            (lp * weight).backward()
    gradients = {
        name: parameter.grad.detach().float().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    return gradients, max_error


def audit(bank, output, arm="within", kind="rloo", epsilon=1e-6, lam=None):
    from safetensors.torch import save_file

    manifest, pools = read_bank(bank)
    weighted = bank_weights(manifest, pools, arm, kind, epsilon, lam)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    model, _ = load_model(manifest["config"])
    gradients, error = aggregate_gradient(model, weighted)
    save_file(gradients, str(out / "gradient.safetensors"))
    write_jsonl(
        out / "weights.jsonl",
        [{"trajectory_id": r["trajectory_id"], "weight": w} for r, w in weighted],
    )
    result = {
        "arm": arm,
        "kind": kind,
        "epsilon": epsilon,
        "dose_lambda": lam if arm == "dose" else None,
        "bank": str(Path(bank).resolve()),
        "bank_sha256": sha256(Path(bank) / "rollouts.jsonl"),
        "config": manifest["config"],
        "parameterization": "full_parameters",
        "gradient_convention": "ascent",
        "gradient_sha256": sha256(out / "gradient.safetensors"),
        "max_score_point_error": error,
        "gradient_l2": math.sqrt(sum(float(g.double().square().sum()) for g in gradients.values())),
        "provenance": provenance(),
        "status": "frozen_finite_bank_gradient",
    }
    write_json(out / "receipt.json", result)
    return result

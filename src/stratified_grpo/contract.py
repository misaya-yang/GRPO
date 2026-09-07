"""Small, explicit v5.1 execution contract and sealed data identities."""

import json
import re
from pathlib import Path

from dependent_rollouts.artifacts import sha256
from dependent_rollouts.tasks import read_tasks
from reward_coupling.bank import digest


def validate(config):
    for key in ("K", "B", "m", "macro_repeats", "max_new_tokens", "stage_max_seconds"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    if config["K"] < 2 or config["B"] < config["K"] or config["macro_repeats"] < 2:
        raise ValueError("Need K>=2, B>=K and at least two independent macros")
    if config.get("arms") != ["iid_all", "stratified_full"]:
        raise ValueError("Primary arms must be IID-all and Strat-full")
    if config.get("stage") not in ("dev", "confirmation", "common_checkpoint"):
        raise ValueError("Unknown stage")
    if config.get("base_dtype") not in ("float32", "bfloat16"):
        raise ValueError("Unsupported base precision")
    for key in ("model_revision", "tokenizer_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", config.get(key, "")):
            raise ValueError("Pin actual model/tokenizer revisions")
    if (
        config.get("score") != "original_policy_sum_including_emitted_EOS"
        or config.get("epsilon") != 1e-6
    ):
        raise ValueError("Score or advantage contract differs")
    if config.get("sampling") != "independent_conditional_interval" or config.get("onset", -1) < 0:
        raise ValueError("Independent conditional interval sampler required")
    if config.get("token_order") != "vocabulary_id" or config.get("temperature") != 1.0:
        raise ValueError("Only frozen vocabulary ordering and temperature 1 are implemented")
    for key in ("token_logp_tolerance", "sequence_logp_tolerance", "cdf_tv_tolerance"):
        if not 0 < config.get(key, 0) < float("inf"):
            raise ValueError(f"Positive finite {key} required")
    if not 0 < config.get("practical_threshold", 0) < 1:
        raise ValueError("Freeze practical variance-time threshold")
    if config.get("metric") != "full_lora_gradient_trace":
        raise ValueError("This pipeline measures the full declared LoRA trace")
    return config


def selected_tasks(config, path):
    if sha256(path) != config["tasks_sha256"]:
        raise ValueError("Tasks file changed")
    rows = read_tasks(path)
    ids = config["prompt_ids"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Invalid frozen prompt ids")
    lookup = {r["prompt_id"]: r for r in rows}
    if not set(ids) <= lookup.keys():
        raise ValueError("Missing requested tasks")
    return [lookup[i] for i in ids]


def method_contract(config):
    keys = (
        "K",
        "B",
        "m",
        "model_revision",
        "tokenizer_revision",
        "checkpoint_manifest_sha256",
        "base_dtype",
        "lora",
        "adapter_sha256",
        "sampling",
        "onset",
        "token_order",
        "temperature",
        "max_new_tokens",
        "epsilon",
        "score",
        "metric",
        "token_logp_tolerance",
        "sequence_logp_tolerance",
        "cdf_tv_tolerance",
        "practical_threshold",
        "use_cache",
    )
    return {k: config.get(k) for k in keys}


def require_dev_freeze(config):
    if config["stage"] == "dev":
        return
    path = Path(config["dev_freeze"])
    if sha256(path) != config["dev_freeze_sha256"]:
        raise ValueError("Dev freeze changed")
    freeze = json.loads(path.read_text())
    if freeze["status"] != "FROZEN_FOR_SCREENING" or freeze["method_hash"] != digest(
        method_contract(config)
    ):
        raise ValueError("Dev method does not match confirmation")
    if set(config["prompt_ids"]) & set(freeze["dev_prompt_ids"]):
        raise ValueError("Dev and confirmation prompts overlap")
    if config["seed"] == freeze["dev_seed"]:
        raise ValueError("Use a fresh confirmation seed")
    for evidence in freeze["evidence"]:
        if sha256(evidence["path"]) != evidence["sha256"]:
            raise ValueError("Dev evidence changed")

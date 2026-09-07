"""Preparation/readiness is separate from pilot GO and GPU execution."""

import json
import math
import re
from pathlib import Path

from dependent_rollouts.artifacts import sha256

from .bank import read_tasks


def validate_config(config, require_pins=True):
    fixed = {
        "actor_sampling": "iid",
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "std_ddof": 0,
        "epsilon": 1e-6,
        "epsilon_position": "outside_sqrt",
        "loss_reduction": "sum_tokens_then_mean_responses",
        "clip_policy_ratio": False,
        "kl_coefficient": 0.0,
        "optimizer_map": "identity",
        "primary_gradient_scope": "all_actor_parameters",
        "allow_reward_based_filtering": False,
        "dtype": "float32",
        "attention": "eager",
    }
    for key, value in fixed.items():
        if config.get(key) != value:
            raise ValueError(f"Feedback contract requires {key}={value!r}")
    for key in ("group_size", "groups_per_prompt", "max_new_tokens"):
        if type(config.get(key)) is not int or config[key] < (2 if key == "group_size" else 1):
            raise ValueError(f"Invalid {key}")
    if config.get("feedback") not in ("code", "rubric"):
        raise ValueError("Unknown feedback channel")
    if config.get("executor", "native") not in ("native", "docker"):
        raise ValueError("Unknown executor")
    for key, upper in (("test_timeout_seconds", 60), ("stage_max_seconds", None)):
        value = config.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {key}")
        if upper is not None and value > upper:
            raise ValueError(f"Invalid {key}")
    tolerance = config.get("token_logp_tolerance")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid token_logp_tolerance")
    if require_pins:
        for key in ("model_revision", "tokenizer_revision"):
            if not re.fullmatch(r"[0-9a-f]{40}", config.get(key, "")):
                raise ValueError(f"Pin {key} before model loading")
        if not re.fullmatch(r"[0-9a-f]{64}", config.get("tasks_sha256", "")):
            raise ValueError("Freeze task manifest SHA256")
    return config


def preflight(config, tasks_path=None):
    blockers = []
    try:
        validate_config(config)
    except ValueError as exc:
        blockers.append(str(exc))
    for key in ("model_revision", "tokenizer_revision", "tasks_sha256"):
        if "MUST_" in str(config.get(key, "MUST_SUPPLY")):
            blockers.append(f"Unfrozen: {key}")
    if tasks_path:
        tasks = read_tasks(tasks_path)
        if sha256(tasks_path) != config.get("tasks_sha256"):
            blockers.append("Task manifest hash mismatch")
        counts = {s: sum(t["split"] == s for t in tasks) for s in ("Dev", "C", "D")}
        for split, expected in config["prompt_counts"].items():
            if counts.get(split) != expected:
                blockers.append(f"{split}: expected {expected} prompts, found {counts.get(split)}")
    else:
        blockers.append("No frozen task manifest supplied")
    for key in ("reference_receipt", "dev_freeze"):
        path = config.get(key)
        if not path or not Path(path).is_file():
            blockers.append(f"Missing {key}")
        elif sha256(path) != config.get(key + "_sha256"):
            blockers.append(f"Modified {key}")
    return {
        "status": "BLOCKED" if blockers else "PREPARED_REQUIRES_GPU_CONTRACT",
        "blockers": blockers,
        "pilot_decision": "NOT_RUN",
        "pretrained_evidence": False,
    }


def check_task_hash(config, path):
    if sha256(path) != config["tasks_sha256"]:
        raise ValueError("Task manifest changed since freeze")


def read_config(path):
    return validate_config(json.loads(Path(path).read_text()))

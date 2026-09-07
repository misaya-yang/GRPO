"""Frozen task/actor/feedback identities with complete original group ownership."""

import hashlib
import json
from pathlib import Path

import numpy as np

from dependent_rollouts.artifacts import sha256, write_json

from .advantages import advantages
from .expectation import expected_advantages, rubric_matrix


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_tasks(path):
    tasks = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    ids, prompts = set(), set()
    for task in tasks:
        normalized = " ".join(task["prompt"].split())
        if (
            task["prompt_id"] in ids
            or normalized in prompts
            or task["split"] not in ("Dev", "C", "D", "train", "evaluation")
        ):
            raise ValueError("Duplicate task or invalid split")
        if "criteria" in task:
            if not task["criteria"] or not all(isinstance(c, str) and c for c in task["criteria"]):
                raise ValueError("Need frozen nonempty rubric criteria")
        else:
            if not task.get("tests") or not all(isinstance(t, str) and t for t in task["tests"]):
                raise ValueError("Need frozen per-test Python assertions")
            if not task.get("reference_code"):
                raise ValueError("Missing reference program")
        ids.add(task["prompt_id"])
        prompts.add(normalized)
    if not tasks:
        raise ValueError("Empty task manifest")
    return tasks


def seal(directory, filename, status="complete"):
    directory = Path(directory)
    receipt = {
        "status": status,
        "file": filename,
        "sha256": sha256(directory / filename),
        "manifest_sha256": sha256(directory / "manifest.json"),
    }
    write_json(directory / "receipt.json", receipt)
    return receipt


def read_bank(path):
    path = Path(path)
    receipt = json.loads((path / "receipt.json").read_text())
    if receipt["status"] != "complete" or receipt["file"] != "rows.jsonl":
        raise ValueError("Incomplete bank")
    if (
        sha256(path / "rows.jsonl") != receipt["sha256"]
        or sha256(path / "manifest.json") != receipt["manifest_sha256"]
    ):
        raise ValueError("Modified bank or manifest")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("actor_sampling") != "iid":
        raise ValueError("Feedback branch requires IID actor")
    eos_ids = manifest.get("eos_ids")
    if (
        not isinstance(eos_ids, list)
        or not eos_ids
        or any(type(token) is not int for token in eos_ids)
        or len(set(eos_ids)) != len(eos_ids)
    ):
        raise ValueError("Missing or invalid frozen EOS identity")
    rows = [json.loads(s) for s in (path / "rows.jsonl").read_text().splitlines()]
    config = manifest["config"]
    k = config["group_size"]
    tolerance = config["token_logp_tolerance"]
    groups, ids, actor_streams = {}, set(), set()
    for row in rows:
        identity = row["trajectory_id"]
        if identity in ids:
            raise ValueError("Duplicate trajectory")
        ids.add(identity)
        if row["actor_rng_stream"] in actor_streams:
            raise ValueError("Duplicate actor RNG stream")
        actor_streams.add(row["actor_rng_stream"])
        if row.get("prompt_hash") != digest(row["prompt_ids"]):
            raise ValueError("Prompt identity mismatch")
        response = row["response_ids"]
        sampling = np.asarray(row["sampling_token_logp"], float)
        scoring = np.asarray(row["scoring_token_logp"], float)
        if (
            not response
            or row["response_length"] != len(response)
            or sampling.shape != (len(response),)
            or scoring.shape != (len(response),)
        ):
            raise ValueError("Missing response token scores")
        if not np.isfinite(sampling).all() or not np.isfinite(scoring).all():
            raise ValueError("Nonfinite token scores")
        if np.max(np.abs(sampling - scoring)) > tolerance:
            raise ValueError("Sampling/scoring token logp disagreement")
        if abs(float(sampling.sum() - scoring.sum())) > config.get(
            "sequence_logp_tolerance", float("inf")
        ):
            raise ValueError("Sequence score point disagreement")
        if not np.isclose(row["old_logp"], sampling.sum(), atol=tolerance, rtol=0):
            raise ValueError("Response logp sum mismatch")
        if row["active_mask"] != [1] * len(response):
            raise ValueError("Padding or incomplete active mask")
        emitted = [i for i, token in enumerate(response) if token in eos_ids]
        expected_eos = len(response) - 1 if emitted else None
        if (
            emitted not in ([], [len(response) - 1])
            or row["eos_index_or_null"] != expected_eos
            or row["truncated"] is not (expected_eos is None)
        ):
            raise ValueError("EOS/truncation identity mismatch")
        if manifest.get("split") != row["split"]:
            raise ValueError("Bank split mismatch")
        groups.setdefault((row["prompt_id"], row["group_id"]), []).append(row)
    if not groups:
        raise ValueError("Empty bank")
    for group in groups.values():
        group.sort(key=lambda r: r["slot_id"])
        if len(group) != k or [r["slot_id"] for r in group] != list(range(k)):
            raise ValueError("Incomplete original actor group")
        for key in ("prompt_ids", "split", "group_seed"):
            if any(r[key] != group[0][key] for r in group):
                raise ValueError(f"Group contract mismatch: {key}")
    return manifest, groups


def group_weights(group, config):
    b = np.array([r["test_verdicts"] for r in group], float)
    if b.ndim != 2 or not np.isin(b, [0, 1]).all():
        raise ValueError("Need complete binary test/criterion verdicts")
    for row in group:
        if row["test_manifest_hash"] != group[0]["test_manifest_hash"]:
            raise ValueError("Mixed test/criterion contracts")
    scale = None
    if config["feedback"] == "rubric":
        scale = config["rubric_keep"]
        matrix, masks = rubric_matrix(b, scale)
        contexts = [list(mask) for mask in masks]
    else:
        matrix, contexts = b, list(range(b.shape[1]))
    mu = np.ones(matrix.shape[1]) / matrix.shape[1]
    result = expected_advantages(matrix, mu, epsilon=config["epsilon"], grid_scale=scale)
    result["full_rubric"] = advantages(b.mean(axis=1), epsilon=config["epsilon"])
    result["suite"] = b.prod(axis=1)
    result["contexts"], result["context_probabilities"] = contexts, mu
    return result


def assert_independent(left, right):
    left_rows = [r for group in left.values() for r in group]
    right_rows = [r for group in right.values() for r in group]
    for key in ("prompt_id", "prompt_hash", "group_seed", "actor_rng_stream"):
        if {x[key] for x in left_rows} & {x[key] for x in right_rows}:
            raise ValueError(f"C/D independence violated: {key}")

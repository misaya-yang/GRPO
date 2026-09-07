"""Deterministic test matrices or explicitly cached, response-local judge verdicts."""

import json
from pathlib import Path

from dependent_rollouts.artifacts import sha256, write_json, write_jsonl

from .bank import digest, group_weights, read_bank, read_tasks, seal
from .contract import check_task_hash
from .execute_tests import make_executor


def score(bank, tasks_path, output, verdicts_path=None):
    manifest, groups = read_bank(bank)
    config = manifest["config"]
    check_task_hash(config, tasks_path)
    tasks = {t["prompt_id"]: t for t in read_tasks(tasks_path)}
    cached = {}
    if config["feedback"] == "rubric":
        if not verdicts_path:
            raise ValueError("Supply complete response-local criterion verdict bank")
        payload = json.loads(Path(verdicts_path).read_text())
        for key in (
            "judge_id",
            "judge_revision",
            "prompt_template_hash",
            "criterion_manifest_hash",
            "private_randomness",
        ):
            if not payload.get(key):
                raise ValueError(f"Missing judge provenance: {key}")
        if payload.get("response_local") is not True:
            raise ValueError("Group-reading judges violate exogenous feedback contract")
        for row in payload["rows"]:
            if row["trajectory_id"] in cached:
                raise ValueError("Duplicate cached verdict")
            cached[row["trajectory_id"]] = row
        ids = {r["trajectory_id"] for g in groups.values() for r in g}
        if set(cached) != ids:
            raise ValueError("Cached verdict bank coverage mismatch")
    else:
        executor = make_executor(config)
        executor.preflight()
        relevant = [tasks[p] for p in sorted({p for p, _ in groups})]
        executor.check_references(relevant)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        **manifest,
        "candidate_bank_sha256": sha256(Path(bank) / "rows.jsonl"),
        "scoring": "cached_response_local_judge" if cached else "fresh_container_each_test",
        "verdict_bank_sha256": sha256(verdicts_path) if cached else None,
    }
    write_json(out / "manifest.json", manifest)
    with (out / "rows.jsonl").open("x") as handle:
        for group in groups.values():
            for original in group:
                row = dict(original)
                task = tasks[row["prompt_id"]]
                if cached:
                    entry = cached[row["trajectory_id"]]
                    if entry["response_hash"] != digest(row["response_ids"]):
                        raise ValueError("Judge verdict refers to a different response")
                    if "criteria" in task and len(entry["criterion_verdicts"]) != len(
                        task["criteria"]
                    ):
                        raise ValueError("Judge did not cover this prompt's criterion set")
                    row.update(
                        {
                            "test_verdicts": entry["criterion_verdicts"],
                            "test_manifest_hash": payload["criterion_manifest_hash"],
                            "judge_provenance": {k: v for k, v in payload.items() if k != "rows"},
                        }
                    )
                else:
                    row.update(executor.evaluate(row["text"], task))
                    row["test_manifest_hash"] = digest(task["tests"])
                row["suite_result"] = int(all(row["test_verdicts"]))
                row["average_test_result"] = sum(row["test_verdicts"]) / len(row["test_verdicts"])
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
    # Full shape/value validation before sealing, no deletion/replacement.
    rows = [json.loads(s) for s in (out / "rows.jsonl").read_text().splitlines()]
    for offset in range(0, len(rows), config["group_size"]):
        group_weights(rows[offset : offset + config["group_size"]], config)
    return seal(out, "rows.jsonl")


def integrate(bank, output):
    manifest, groups = read_bank(bank)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "manifest.json", {**manifest, "scored_bank_sha256": sha256(Path(bank) / "rows.jsonl")}
    )
    rows = []
    for group in groups.values():
        weights = group_weights(group, manifest["config"])
        for slot, row in enumerate(group):
            rows.append(
                {
                    "trajectory_id": row["trajectory_id"],
                    "prompt_id": row["prompt_id"],
                    "group_id": row["group_id"],
                    "context_ids": weights["contexts"],
                    "context_probabilities": weights["context_probabilities"],
                    **{
                        arm: float(weights[arm][slot])
                        for arm in (
                            "shared",
                            "independent",
                            "rloo",
                            "difference",
                            "average",
                            "suite",
                            "full_rubric",
                        )
                    },
                }
            )
    write_jsonl(out / "weights.jsonl", rows)
    return seal(out, "weights.jsonl")

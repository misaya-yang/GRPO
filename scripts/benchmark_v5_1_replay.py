"""Time an exact replay of one existing Dev macro, without adding research samples."""

import argparse
import json
import time
from pathlib import Path

from dependent_rollouts.artifacts import sha256, write_json
from dependent_rollouts.tasks import read_tasks
from stratified_grpo.model import aggregate_gradient, load_model
from stratified_grpo.pipeline import collect_macro, compiled_weights

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--dev", required=True)
p.add_argument("--tasks", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
dev = Path(a.dev)
c = json.loads((dev / "manifest.json").read_text())["config"]
original = [json.loads(s) for s in (dev / "rows.jsonl").read_text().splitlines()]
expected = [
    r
    for r in original
    if r["prompt_id"] == c["prompt_ids"][0] and r["macro"] == 0 and r["arm"] == "stratified_full"
]
task = next(t for t in read_tasks(a.tasks) if t["prompt_id"] == c["prompt_ids"][0])
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
t = time.monotonic()
model, tokenizer = load_model(c)
load = time.monotonic() - t
rows, costs = collect_macro(model, tokenizer, c, task, 0, "stratified_full", time.monotonic() + 600)
for r, e in zip(rows, expected, strict=True):
    for key in (
        "response_ids",
        "sampling_token_logp",
        "rng_seed",
        "reward",
        "finish_reason",
        "release_step",
        "fresh_bits_used",
    ):
        if r[key] != e[key]:
            raise ValueError(f"Exact replay changed {key}: {r['response_id']}")
t = time.monotonic()
w = compiled_weights(rows, c, "stratified_full")
costs["weight_seconds"] = time.monotonic() - t
t = time.monotonic()
g, errors = aggregate_gradient(
    model, rows, w, c["token_logp_tolerance"], c["sequence_logp_tolerance"]
)
costs["replay_seconds"] = time.monotonic() - t
old = next(
    json.loads(s)
    for s in (dev / "progress.jsonl").read_text().splitlines()
    if json.loads(s)["prompt_id"] == task["prompt_id"]
    and json.loads(s)["macro"] == 0
    and json.loads(s)["arm"] == "stratified_full"
)
report = {
    "status": "EXACT_REPLAY_PASS",
    "sample_count_added": 0,
    "rows": len(rows),
    "original_rows_sha256": sha256(dev / "rows.jsonl"),
    "costs": costs,
    "old_costs": old["costs"],
    "new_over_old_macro_cost": sum(costs.values()) / old["total_macro_seconds"],
    "model_load_seconds": load,
    "errors": errors,
}
write_json(out / "receipt.json", report)
print(json.dumps(report, indent=2))

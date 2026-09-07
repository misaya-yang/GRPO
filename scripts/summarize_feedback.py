"""Summarize a sealed fixed bank; candidate-credit diagnostics, not a pilot verdict."""

import argparse
import json
from pathlib import Path

import numpy as np

from dependent_rollouts.artifacts import sha256, write_json
from reward_coupling.bank import group_weights, read_bank
from reward_coupling.expectation import expected_advantages
from reward_coupling.statistics import positive_scale

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bank", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
manifest, groups = read_bank(args.bank)
rows = [r for group in groups.values() for r in group]
summaries = []
negative_error = 0.0
for (prompt, group_id), group in groups.items():
    weights = group_weights(group, manifest["config"])
    b = np.array([r["test_verdicts"] for r in group])
    if manifest["config"]["feedback"] == "code":
        for k in (2, 3):
            control = expected_advantages(b[:k])
            negative_error = max(negative_error, float(np.max(abs(control["difference"]))))
    summaries.append(
        {
            "prompt_id": prompt,
            "group_id": group_id,
            "average_test_reward": float(b.mean()),
            "suite_rate": float(b.prod(axis=1).mean()),
            "shared_credit_norm": float(np.linalg.norm(weights["shared"])),
            "independent_credit_norm": float(np.linalg.norm(weights["independent"])),
            "difference_credit_norm": float(np.linalg.norm(weights["difference"])),
            "candidate_credit_positive_scale": positive_scale(
                weights["shared"], weights["independent"]
            ),
            "interpretation": "conditional_fixed_candidate_credit_not_functional_gradient",
        }
    )
result = {
    "status": "FIXED_BANK_CREDIT_DIAGNOSTIC",
    "bank_sha256": sha256(Path(args.bank) / "rows.jsonl"),
    "model": manifest["config"]["model_id"],
    "revision": manifest["config"]["model_revision"],
    "prompts": len({r["prompt_id"] for r in rows}),
    "groups": len(groups),
    "responses": len(rows),
    "generated_tokens": sum(r["response_length"] for r in rows),
    "truncated": sum(r["truncated"] for r in rows),
    "max_token_logp_error": max(r["max_token_logp_error"] for r in rows),
    "suite_passes": sum(r["suite_result"] for r in rows),
    "parse_failures": sum(r.get("parser_status") != "parsed" for r in rows),
    "K2_K3_binary_negative_control_max_error": negative_error,
    "group_diagnostics": summaries,
    "pretrained_gradient_difference": "NOT_MEASURED_HERE",
    "pilot_decision": "INCONCLUSIVE",
    "independent_confirmation": False,
}
write_json(args.output, result)
print(json.dumps(result, indent=2))

"""CPU-only post-hoc diagnosis of saved rows/Grams; never changes run rewards."""

import argparse
import hashlib
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from dependent_rollouts.tasks import verify
from stratified_grpo.statistics import compare, trace_variance
from stratified_grpo.weights import full_stratified_weights, iid_weights


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screen = args.run / "screen"
    config = json.loads((args.run / "screen_config.json").read_text())
    rows = read_lines(screen / "rows.jsonl")
    events = read_lines(screen / "progress.jsonl")
    records = json.loads((screen / "macro_statistics.json").read_text())
    receipt = json.loads((screen / "receipt.json").read_text())
    tasks = {r["prompt_id"]: r for r in read_lines(args.tasks)}
    hashes = {
        name: hashlib.sha256((screen / name).read_bytes()).hexdigest()
        for name in ("rows.jsonl", "progress.jsonl", "macro_statistics.json", "analysis.json")
    }
    assert hashes["rows.jsonl"] == receipt["rows_sha256"]
    assert hashes["progress.jsonl"] == receipt["progress_sha256"]
    assert len(rows) == receipt["rows"] == 256
    assert len(events) == receipt["macros"] == 32
    assert len({r["response_id"] for r in rows}) == len(rows)
    assert len({r["rng_seed"] for r in rows}) == len(rows)
    assert all(
        verify(r["text"], tasks[r["prompt_id"]], r["finish_reason"])
        == {"reward": r["reward"], "category": r["category"]}
        for r in rows
    )
    max_weight_error = 0.0
    for event in events:
        bank = [r for r in rows if all(r[k] == event[k] for k in ("prompt_id", "macro", "arm"))]
        assert len(bank) == 8
        assert [(r["block"], r["slot"]) for r in bank] == [
            (b, s) for b in range(4) for s in range(2)
        ]
        fn = iid_weights if event["arm"] == "iid_all" else full_stratified_weights
        weights = fn(np.array([r["reward"] for r in bank]).reshape(4, 2), 4, config["epsilon"])
        max_weight_error = max(
            max_weight_error, float(np.max(np.abs(weights.ravel() - event["weights"])))
        )
    assert max_weight_error < 1e-12
    reproduction = compare(records, config["practical_threshold"], 2000, config["analysis_seed"])
    original = json.loads((screen / "analysis.json").read_text())
    for key in ("rho", "rho_time", "rho_interval", "rho_time_interval"):
        assert np.allclose(reproduction[key], original[key], atol=1e-12, rtol=0)

    def ratio(selected):
        vi = np.mean([trace_variance(r["iid_gram"]) for r in selected])
        vs = np.mean([trace_variance(r["stratified_gram"]) for r in selected])
        q = np.mean([np.mean(r["stratified_costs"]) for r in selected]) / np.mean(
            [np.mean(r["iid_costs"]) for r in selected]
        )
        return {"rho": float(vs / vi), "q": float(q), "rho_time": float(q * vs / vi)}

    prompt_results = []
    for record in records:
        pid = record["prompt_id"]
        result = {"prompt_id": pid, "source_id": tasks[pid]["source_id"], **ratio([record])}
        for arm, label in (("iid_all", "iid"), ("stratified_full", "stratified")):
            rr = [r for r in rows if r["prompt_id"] == pid and r["arm"] == arm]
            ee = [e for e in events if e["prompt_id"] == pid and e["arm"] == arm]
            result[label] = {
                "categories": dict(Counter(r["category"] for r in rr)),
                "variance": trace_variance(record[label + "_gram"]),
                "all_equal_macros": sum(e["all_equal_rewards"] for e in ee),
                "macro_reward_means": [e["reward_mean"] for e in ee],
                "gradient_norms": [e["l2_B"] for e in ee],
                "mean_tokens": float(np.mean([len(r["response_ids"]) for r in rr])),
            }
        strat = [r for r in rows if r["prompt_id"] == pid and r["arm"] == "stratified_full"]
        result["stratum_successes_out_of_16"] = [
            sum(r["reward"] for r in strat if r["stratum"] == j) for j in range(2)
        ]
        result["release_quantiles"] = np.quantile(
            [r["release_step"] for r in strat], [0, 0.5, 0.9, 1]
        ).tolist()
        result["mean_difference_to_noise_rms"] = (
            record["mean_gradient_difference_l2"] / record["estimated_sampling_noise_rms"]
        )
        result["leave_this_prompt_out"] = ratio([r for r in records if r["prompt_id"] != pid])
        prompt_results.append(result)

    # Diagnostic deletion only: primary estimates retain every bank and prompt.
    deletions = []
    for i, record in enumerate(records):
        for arm in ("iid", "stratified"):
            for bank in range(4):
                changed = json.loads(json.dumps(records))
                keep = [j for j in range(4) if j != bank]
                changed[i][arm + "_gram"] = np.asarray(record[arm + "_gram"])[
                    np.ix_(keep, keep)
                ].tolist()
                changed[i][arm + "_costs"] = np.asarray(record[arm + "_costs"])[keep].tolist()
                deletions.append(
                    {"prompt_id": record["prompt_id"], "arm": arm, "bank": bank, **ratio(changed)}
                )

    # Counterfactual format check: exactly one numeric tag, allow text after it;
    # still require EOS. This is NOT a replacement reward or a new gradient result.
    relaxed = []
    for row in rows:
        tags = re.findall(r"<answer>([^<>]+)</answer>", row["text"])
        valid = row["finish_reason"] == "eos" and len(tags) == 1
        valid = valid and bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\d+/\d+)", tags[0].strip()))
        try:
            correct = valid and Fraction(tags[0].strip()) == Fraction(
                tasks[row["prompt_id"]]["target"]
            )
        except ZeroDivisionError:
            correct = False
        relaxed.append(
            {
                "response_id": row["response_id"],
                "old_reward": row["reward"],
                "diagnostic_reward": int(correct),
            }
        )
    output = {
        "status": "ANALYZED; saved-statistic and reward/weight reproduction passed; no model replay",
        "scope": "post_hoc_exploratory; no new samples, no reward/config mutation",
        "source_hashes": hashes,
        "max_weight_error": max_weight_error,
        "reproduced_primary": reproduction,
        "by_prompt": prompt_results,
        "categories": dict(Counter(r["category"] for r in rows)),
        "leave_one_macro_out": deletions,
        "format_diagnostic": {
            "definition": "single numeric answer tag, EOS required, trailing text allowed",
            "correct_count": sum(r["diagnostic_reward"] for r in relaxed),
            "changed_rows": [r for r in relaxed if r["old_reward"] != r["diagnostic_reward"]],
        },
        "costs_by_arm": {
            arm: {
                **{
                    k: sum(e["costs"][k] for e in events if e["arm"] == arm)
                    for k in events[0]["costs"]
                },
                "tokens": sum(len(r["response_ids"]) for r in rows if r["arm"] == arm),
            }
            for arm in ("iid_all", "stratified_full")
        },
        "projection_variance_ratios": [
            float(
                np.mean(
                    [
                        np.var(
                            [
                                e["functional_projections"][j]
                                for e in events
                                if e["prompt_id"] == r["prompt_id"]
                                and e["arm"] == "stratified_full"
                            ],
                            ddof=1,
                        )
                        for r in records
                    ]
                )
                / np.mean(
                    [
                        np.var(
                            [
                                e["functional_projections"][j]
                                for e in events
                                if e["prompt_id"] == r["prompt_id"] and e["arm"] == "iid_all"
                            ],
                            ddof=1,
                        )
                        for r in records
                    ]
                )
            )
            for j in range(4)
        ],
        "unmeasured": [
            "per_response_score",
            "stratum_influence_function_mean",
            "counterfactual_reward_gradient",
            "online_training_gain",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "categories": output["categories"],
                "format_correct_count": output["format_diagnostic"]["correct_count"],
                "by_prompt": prompt_results,
                "costs": output["costs_by_arm"],
                "deletion_rho_range": [
                    min(x["rho"] for x in deletions),
                    max(x["rho"] for x in deletions),
                ],
                "projections": output["projection_variance_ratios"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

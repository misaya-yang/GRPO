"""Common-step S/I calibration with Dev-frozen reference likelihood readouts.

The shared gradient is reconstructed as G_I + (G_S-G_I) on the identical bank.
Dev calibrates the step; C may reuse those frozen functions with an independent D
bank. This uses linearity, not a second independent estimator. No pilot GO is emitted.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from dependent_rollouts.artifacts import sha256, write_json
from dependent_rollouts.statistics import importance_reward
from reward_coupling.bank import assert_independent, digest, group_weights, read_bank, read_tasks
from reward_coupling.gradient_audit import aggregate_gradient, load_gradient
from reward_coupling.local_steps import _projected_gradient, evaluate, perturb_from_checkpoint
from reward_coupling.sample import encode_prompt, load_model, token_logps
from reward_coupling.statistics import positive_scale

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bank", required=True)
parser.add_argument("--difference", required=True)
parser.add_argument("--tasks", required=True)
parser.add_argument("--step", type=float, default=1e-4)
parser.add_argument("--output", required=True)
parser.add_argument("--functions")
parser.add_argument("--evaluation-bank")
parser.add_argument("--project-evaluation", action="store_true")
parser.add_argument("--project-only", action="store_true")
args = parser.parse_args()
if not np.isfinite(args.step) or args.step <= 0:
    raise ValueError("Need a positive Dev step")
if args.project_evaluation and not args.evaluation_bank:
    raise ValueError("--project-evaluation requires --evaluation-bank")
manifest, groups = read_bank(args.bank)
splits = {row["split"] for group in groups.values() for row in group}
if args.project_only and (
    splits != {"C"} or not args.functions or not args.evaluation_bank or not args.project_evaluation
):
    raise ValueError(
        "--project-only requires C, Dev-frozen functions, independent D, and --project-evaluation"
    )
if splits not in ({"Dev"}, {"C"}) or (
    splits == {"C"} and (not args.functions or not args.evaluation_bank)
):
    raise ValueError(
        "C stage requires Dev-frozen functions and independent D; selection is Dev-only"
    )
config = manifest["config"]
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
source_path = Path(__file__).resolve()
evaluation_path = Path(args.evaluation_bank).resolve() if args.evaluation_bank else None
execution_identity = {
    "script_source": str(source_path),
    "script_source_sha256": sha256(source_path),
    "evaluation_bank": str(evaluation_path) if evaluation_path else None,
    "evaluation_bank_rows_sha256": sha256(evaluation_path / "rows.jsonl")
    if evaluation_path
    else None,
    "evaluation_bank_manifest_sha256": sha256(evaluation_path / "manifest.json")
    if evaluation_path
    else None,
}
write_json(
    out / "plan.json",
    {
        "config": config,
        "bank_sha256": sha256(Path(args.bank) / "rows.jsonl"),
        "step": args.step,
        "half_step": args.step / 2,
        "readout_rule": "first two frozen Dev prompts: canonical function vs constant-None stub log odds",
        "status": "Dev_calibration_predeclared_before_functional_readout"
        if splits == {"Dev"}
        else "C_D_preliminary_frozen_readout",
        "functions_source": args.functions,
        **execution_identity,
    },
)
start = time.monotonic()
model, tokenizer = load_model(config)
model._feedback_offload_activations = True
eos = model.generation_config.eos_token_id
eos = eos[0] if isinstance(eos, list) else eos
if args.functions:
    features = json.loads(Path(args.functions).read_text())
    source_directory = Path(args.functions).parent
    source_plan = json.loads((source_directory / "plan.json").read_text())
    source_receipt = json.loads((source_directory / "receipt.json").read_text())
    if source_receipt.get("functions_sha256") != sha256(args.functions):
        raise ValueError("Frozen functions changed after Dev calibration")
    for key in ("model_id", "model_revision", "tokenizer_revision", "checkpoint_manifest_sha256"):
        if source_plan["config"].get(key) != config.get(key):
            raise ValueError("Frozen function checkpoint mismatch")
else:
    features = []
    for task in [t for t in read_tasks(args.tasks) if t["split"] == "Dev"][:2]:
        prompt = encode_prompt(tokenizer, task["prompt"])
        texts = [
            task["reference_code"].strip(),
            f"def {task['entry_point']}(*args, **kwargs):\n    return None",
        ]
        definitions = []
        for coefficient, text in zip((1.0, -1.0), texts, strict=True):
            response = tokenizer.encode(
                "```python\n" + text + "\n```", add_special_tokens=False
            ) + [eos]
            row = {
                "prompt_ids": prompt,
                "response_ids": response,
                "coefficient": coefficient,
                "split": "Dev",
            }
            with torch.no_grad():
                row["sampling_token_logp"] = token_logps(model, prompt, response).cpu().tolist()
            definitions.append(row)
        features.append({"prompt_id": task["prompt_id"], "definitions": definitions})
if (
    not features
    or len({feature["prompt_id"] for feature in features}) != len(features)
    or any(
        not feature.get("definitions")
        or any(row.get("split") != "Dev" for row in feature["definitions"])
        for feature in features
    )
):
    raise ValueError("Need unique, nonempty Dev-frozen function definitions")
write_json(out / "functions.json", features)


def values():
    return np.array(
        [
            sum(
                r["coefficient"] * lp
                for r, lp in zip(
                    feature["definitions"], evaluate(model, feature["definitions"]), strict=True
                )
            )
            for feature in features
        ]
    )


baseline = values()
evaluation_rows = []
evaluation_values = {}
if args.evaluation_bank:
    evaluation_manifest, evaluation_groups = read_bank(args.evaluation_bank)
    assert_independent(groups, evaluation_groups)
    if evaluation_manifest["config"] != config:
        raise ValueError("C/D generating configuration mismatch")
    evaluation_rows = [r for group in evaluation_groups.values() for r in group]
    if any(r["split"] != "D" for r in evaluation_rows):
        raise ValueError("Expected D evaluation bank")
    evaluation_base = evaluate(model, evaluation_rows)
    max_eval_gap = max(abs(evaluation_base - np.array([r["old_logp"] for r in evaluation_rows])))
    evaluation_tolerance = config.get("sequence_logp_tolerance")
    if evaluation_tolerance is None:
        evaluation_tolerance = config["token_logp_tolerance"] * max(
            r["response_length"] for r in evaluation_rows
        )
    if max_eval_gap > evaluation_tolerance:
        raise ValueError("Evaluation bank score point mismatch")
dm, difference = load_gradient(args.difference)
if (
    dm["bank_sha256"] != sha256(Path(args.bank) / "rows.jsonl")
    or dm["config"] != config
    or dm["arm"] != "difference"
):
    raise ValueError("Difference must come from the identical frozen treatment bank")
prediction = []
projection_error = 0.0
for feature in features:
    weighted = [(row, row["coefficient"]) for row in feature["definitions"]]
    slope, error = _projected_gradient(model, weighted, difference, config["token_logp_tolerance"])
    projection_error = max(projection_error, error)
    prediction.append(slope)
    print(
        json.dumps(
            {
                "stage": "autograd_prediction",
                "prompt_id": feature["prompt_id"],
                "slope": slope,
                "token_error": error,
            }
        ),
        flush=True,
    )
prediction = np.array(prediction)
write_json(
    out / "predicted_slopes.json",
    {"values": prediction, "functions_sha256": sha256(out / "functions.json")},
)
target_slopes = None
if evaluation_rows and args.project_evaluation:
    directional = []
    partial_path = out / "D_target_rows.partial.jsonl"
    with partial_path.open("x") as partial:
        for index, row in enumerate(evaluation_rows):
            average, suite = (
                float(np.mean(row["test_verdicts"])),
                float(np.prod(row["test_verdicts"])),
            )
            slope = 0.0
            if average != 0 or suite != 0:
                slope, error = _projected_gradient(
                    model, [(row, 1.0)], difference, config["token_logp_tolerance"]
                )
                projection_error = max(projection_error, error)
            record = {
                "prompt_id": row["prompt_id"],
                "trajectory_id": row["trajectory_id"],
                "score_projection": slope,
                "score_projection_computed": bool(average != 0 or suite != 0),
                "zero_reward_placeholder": bool(average == 0 and suite == 0),
                "average": average,
                "suite": suite,
            }
            directional.append(record)
            partial.write(
                json.dumps({"artifact_status": "unsealed_diagnostic", **record}, allow_nan=False)
                + "\n"
            )
            partial.flush()
            print(json.dumps({"stage": "D_target_projection", "row": index + 1}), flush=True)
    target_slopes = {}
    for target in ("average", "suite"):
        per_prompt = {
            prompt: float(
                np.mean(
                    [
                        r[target] * r["score_projection"]
                        for r in directional
                        if r["prompt_id"] == prompt
                    ]
                )
            )
            for prompt in sorted({r["prompt_id"] for r in directional})
        }
        target_slopes[target] = {
            "estimate": float(np.mean(list(per_prompt.values()))),
            "per_prompt": per_prompt,
            "uncertainty": "C_fixed_D_prompt_values_not_full_two_bank_CI",
        }
    write_json(
        out / "D_target_prediction.json",
        {
            "targets": target_slopes,
            "rows": directional,
            "partial_artifact": partial_path.name,
            "partial_artifact_status": "unsealed_diagnostic_not_completion_indicator",
        },
    )
del difference
if args.project_only:
    result = {
        "status": "C_D_preliminary_direction_only",
        "functions_sha256": sha256(out / "functions.json"),
        "bank_sha256": dm["bank_sha256"],
        "difference_sha256": sha256(Path(args.difference) / "gradient.safetensors"),
        "max_token_error": projection_error,
        "readout_ids": [feature["prompt_id"] for feature in features],
        "slopes": prediction,
        "independent_slopes": None,
        "steps": [],
        "elapsed_seconds": time.monotonic() - start,
        "shared_reconstruction": "NOT_RUN",
        "independent_reward_consequence": "D_target_direction_projection_only",
        "independent_target_slopes": target_slopes,
        "pilot_decision": "INCONCLUSIVE",
        "local_step_gate": "NOT_PASSED_ON_DEV",
        "definition_hash": digest(features),
        **execution_identity,
    }
    write_json(out / "receipt.json", result)
    print(
        json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"]}),
        flush=True,
    )
    raise SystemExit(0)

by_prompt = {}
for (prompt_id, _), group in groups.items():
    by_prompt.setdefault(prompt_id, []).append(group)
weighted = []
for prompt_groups in by_prompt.values():
    for group in prompt_groups:
        coefficients = group_weights(group, config)["independent"] / (
            len(by_prompt) * len(prompt_groups) * len(group)
        )
        weighted.extend(zip(group, coefficients, strict=True))
model._feedback_progress = True
direction, error = aggregate_gradient(model, weighted, config["token_logp_tolerance"])
independent_slopes = []
for feature in features:
    slope, _ = _projected_gradient(
        model,
        [(r, r["coefficient"]) for r in feature["definitions"]],
        direction,
        config["token_logp_tolerance"],
    )
    independent_slopes.append(slope)
    print(
        json.dumps(
            {
                "stage": "independent_function_slope",
                "prompt_id": feature["prompt_id"],
                "slope": slope,
            }
        ),
        flush=True,
    )
independent_slopes = np.array(independent_slopes)
write_json(
    out / "independent_slopes.json",
    {
        "values": independent_slopes,
        "shared_slopes": independent_slopes + prediction,
        "positive_scale": positive_scale(independent_slopes + prediction, independent_slopes),
    },
)
measurements = {}
for arm in ("independent", "shared"):
    if arm == "shared":
        with safe_open(
            Path(args.difference) / "gradient.safetensors", framework="pt", device="cpu"
        ) as handle:
            for name, tensor in direction.items():
                tensor.add_(handle.get_tensor(name))
    for eta in (args.step, args.step / 2):
        print(json.dumps({"stage": "local_step", "arm": arm, "eta": eta}), flush=True)
        with perturb_from_checkpoint(model, direction, eta, config):
            measurements[f"{arm}:{eta}"] = values()
            if evaluation_rows:
                evaluation_values[f"{arm}:{eta}"] = evaluate(model, evaluation_rows)
        restored = values()
        if not np.array_equal(restored, baseline):
            raise ValueError("Checkpoint restoration changed the baseline functional values")
        write_json(
            out / f"{arm}_{eta}.json",
            {"values": measurements[f"{arm}:{eta}"], "baseline": baseline},
        )
reports = []
for eta in (args.step, args.step / 2):
    actual = measurements[f"shared:{eta}"] - measurements[f"independent:{eta}"]
    predicted = eta * prediction
    reports.append(
        {
            "step": eta,
            "predicted_shared_minus_independent": predicted,
            "actual_shared_minus_independent": actual,
            "residual": actual - predicted,
            "relative_residual": float(
                np.linalg.norm(actual - predicted) / max(np.linalg.norm(predicted), 1e-12)
            ),
        }
    )
result = {
    "status": "Dev_calibration_not_confirmation"
    if splits == {"Dev"}
    else "C_D_preliminary_local_check",
    "functions_sha256": sha256(out / "functions.json"),
    "bank_sha256": dm["bank_sha256"],
    "difference_sha256": sha256(Path(args.difference) / "gradient.safetensors"),
    "max_token_error": error,
    "readout_ids": [f["prompt_id"] for f in features],
    "slopes": prediction,
    "independent_slopes": independent_slopes,
    "functional_positive_scale": positive_scale(
        independent_slopes + prediction, independent_slopes
    ),
    "steps": reports,
    "elapsed_seconds": time.monotonic() - start,
    "shared_reconstruction": "same_bank_gradient_linearity",
    "independent_reward_consequence": "NOT_MEASURED",
    "independent_target_slopes": target_slopes,
    "pilot_decision": "INCONCLUSIVE",
    "definition_hash": digest(features),
    **execution_identity,
}
if evaluation_rows:
    summaries = {}
    for label, new_logps in evaluation_values.items():
        summaries[label] = {}
        for prompt in sorted({r["prompt_id"] for r in evaluation_rows}):
            idx = [i for i, r in enumerate(evaluation_rows) if r["prompt_id"] == prompt]
            summaries[label][prompt] = {
                target: importance_reward(
                    [
                        float(
                            np.mean(evaluation_rows[i]["test_verdicts"])
                            if target == "average"
                            else np.prod(evaluation_rows[i]["test_verdicts"])
                        )
                        for i in idx
                    ],
                    evaluation_base[idx],
                    new_logps[idx],
                )
                for target in ("average", "suite")
            }
    write_json(
        out / "independent_D_importance.json",
        {
            "per_prompt": summaries,
            "evaluation_bank_sha256": sha256(Path(args.evaluation_bank) / "rows.jsonl"),
            "new_generation_after_step": False,
            "max_base_score_gap": float(max_eval_gap),
        },
    )
    result["independent_reward_consequence"] = "D_bank_likelihood_ratio_requires_overlap_review"
write_json(out / "receipt.json", result)
print(
    json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"]}),
    flush=True,
)

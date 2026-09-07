"""Bounded v4 repair diagnostics over frozen banks, directions, and readouts."""

from pathlib import Path

import numpy as np

from dependent_rollouts.artifacts import sha256
from reward_coupling.advantages import advantages
from reward_coupling.bank import digest, group_weights
from reward_coupling.local_steps import (
    _checkpoint_restore_plan,
    _projected_gradient,
    checkpoint_parameter_status,
    perturb_from_checkpoint,
    restore_from_checkpoint,
)
from reward_coupling.sample import token_logps


def analyze_bank_structure(manifest, groups, expected_failure_id=None, tolerance=1e-12):
    """Describe exact S/I credit structure without dropping zero-difference groups."""
    config = manifest["config"]
    summaries = []
    failure_rows = []
    partial_rows = 0
    nonzero_groups = 0
    for (prompt_id, group_id), group in sorted(groups.items()):
        matrix = np.asarray([row["test_verdicts"] for row in group], dtype=np.float64)
        q = matrix.mean(axis=1)
        weights = group_weights(group, config)
        difference = np.asarray(weights["difference"], dtype=np.float64)
        partial = (q > 0) & (q < 1)
        partial_rows += int(partial.sum())
        nonzero = bool(np.max(np.abs(difference)) > tolerance)
        nonzero_groups += int(nonzero)
        if nonzero:
            reason = "nonzero_exact_shared_independent_credit"
        elif np.all(~partial):
            reason = "all_rows_configuration_invariant_q_in_0_1"
        elif np.ptp(matrix, axis=0).max(initial=0) == 0:
            reason = "test_columns_constant_across_group"
        else:
            reason = "exact_shared_independent_expectations_cancel"
        summaries.append(
            {
                "prompt_id": prompt_id,
                "group_id": group_id,
                "rows": len(group),
                "q_by_slot": q.tolist(),
                "partial_correct_slots": np.flatnonzero(partial).astype(int).tolist(),
                "difference_advantage": difference.tolist(),
                "difference_l2": float(np.linalg.norm(difference)),
                "zero_difference": not nonzero,
                "zero_difference_reason": reason,
                "included_in_bank_denominator": True,
            }
        )
        for row, row_q in zip(group, q, strict=True):
            suite = float(np.prod(row["test_verdicts"]))
            if suite == 0:
                failure_rows.append(
                    {
                        "trajectory_id": row["trajectory_id"],
                        "prompt_id": row["prompt_id"],
                        "group_id": row["group_id"],
                        "slot_id": row["slot_id"],
                        "average_reward": float(row_q),
                        "suite_reward": suite,
                        "response_ids_hash": digest(row["response_ids"]),
                        "response_mask_hash": digest(row["active_mask"]),
                    }
                )
    located = [row for row in failure_rows if row["trajectory_id"] == expected_failure_id]
    if expected_failure_id is not None and len(located) != 1:
        raise ValueError(
            f"Expected exactly one frozen failure {expected_failure_id}, found {len(located)}"
        )
    return {
        "status": "CPU_FIXED_BANK_STRUCTURE_ONLY",
        "bank_rows_sha256": manifest.get("bank_sha256"),
        "groups": len(groups),
        "rows": sum(len(group) for group in groups.values()),
        "partial_correct_rows": partial_rows,
        "nonzero_difference_groups": nonzero_groups,
        "zero_difference_groups": len(groups) - nonzero_groups,
        "group_diagnostics": summaries,
        "suite_failure_rows": failure_rows,
        "expected_failure": located[0] if located else None,
        "interpretation": (
            "fixed_bank_exact_credit_structure_not_population_gradient_or_confirmation"
        ),
    }


def direction_identity(direction_path, manifest):
    """Hash direction bytes and the exact named parameter space."""
    from safetensors import safe_open

    path = Path(direction_path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        schema = [
            {
                "name": name,
                "shape": list(handle.get_slice(name).get_shape()),
                "dtype": str(handle.get_slice(name).get_dtype()),
            }
            for name in sorted(handle.keys())
        ]
    config = manifest["config"]
    identity = {
        "direction_hash": sha256(path),
        "parameter_space_hash": digest(schema),
        "parameter_schema": schema,
        "checkpoint_hash": config.get("checkpoint_manifest_sha256"),
        "model_revision": config.get("model_revision"),
        "tokenizer_revision": config.get("tokenizer_revision"),
        "score_reduction": config.get("loss_reduction"),
        "dtype_contract": config.get("dtype"),
    }
    required = (
        "direction_hash",
        "parameter_space_hash",
        "checkpoint_hash",
        "score_reduction",
        "dtype_contract",
    )
    if any(not identity[key] for key in required):
        raise ValueError("Incomplete checkpoint/direction projection identity")
    return identity


def normalize_projection_rows(
    groups,
    legacy_rows,
    identity,
    reward_field="suite",
):
    """Replace legacy zero placeholders with explicit null/not_measured records."""
    if reward_field not in ("average", "suite"):
        raise ValueError("reward_field must be average or suite")
    bank_rows = [row for group in groups.values() for row in group]
    bank_by_id = {row["trajectory_id"]: row for row in bank_rows}
    legacy_by_id = {}
    for row in legacy_rows:
        trajectory_id = row["trajectory_id"]
        if trajectory_id in legacy_by_id:
            raise ValueError(f"Duplicate projection identity: {trajectory_id}")
        legacy_by_id[trajectory_id] = row
    if set(legacy_by_id) != set(bank_by_id):
        missing = sorted(set(bank_by_id) - set(legacy_by_id))
        extra = sorted(set(legacy_by_id) - set(bank_by_id))
        raise ValueError(f"Projection/bank identity mismatch: missing={missing}, extra={extra}")

    normalized = []
    for trajectory_id, bank_row in sorted(
        bank_by_id.items(),
        key=lambda item: (
            item[1]["prompt_id"],
            item[1]["group_id"],
            item[1]["slot_id"],
        ),
    ):
        old = legacy_by_id[trajectory_id]
        if old.get("projection_status") in ("measured", "not_measured"):
            status = old["projection_status"]
            value = old.get("projection_value")
        elif old.get("score_projection_computed") is True:
            status = "measured"
            value = old.get("score_projection")
        else:
            status = "not_measured"
            value = None
        if status == "measured" and (not isinstance(value, (int, float)) or not np.isfinite(value)):
            raise ValueError(f"Invalid measured projection: {trajectory_id}")
        if status == "not_measured" and value is not None:
            raise ValueError(f"Unmeasured projection must be null: {trajectory_id}")
        average = float(np.mean(bank_row["test_verdicts"]))
        suite = float(np.prod(bank_row["test_verdicts"]))
        reward = average if reward_field == "average" else suite
        row_identity = {
            **identity,
            "trajectory_id": trajectory_id,
            "prompt_id": bank_row["prompt_id"],
            "group_id": bank_row["group_id"],
            "slot_id": bank_row["slot_id"],
            "response_ids_hash": digest(bank_row["response_ids"]),
            "response_mask_hash": digest(bank_row["active_mask"]),
        }
        normalized.append(
            {
                **row_identity,
                "projection_identity_hash": digest(row_identity),
                "reward_field": reward_field,
                "reward": reward,
                "average_reward": average,
                "suite_reward": suite,
                "projection_status": status,
                "projection_value": float(value) if status == "measured" else None,
                "legacy_projection_status": (
                    "zero_placeholder_reclassified_as_not_measured"
                    if old.get("zero_reward_placeholder") and status == "not_measured"
                    else "preserved_measured_value"
                ),
            }
        )
    return normalized


def _weighted_projection(records, coefficients):
    missing = {
        row["trajectory_id"]: float(coefficient)
        for row, coefficient in zip(records, coefficients, strict=True)
        if coefficient != 0 and row["projection_status"] != "measured"
    }
    known = sum(
        coefficient * (row["projection_value"] or 0.0)
        for row, coefficient in zip(records, coefficients, strict=True)
        if row["projection_status"] == "measured"
    )
    if missing:
        return {
            "status": "not_measured",
            "value": None,
            "known_measured_contribution": float(known),
            "missing_projection_coefficients": missing,
            "missing_projection_ids": list(missing),
        }
    return {
        "status": "measured",
        "value": float(known),
        "missing_projection_ids": [],
    }


def projection_estimators(records):
    """Return equal-prompt raw, b=1, and RLOO direction projections."""
    prompts = sorted({row["prompt_id"] for row in records})
    if not prompts:
        raise ValueError("Empty projection records")
    grouped = {prompt: [row for row in records if row["prompt_id"] == prompt] for prompt in prompts}
    estimates = {}
    for name in ("old_raw_reward_score", "fixed_baseline_b1", "rloo"):
        flattened = []
        coefficients = []
        for prompt in prompts:
            rows = grouped[prompt]
            rewards = np.asarray([row["reward"] for row in rows], dtype=np.float64)
            if len(rows) < 2:
                raise ValueError("RLOO diagnostics require at least two rows per prompt")
            if name == "old_raw_reward_score":
                weights = rewards
            elif name == "fixed_baseline_b1":
                weights = rewards - 1.0
            else:
                weights = advantages(rewards, kind="rloo")
            weights = weights / (len(prompts) * len(rows))
            flattened.extend(rows)
            coefficients.extend(weights.tolist())
        estimates[name] = _weighted_projection(flattened, coefficients)
    return {
        "status": (
            "complete_centered_repair"
            if all(item["status"] == "measured" for item in estimates.values())
            else "missing_projection_explicit"
        ),
        "prompt_weighting": "equal_prompt_then_equal_response",
        "estimators": estimates,
        "interpretation": "post_hoc_v4_diagnostic_not_new_confirmation_or_pilot_GO",
    }


def replace_single_missing_projection(records, trajectory_id, value):
    if not np.isfinite(value):
        raise ValueError("Projection must be finite")
    matches = [row for row in records if row["trajectory_id"] == trajectory_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one projection row for {trajectory_id}")
    row = matches[0]
    if row["projection_status"] != "not_measured" or row["projection_value"] is not None:
        raise ValueError("Repair target is not explicitly missing")
    missing = [item["trajectory_id"] for item in records if item["projection_status"] != "measured"]
    if missing != [trajectory_id]:
        raise ValueError(f"Repair is bounded to one missing row, found {missing}")
    row["projection_status"] = "measured"
    row["projection_value"] = float(value)
    row["legacy_projection_status"] = "measured_by_bounded_v4_repair"
    return records


def _function_values(model, functions):
    import torch

    values = []
    with torch.no_grad():
        for feature in functions:
            definitions = feature["definitions"]
            if not definitions:
                raise ValueError("Frozen function has no definitions")
            values.append(
                sum(
                    float(row["coefficient"])
                    * float(token_logps(model, row["prompt_ids"], row["response_ids"]).sum())
                    for row in definitions
                )
            )
    return np.asarray(values, dtype=np.float64)


def _function_slopes(model, functions, direction, tolerance):
    slopes = []
    max_error = 0.0
    for feature in functions:
        weighted = [(row, float(row["coefficient"])) for row in feature["definitions"]]
        slope, error = _projected_gradient(model, weighted, direction, tolerance)
        slopes.append(slope)
        max_error = max(max_error, error)
    return np.asarray(slopes, dtype=np.float64), max_error


def _measure_at(model, config, direction, scale, functions, restore_plan):
    restore_from_checkpoint(model, config, restore_plan)
    before = checkpoint_parameter_status(model, config, restore_plan)
    with perturb_from_checkpoint(model, direction, scale, config):
        changed = checkpoint_parameter_status(model, config, restore_plan)
        values = _function_values(model, functions)
    after = checkpoint_parameter_status(model, config, restore_plan)
    return values, {
        "checkpoint_exact_before": before["exact"],
        "checkpoint_exact_after": after["exact"],
        "nonzero_parameter_changes": changed["changed_parameter_elements"],
        "nonzero_parameter_tensors": changed["changed_parameter_tensors"],
        "parameter_elements": changed["parameter_elements"],
    }


def local_finite_difference_diagnostic(
    model,
    config,
    functions,
    difference,
    steps,
    origin_repeats=3,
    shared=None,
    independent=None,
):
    """Measure numerical floor and reset-safe central differences on frozen readouts."""
    steps = [float(step) for step in steps]
    if len(steps) != 2 or any(not np.isfinite(step) or step <= 0 for step in steps):
        raise ValueError("Exactly two positive finite h values are required")
    if len(set(steps)) != 2:
        raise ValueError("The two h values must differ")
    if type(origin_repeats) is not int or origin_repeats < 2:
        raise ValueError("origin_repeats must be at least two")
    if (shared is None) != (independent is None):
        raise ValueError("Real-arm diagnostic requires both shared and independent directions")
    restore_plan = _checkpoint_restore_plan(model, config)
    if restore_plan is None:
        raise ValueError("Exact checkpoint reset is required")

    origins = []
    origin_status = []
    for _ in range(origin_repeats):
        restore_from_checkpoint(model, config, restore_plan)
        status = checkpoint_parameter_status(model, config, restore_plan)
        origins.append(_function_values(model, functions))
        origin_status.append(status)
    origins = np.asarray(origins)
    floor_by_readout = np.ptp(origins, axis=0)
    slopes, max_error = _function_slopes(
        model, functions, difference, config["token_logp_tolerance"]
    )
    restore_from_checkpoint(model, config, restore_plan)

    central = []
    all_reset = all(status["exact"] for status in origin_status)
    for h in steps:
        plus, plus_status = _measure_at(model, config, difference, h / 2, functions, restore_plan)
        minus, minus_status = _measure_at(
            model, config, difference, -h / 2, functions, restore_plan
        )
        estimate = (plus - minus) / h
        remainder = estimate - slopes
        scale = max(float(np.linalg.norm(slopes)), float(np.max(floor_by_readout)), 1e-30)
        central.append(
            {
                "h": h,
                "plus": plus.tolist(),
                "minus": minus.tolist(),
                "central_difference": estimate.tolist(),
                "jacobian_delta": slopes.tolist(),
                "absolute_remainder": np.abs(remainder).tolist(),
                "relative_remainder_l2": float(np.linalg.norm(remainder) / scale),
                "plus_parameter_status": plus_status,
                "minus_parameter_status": minus_status,
            }
        )
        all_reset = all_reset and all(
            (
                plus_status["checkpoint_exact_before"],
                plus_status["checkpoint_exact_after"],
                minus_status["checkpoint_exact_before"],
                minus_status["checkpoint_exact_after"],
            )
        )

    real_arms = []
    if shared is not None:
        for h in steps:
            shared_plus, sp = _measure_at(model, config, shared, h, functions, restore_plan)
            independent_plus, ip = _measure_at(
                model, config, independent, h, functions, restore_plan
            )
            shared_minus, sm = _measure_at(model, config, shared, -h, functions, restore_plan)
            independent_minus, im = _measure_at(
                model, config, independent, -h, functions, restore_plan
            )
            positive = shared_plus - independent_plus
            negative = shared_minus - independent_minus
            real_arms.append(
                {
                    "h": h,
                    "D_positive": positive.tolist(),
                    "D_negative": negative.tolist(),
                    "odd_part": ((positive - negative) / 2).tolist(),
                    "even_part": ((positive + negative) / 2).tolist(),
                    "measurements": {
                        "shared_positive": sp,
                        "independent_positive": ip,
                        "shared_negative": sm,
                        "independent_negative": im,
                    },
                    "scope": "fixed_readout_real_arm_local_diagnostic_not_reward_consequence",
                }
            )
            all_reset = all_reset and all(
                status["checkpoint_exact_before"] and status["checkpoint_exact_after"]
                for status in (sp, ip, sm, im)
            )

    restore_from_checkpoint(model, config, restore_plan)
    final_status = checkpoint_parameter_status(model, config, restore_plan)
    all_reset = all_reset and final_status["exact"]
    if not all_reset:
        raise RuntimeError("Exact checkpoint reset failed")
    return {
        "status": "V4_LOCAL_NUMERICAL_DIAGNOSTIC_ONLY",
        "origin_values": origins.tolist(),
        "repeated_forward_floor": {
            "by_readout": floor_by_readout.tolist(),
            "maximum": float(np.max(floor_by_readout)),
            "repeats": origin_repeats,
        },
        "central_difference": central,
        "real_arm_positive_negative": real_arms,
        "max_token_error": max_error,
        "exact_reset_status": "PASS",
        "final_checkpoint_status": final_status,
        "resampling": False,
        "pilot_decision": "INCONCLUSIVE",
    }

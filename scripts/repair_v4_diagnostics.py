#!/usr/bin/env python
"""Bounded v4 diagnostics over frozen banks; never regenerates answers or declares GO."""

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file

from dependent_rollouts.artifacts import provenance, sha256, write_json, write_jsonl
from reward_coupling.bank import read_bank
from reward_coupling.diagnostics import (
    analyze_bank_structure,
    direction_identity,
    local_finite_difference_diagnostic,
    normalize_projection_rows,
    projection_estimators,
    replace_single_missing_projection,
)
from reward_coupling.local_steps import _gradient_metadata, _projected_gradient
from reward_coupling.sample import load_model


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _fresh_output(path):
    output = Path(path)
    output.mkdir(parents=True, exist_ok=False)
    return output


def _direction_contract(direction_directory, bank_manifest):
    direction_manifest, direction_path = _gradient_metadata(direction_directory)
    for key in (
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "checkpoint_manifest_sha256",
        "dtype",
        "loss_reduction",
    ):
        if direction_manifest["config"].get(key) != bank_manifest["config"].get(key):
            raise ValueError(f"Direction/bank checkpoint contract mismatch: {key}")
    if direction_manifest.get("arm") != "difference":
        raise ValueError("The v4 repair requires the frozen old C difference direction")
    return direction_manifest, direction_path


def analyze_command(args):
    manifest, groups = read_bank(args.bank)
    manifest = {**manifest, "bank_sha256": sha256(Path(args.bank) / "rows.jsonl")}
    report = analyze_bank_structure(
        manifest,
        groups,
        expected_failure_id=args.expected_failure,
        tolerance=args.tolerance,
    )
    report["bank"] = str(Path(args.bank).resolve())
    report["script_source_sha256"] = sha256(__file__)
    output = Path(args.output)
    write_json(output, report)
    print(json.dumps(report, indent=2, allow_nan=False))


def repair_command(args):
    manifest, groups = read_bank(args.bank)
    direction_manifest, direction_path = _direction_contract(args.direction, manifest)
    identity = direction_identity(direction_path, direction_manifest)
    legacy_rows = _jsonl(args.projection_rows)
    records = normalize_projection_rows(groups, legacy_rows, identity, args.reward_field)
    missing = [row["trajectory_id"] for row in records if row["projection_status"] != "measured"]
    if missing != [args.trajectory_id]:
        raise ValueError(
            f"Bounded repair requires only {args.trajectory_id} missing, found {missing}"
        )
    before = projection_estimators(records)
    output = _fresh_output(args.output)
    write_json(
        output / "plan.json",
        {
            "status": "PREPARED_BOUNDED_V4_REPAIR",
            "bank": str(Path(args.bank).resolve()),
            "bank_sha256": sha256(Path(args.bank) / "rows.jsonl"),
            "legacy_projection_rows": str(Path(args.projection_rows).resolve()),
            "legacy_projection_rows_sha256": sha256(args.projection_rows),
            "direction": str(Path(args.direction).resolve()),
            **identity,
            "repair_trajectory_id": args.trajectory_id,
            "reward_field": args.reward_field,
            "resampling": False,
            "prepare_only": args.prepare_only,
            "before_repair": before,
        },
    )
    if args.prepare_only:
        write_jsonl(output / "projection_rows.jsonl", records)
        receipt = {
            "status": "PREPARED_MISSING_PROJECTION_EXPLICIT_NOT_MEASURED",
            "projection_rows_sha256": sha256(output / "projection_rows.jsonl"),
            "repair_trajectory_id": args.trajectory_id,
            "projection_status": "not_measured",
            "projection_value": None,
            "estimators": before,
            "pilot_decision": "INCONCLUSIVE",
        }
        write_json(output / "receipt.json", receipt)
        print(json.dumps(receipt, indent=2, allow_nan=False))
        return

    target = next(
        row
        for group in groups.values()
        for row in group
        if row["trajectory_id"] == args.trajectory_id
    )
    model, _ = load_model(manifest["config"])
    model._feedback_offload_activations = manifest["config"].get("offload_gradients", False)
    direction = load_file(str(direction_path), device="cpu")
    projection, max_error = _projected_gradient(
        model,
        [(target, 1.0)],
        direction,
        manifest["config"]["token_logp_tolerance"],
    )
    del direction
    replace_single_missing_projection(records, args.trajectory_id, projection)
    after = projection_estimators(records)
    write_jsonl(output / "projection_rows.jsonl", records)
    receipt = {
        "status": "COMPLETE_BOUNDED_V4_MISSING_PROJECTION_REPAIR",
        "projection_rows_sha256": sha256(output / "projection_rows.jsonl"),
        "repair_trajectory_id": args.trajectory_id,
        "projection_status": "measured",
        "projection_value": projection,
        "max_token_error": max_error,
        "before_repair": before,
        "after_repair": after,
        "identity": identity,
        "resampling": False,
        "scope": "post_hoc_old_v4_diagnostic_not_new_confirmation",
        "pilot_decision": "INCONCLUSIVE",
        "provenance": provenance(manifest["config"]),
    }
    write_json(output / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2, allow_nan=False))


def local_command(args):
    config = json.loads(Path(args.config).read_text())
    functions = json.loads(Path(args.functions).read_text())
    difference_manifest, difference_path = _gradient_metadata(args.difference)
    if difference_manifest["config"] != config:
        raise ValueError("Difference direction and local diagnostic config differ")
    if difference_manifest.get("arm") != "difference":
        raise ValueError("Local diagnostic requires the S-I difference direction")
    difference = load_file(str(difference_path), device="cpu")
    shared = independent = None
    direction_hashes = {
        "difference": direction_identity(difference_path, difference_manifest),
    }
    if args.real_arms:
        if not args.shared or not args.independent:
            raise ValueError("--real-arms requires --shared and --independent")
        shared_manifest, shared_path = _gradient_metadata(args.shared)
        independent_manifest, independent_path = _gradient_metadata(args.independent)
        if shared_manifest["config"] != config or independent_manifest["config"] != config:
            raise ValueError("Real-arm direction checkpoint contract mismatch")
        if (
            shared_manifest.get("arm") != "shared"
            or independent_manifest.get("arm") != "independent"
        ):
            raise ValueError("Real-arm diagnostics require shared and independent directions")
        if shared_manifest.get("bank_sha256") != independent_manifest.get("bank_sha256"):
            raise ValueError("Real-arm directions must come from the same frozen bank")
        shared = load_file(str(shared_path), device="cpu")
        independent = load_file(str(independent_path), device="cpu")
        direction_hashes["shared"] = direction_identity(shared_path, shared_manifest)
        direction_hashes["independent"] = direction_identity(independent_path, independent_manifest)
    output = _fresh_output(args.output)
    write_json(
        output / "plan.json",
        {
            "status": "PREPARED_V4_LOCAL_DIAGNOSTIC",
            "config": config,
            "config_sha256": sha256(args.config),
            "functions": str(Path(args.functions).resolve()),
            "functions_sha256": sha256(args.functions),
            "steps": args.h,
            "origin_repeats": args.origin_repeats,
            "real_arms": args.real_arms,
            "resampling": False,
            "directions": direction_hashes,
        },
    )
    model, _ = load_model(config)
    model._feedback_offload_activations = config.get("offload_gradients", False)
    report = local_finite_difference_diagnostic(
        model,
        config,
        functions,
        difference,
        args.h,
        origin_repeats=args.origin_repeats,
        shared=shared,
        independent=independent,
    )
    report.update(
        {
            "config_sha256": sha256(args.config),
            "functions_sha256": sha256(args.functions),
            "directions": direction_hashes,
            "scope": "fixed_readout_numerical_diagnostic_not_reward_or_training",
        }
    )
    write_json(output / "receipt.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    command = commands.add_parser("analyze-bank")
    command.add_argument("--bank", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--expected-failure")
    command.add_argument("--tolerance", type=float, default=1e-12)
    command.set_defaults(run=analyze_command)

    command = commands.add_parser("repair-projection")
    command.add_argument("--bank", required=True)
    command.add_argument("--direction", required=True)
    command.add_argument("--projection-rows", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--trajectory-id", default="Mbpp/474:0:7")
    command.add_argument("--reward-field", choices=["average", "suite"], default="suite")
    command.add_argument("--prepare-only", action="store_true")
    command.set_defaults(run=repair_command)

    command = commands.add_parser("local-diagnostic")
    command.add_argument("--config", required=True)
    command.add_argument("--difference", required=True)
    command.add_argument("--functions", required=True)
    command.add_argument("--h", type=float, nargs=2, required=True)
    command.add_argument("--origin-repeats", type=int, default=3)
    command.add_argument("--real-arms", action="store_true")
    command.add_argument("--shared")
    command.add_argument("--independent")
    command.add_argument("--output", required=True)
    command.set_defaults(run=local_command)
    return root


def main():
    args = parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()

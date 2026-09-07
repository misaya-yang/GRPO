"""CPU tests for the bounded v4 diagnostic repair."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from dependent_rollouts.artifacts import sha256, write_json
from reward_coupling.bank import digest
from reward_coupling.diagnostics import (
    analyze_bank_structure,
    local_finite_difference_diagnostic,
    normalize_projection_rows,
    projection_estimators,
    replace_single_missing_projection,
)
from reward_coupling.local_steps import checkpoint_parameter_status
from reward_coupling.sample import token_logps


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(41)
        self.embedding = torch.nn.Embedding(12, 6)
        self.output = torch.nn.Linear(6, 12, bias=False)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


def _row(prompt, group, slot, verdicts, response=None):
    response = response or [3, 4]
    return {
        "prompt_id": prompt,
        "group_id": group,
        "slot_id": slot,
        "trajectory_id": f"{prompt}:{group}:{slot}",
        "prompt_ids": [1, 2],
        "prompt_hash": digest([1, 2]),
        "response_ids": response,
        "active_mask": [1] * len(response),
        "sampling_token_logp": [-1.0] * len(response),
        "test_verdicts": verdicts,
        "test_manifest_hash": "fixture",
        "split": "D",
        "group_seed": group + 10,
        "actor_rng_stream": slot + 100,
    }


def test_fixed_bank_structure_keeps_zero_groups_and_locates_failure():
    config = {"feedback": "code", "epsilon": 1e-6}
    invariant = [_row("constant", 0, slot, [slot % 2] * 4) for slot in range(4)]
    mixed_verdicts = [
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
    ]
    mixed = [_row("Mbpp/474", 0, slot, verdicts) for slot, verdicts in enumerate(mixed_verdicts)]
    groups = {("constant", 0): invariant, ("Mbpp/474", 0): mixed}
    report = analyze_bank_structure(
        {"config": config, "bank_sha256": "bank"},
        groups,
        expected_failure_id="Mbpp/474:0:3",
    )
    assert report["groups"] == 2
    assert report["rows"] == 8
    assert report["expected_failure"]["trajectory_id"] == "Mbpp/474:0:3"
    constant = next(item for item in report["group_diagnostics"] if item["prompt_id"] == "constant")
    assert constant["zero_difference"]
    assert constant["zero_difference_reason"] == "all_rows_configuration_invariant_q_in_0_1"
    assert constant["included_in_bank_denominator"] is True
    assert report["nonzero_difference_groups"] == 1


def test_missing_projection_is_null_and_repairs_raw_b1_rloo():
    groups = {
        ("a", 0): [
            _row("a", 0, 0, [1, 1]),
            _row("a", 0, 1, [1, 1]),
        ],
        ("b", 0): [
            _row("b", 0, 0, [1, 1]),
            _row("b", 0, 1, [0, 0]),
        ],
    }
    legacy = [
        {
            "trajectory_id": "a:0:0",
            "score_projection": 2.0,
            "score_projection_computed": True,
        },
        {
            "trajectory_id": "a:0:1",
            "score_projection": -2.0,
            "score_projection_computed": True,
        },
        {
            "trajectory_id": "b:0:0",
            "score_projection": 4.0,
            "score_projection_computed": True,
        },
        {
            "trajectory_id": "b:0:1",
            "score_projection": 0.0,
            "score_projection_computed": False,
            "zero_reward_placeholder": True,
        },
    ]
    identity = {
        "checkpoint_hash": "checkpoint",
        "parameter_space_hash": "parameters",
        "direction_hash": "direction",
        "score_reduction": "sum_tokens_then_mean_responses",
        "dtype_contract": "float32",
    }
    records = normalize_projection_rows(groups, legacy, identity, reward_field="suite")
    missing = next(row for row in records if row["trajectory_id"] == "b:0:1")
    assert missing["projection_status"] == "not_measured"
    assert missing["projection_value"] is None
    assert len(missing["projection_identity_hash"]) == 64

    before = projection_estimators(records)
    assert before["estimators"]["old_raw_reward_score"] == {
        "status": "measured",
        "value": 1.0,
        "missing_projection_ids": [],
    }
    assert before["estimators"]["fixed_baseline_b1"]["value"] is None
    assert before["estimators"]["rloo"]["value"] is None

    replace_single_missing_projection(records, "b:0:1", -6.0)
    after = projection_estimators(records)
    assert after["status"] == "complete_centered_repair"
    assert after["estimators"]["old_raw_reward_score"]["value"] == pytest.approx(1.0)
    assert after["estimators"]["fixed_baseline_b1"]["value"] == pytest.approx(1.5)
    assert after["estimators"]["rloo"]["value"] == pytest.approx(2.5)


def _checkpoint_fixture(tmp_path):
    model = TinyCausalLM().float().eval().requires_grad_(True)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    checkpoint_file = checkpoint / "model.safetensors"
    save_file(
        {name: parameter.detach().clone() for name, parameter in model.named_parameters()},
        checkpoint_file,
    )
    manifest = tmp_path / "checkpoint.json"
    write_json(manifest, {"files": {checkpoint_file.name: sha256(checkpoint_file)}})
    config = {
        "model_path": str(checkpoint),
        "checkpoint_manifest": str(manifest),
        "checkpoint_manifest_sha256": sha256(manifest),
        "token_logp_tolerance": 1e-9,
    }
    return model, config


def test_local_diagnostic_repeats_origin_central_differences_and_resets(tmp_path):
    model, config = _checkpoint_fixture(tmp_path)
    definitions = []
    for response, coefficient in (([3, 4], 1.0), ([5, 6], -1.0)):
        scores = token_logps(model, [1, 2], response).detach().tolist()
        definitions.append(
            {
                "prompt_ids": [1, 2],
                "response_ids": response,
                "sampling_token_logp": scores,
                "coefficient": coefficient,
                "split": "Dev",
            }
        )
    functions = [{"prompt_id": "readout", "definitions": definitions}]
    torch.manual_seed(43)
    difference = {
        name: torch.randn_like(parameter).cpu() for name, parameter in model.named_parameters()
    }
    shared = {name: value * 0.75 for name, value in difference.items()}
    independent = {name: value * -0.25 for name, value in difference.items()}

    report = local_finite_difference_diagnostic(
        model,
        config,
        functions,
        difference,
        [1e-3, 5e-4],
        origin_repeats=3,
        shared=shared,
        independent=independent,
    )
    assert report["status"] == "V4_LOCAL_NUMERICAL_DIAGNOSTIC_ONLY"
    assert report["repeated_forward_floor"]["maximum"] == 0.0
    assert report["exact_reset_status"] == "PASS"
    assert report["resampling"] is False
    assert len(report["central_difference"]) == 2
    assert len(report["real_arm_positive_negative"]) == 2
    for measurement in report["central_difference"]:
        assert measurement["plus_parameter_status"]["nonzero_parameter_changes"] > 0
        assert measurement["minus_parameter_status"]["nonzero_parameter_changes"] > 0
        assert measurement["plus_parameter_status"]["checkpoint_exact_after"]
        assert measurement["minus_parameter_status"]["checkpoint_exact_after"]
        assert np.isfinite(measurement["relative_remainder_l2"])
    assert checkpoint_parameter_status(model, config)["exact"]

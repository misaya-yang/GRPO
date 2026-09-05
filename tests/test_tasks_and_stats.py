from fractions import Fraction

import numpy as np
import pytest

from dependent_rollouts.artifacts import sha256, write_json, write_jsonl
from dependent_rollouts.llm import read_bank
from dependent_rollouts.statistics import (
    bootstrap_contrast,
    functional_distortion,
    importance_reward,
)
from dependent_rollouts.tasks import generate_arithmetic, rational_expression, verify


def test_read_bank_rejects_subsampled_groups(tmp_path):
    # Dossier Section 11.2: a K=8 group subsampled to fewer elements must never
    # be treated as a native smaller-K law.
    write_json(tmp_path / "manifest.json", {"config": {"k": 4, "sampler": "iid"}})
    rows = [
        {"trajectory_id": f"p:0:{m}", "prompt_id": "p", "group_id": 0, "member": m, "reward": m % 2}
        for m in range(3)
    ]
    write_jsonl(tmp_path / "rollouts.jsonl", rows)
    write_json(
        tmp_path / "receipt.json",
        {"status": "complete", "rollouts_sha256": sha256(tmp_path / "rollouts.jsonl")},
    )
    with pytest.raises(ValueError, match="subsample"):
        read_bank(tmp_path)


def test_rational_ast_and_final_answer_contract():
    assert rational_expression("(1+2)/3", [1, 2, 3]) == Fraction(1)
    for bad in ("__import__('os').system('id')", "2**100000000", "[1][0]", "True", "1/0"):
        with pytest.raises((ValueError, ZeroDivisionError)):
            rational_expression(bad)
    task = {"task_type": "numeric", "target": "42"}
    assert verify("reason 12\n<answer>42</answer>", task, "eos")["reward"] == 1
    for text in (
        "42",
        "<answer>42</answer> trailing",
        "<answer>42</answer><answer>42</answer>",
        "<answer>142</answer>",
    ):
        assert verify(text, task, "eos")["reward"] == 0
    assert verify("<answer>42</answer>", task, "length")["reward"] == 0


def test_generated_split_identity():
    a = generate_arithmetic(12, "discovery", 2027)
    b = generate_arithmetic(12, "confirmation", 2028)
    assert len({r["prompt_id"] for r in a + b}) == 24
    assert {r["template_id"] for r in a}.isdisjoint({r["template_id"] for r in b})


def test_functional_and_unclipped_importance():
    x = np.array([1.0, 2.0, 3.0])
    assert functional_distortion(2 * x, x, x)["delta"] == 0
    assert functional_distortion(x, x, x * 0)["status"] == "unresolved_reference"
    result = importance_reward([0, 1], [0, 0], np.log([0.5, 2.0]))
    assert result["reward"] == 1.0  # not self-normalized: would be .8
    assert result["max_weight"] == 2.0


def test_group_bootstrap_recomputes_pool():
    r = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
    s = np.arange(16).reshape(4, 2, 2)
    result = bootstrap_contrast(r, s, replicates=50)
    assert result["unit"] == "original_group"
    assert np.all(result["lower"] <= result["upper"])

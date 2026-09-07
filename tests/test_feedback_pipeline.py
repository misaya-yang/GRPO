import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dependent_rollouts.artifacts import sha256, write_json, write_jsonl
from reward_coupling.bank import digest, read_bank, seal
from reward_coupling.feedback import integrate, score
from reward_coupling.sample import encode_prompt
from reward_coupling.training import train


def _write_candidate_bank(path, tasks_path, mutate=None, feedback="code"):
    config = {
        "feedback": feedback,
        "rubric_keep": 1,
        "group_size": 2,
        "epsilon": 1e-6,
        "token_logp_tolerance": 1e-5,
        "test_timeout_seconds": 1.0,
        "executor": "native",
        "test_python": sys.executable,
        "tasks_sha256": sha256(tasks_path),
    }
    manifest = {"actor_sampling": "iid", "split": "C", "eos_ids": [9], "config": config}
    rows = [
        {
            "trajectory_id": "p:0:0",
            "prompt_id": "p",
            "prompt_ids": [1],
            "prompt_hash": digest([1]),
            "response_ids": [2, 9],
            "response_length": 2,
            "sampling_token_logp": [-1.0, -2.0],
            "scoring_token_logp": [-1.0, -2.0],
            "old_logp": -3.0,
            "active_mask": [1, 1],
            "eos_index_or_null": 1,
            "truncated": False,
            "actor_rng_stream": 11,
            "group_seed": 7,
            "group_id": 0,
            "slot_id": 0,
            "split": "C",
            "text": "def f(): return 1",
        },
        {
            "trajectory_id": "p:0:1",
            "prompt_id": "p",
            "prompt_ids": [1],
            "prompt_hash": digest([1]),
            "response_ids": [3],
            "response_length": 1,
            "sampling_token_logp": [-1.0],
            "scoring_token_logp": [-1.0],
            "old_logp": -1.0,
            "active_mask": [1],
            "eos_index_or_null": None,
            "truncated": True,
            "actor_rng_stream": 12,
            "group_seed": 7,
            "group_id": 0,
            "slot_id": 1,
            "split": "C",
            "text": "def f(): return 0",
        },
    ]
    if mutate is not None:
        mutate(rows)
    path.mkdir()
    write_json(path / "manifest.json", manifest)
    write_jsonl(path / "rows.jsonl", rows)
    seal(path, "rows.jsonl")


def _write_tasks(path):
    write_jsonl(
        path,
        [
            {
                "prompt_id": "p",
                "prompt": "Implement f",
                "reference_code": "def f(): return 1",
                "tests": ["assert f() == 1"],
                "split": "C",
            }
        ],
    )


@pytest.mark.parametrize("encoded", [[1, 2, 3], {"input_ids": [[1, 2, 3]]}])
def test_encode_prompt_accepts_list_and_mapping_chat_templates(encoded):
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return encoded

    assert encode_prompt(Tokenizer(), "prompt") == [1, 2, 3]


def test_native_score_and_integrate_preserve_group_weights_and_identity(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    _write_tasks(tasks)
    candidates = tmp_path / "candidates"
    _write_candidate_bank(candidates, tasks)

    scored = tmp_path / "scored"
    score(candidates, tasks, scored)
    _, groups = read_bank(scored)
    group = groups[("p", 0)]
    assert [row["test_verdicts"] for row in group] == [[1], [0]]

    weights = tmp_path / "weights"
    integrate(scored, weights)
    rows = [json.loads(line) for line in (weights / "weights.jsonl").read_text().splitlines()]
    assert [row["trajectory_id"] for row in rows] == ["p:0:0", "p:0:1"]
    assert [row["context_probabilities"] for row in rows] == [[1.0], [1.0]]
    assert rows[0]["shared"] == pytest.approx(-rows[1]["shared"])
    assert rows[0]["independent"] == pytest.approx(rows[0]["shared"])


def test_cached_rubric_bank_works_without_code_tests(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    write_jsonl(
        tasks,
        [
            {
                "prompt_id": "p",
                "prompt": "Answer accurately and briefly",
                "criteria": ["accurate", "brief"],
                "split": "C",
            }
        ],
    )
    candidates = tmp_path / "candidates"
    _write_candidate_bank(candidates, tasks, feedback="rubric")
    verdicts = tmp_path / "judge.json"
    write_json(
        verdicts,
        {
            "judge_id": "fixture",
            "judge_revision": "fixture-v1",
            "prompt_template_hash": digest("fixture"),
            "criterion_manifest_hash": digest(["accurate", "brief"]),
            "private_randomness": "fixed_fixture",
            "response_local": True,
            "rows": [
                {
                    "trajectory_id": "p:0:0",
                    "response_hash": digest([2, 9]),
                    "criterion_verdicts": [1, 0],
                },
                {
                    "trajectory_id": "p:0:1",
                    "response_hash": digest([3]),
                    "criterion_verdicts": [0, 1],
                },
            ],
        },
    )
    scored = tmp_path / "scored"
    score(candidates, tasks, scored, verdicts)
    _, groups = read_bank(scored)
    assert groups[("p", 0)][0]["test_verdicts"] == [1, 0]
    integrate(scored, tmp_path / "weights")


def test_dev_resume_keeps_candidates_and_rejects_actor_changes(tmp_path):
    from reward_coupling.sample import resume_candidates

    parent = tmp_path / "partial"
    parent.mkdir()
    previous = {"seed": 123, "token_logp_tolerance": 1e-5}
    write_json(parent / "manifest.json", {"config": previous, "split": "Dev"})
    write_jsonl(parent / "rows.jsonl", [{"actor_rng_stream": 1, "response_ids": [3, 4]}])
    write_json(parent / "score_point_failure.json", {"actor_rng_stream": 2, "response_ids": [5]})
    config = {
        **previous,
        "resume_from": str(parent),
        "token_logp_tolerance": 2e-4,
        "resume_artifact_sha256": {p.name: sha256(p) for p in parent.iterdir()},
    }
    assert resume_candidates(config, "Dev")[2]["response_ids"] == [5]
    with pytest.raises(ValueError, match="actor/data"):
        resume_candidates({**config, "seed": 321}, "Dev")
    with pytest.raises(ValueError, match="Dev operation"):
        resume_candidates(config, "C")
    with pytest.raises(ValueError, match="artifact hashes"):
        resume_candidates({**config, "resume_artifact_sha256": {}}, "Dev")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows[0].update(eos_index_or_null=None),
        lambda rows: rows[0].update(truncated=True),
        lambda rows: rows[0].update(response_length=1),
        lambda rows: rows[0]["scoring_token_logp"].__setitem__(0, -1.1),
        lambda rows: rows[1].update(actor_rng_stream=11),
    ],
)
def test_read_bank_rejects_invalid_response_and_rng_identity(tmp_path, mutate):
    tasks = tmp_path / "tasks.jsonl"
    _write_tasks(tasks)
    candidates = tmp_path / "candidates"
    _write_candidate_bank(candidates, tasks, mutate)
    with pytest.raises(ValueError):
        read_bank(candidates)


@pytest.mark.parametrize("key,value", [("checkpoint_every", 0), ("prompts_per_update", "1")])
def test_training_rejects_invalid_integer_settings_before_executor_or_model(tmp_path, key, value):
    tasks = tmp_path / "tasks.jsonl"
    write_jsonl(
        tasks,
        [
            {
                "prompt_id": "train-p",
                "prompt": "Implement f",
                "reference_code": "def f(): return 1",
                "tests": ["assert f() == 1"],
                "split": "train",
            }
        ],
    )
    evidence = tmp_path / "evidence.json"
    write_json(evidence, {"status": "PASS"})
    item = {"path": str(evidence), "sha256": sha256(evidence)}
    decision = tmp_path / "decision.json"
    write_json(
        decision,
        {
            "gates": {
                name: {"status": "PASS", "evidence": [item]}
                for name in ("implementation", "credit", "non_scalar", "local_steps")
            }
        },
    )
    config = json.loads(Path("configs/feedback/pilot_code.json").read_text())
    config.update(
        {
            "model_revision": "a" * 40,
            "tokenizer_revision": "b" * 40,
            "tasks_sha256": sha256(tasks),
            "decision_sha256": sha256(decision),
            "optimizer": "sgd",
            "learning_rate": 1e-4,
            "updates": 1,
            "prompts_per_update": 1,
            "checkpoint_every": 1,
            "coupling": "shared",
            "advantage": "standardized",
        }
    )
    config[key] = value
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    with pytest.raises(ValueError, match=key):
        train(config_path, tasks, decision, tmp_path / "output")


def test_prepare_mbpp_feedback_freezes_requested_test_count_and_splits(tmp_path):
    source = tmp_path / "mbpp.jsonl"
    decoder = tmp_path / "mbpp.py"
    output = tmp_path / "tasks.jsonl"
    decoder.write_text(
        "def mbpp_deserialize_inputs(task_id, inputs):\n"
        "    return [tuple(value) for value in inputs]\n"
    )
    write_jsonl(
        source,
        [
            {
                "task_id": f"Mbpp/{index}",
                "entry_point": f"f_{index}",
                "prompt": "Return the input.",
                "canonical_solution": f"def f_{index}(x): return x",
                "base_input": [[0], [1]],
                "plus_input": [[2]],
                "atol": 0,
            }
            for index in range(1000, 1112)
        ],
    )
    env = {**os.environ, "PYTHONPATH": "src"}
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_mbpp_feedback.py",
            "--source",
            str(source),
            "--decoder-source",
            str(decoder),
            "--output",
            str(output),
            "--tests-per-task",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 112
    assert {row["source"] for row in rows} == {"MbppPlus-v0.2.0_hash_selected_2_test_subset"}
    assert {
        split: sum(row["split"] == split for row in rows)
        for split in ("Dev", "C", "D", "evaluation")
    } == {
        "Dev": 16,
        "C": 32,
        "D": 32,
        "evaluation": 32,
    }
    assert all(len(row["tests"]) == 2 for row in rows)

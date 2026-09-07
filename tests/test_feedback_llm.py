"""Random tiny CPU model: new mainline score, weights, and branch plumbing."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import Qwen2Config, Qwen2ForCausalLM

from dependent_rollouts.artifacts import sha256, write_json
from dependent_rollouts.interventions import perturb
from reward_coupling.expectation import expected_advantages
from reward_coupling.gradient_audit import aggregate_gradient, dot
from reward_coupling.judge import LocalRubricJudge
from reward_coupling.local_steps import _projected_gradient, perturb_from_checkpoint
from reward_coupling.sample import generate, token_logps


@pytest.fixture
def tiny():
    torch.manual_seed(731)
    torch.set_num_threads(1)
    config = Qwen2Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        eos_token_id=15,
        max_position_embeddings=128,
        attention_dropout=0,
    )
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).cpu().eval()


def test_native_ancestral_score_eos_truncation_and_determinism(tiny):
    for seed in range(4):
        a = generate(tiny, [1, 2], [15], seed, 8)
        b = generate(tiny, [1, 2], [15], seed, 8)
        assert a == b
        assert a["max_token_logp_error"] < 2e-6
        assert a["truncated"] == (a["response_ids"][-1] != 15)
        assert a["active_mask"] == [1] * len(a["response_ids"])
    # EOS must be emitted, not fabricated on length cap.
    a = generate(tiny, [1], list(range(16)), 4, 8)
    assert a["response_length"] == 1 and a["eos_index_or_null"] == 0
    a = generate(tiny, [1], [99], 4, 1)
    assert a["truncated"] and a["eos_index_or_null"] is None


def test_kv_cache_keeps_iid_tokens_and_teacher_score_contract(tiny):
    for seed in range(4):
        plain = generate(tiny, [1, 2], [15], seed, 8)
        cached = generate(tiny, [1, 2], [15], seed, 8, use_cache=True)
        assert cached["response_ids"] == plain["response_ids"]
        assert cached["generation_cache"] is True
        assert cached["max_token_logp_error"] < 2e-5


def test_exact_expected_advantage_gradient_linearity_and_steps(tiny):
    matrix = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]]
    expected = expected_advantages(matrix)
    rows = [dict(generate(tiny, [1, 2], [15], s, 5), prompt_ids=[1, 2]) for s in range(4)]
    gradients = {
        arm: aggregate_gradient(tiny, list(zip(rows, expected[arm] / 4, strict=True)))[0]
        for arm in ("shared", "independent", "difference")
    }
    for name in gradients["shared"]:
        torch.testing.assert_close(
            gradients["shared"][name] - gradients["independent"][name],
            gradients["difference"][name],
        )
    g = gradients["difference"]
    expected_slope = dot(g, g)
    baseline = {n: p.detach().clone() for n, p in tiny.named_parameters()}

    def objective():
        return sum(
            float(w) * float(token_logps(tiny, r["prompt_ids"], r["response_ids"]).sum().detach())
            for r, w in zip(rows, expected["difference"] / 4, strict=True)
        )

    base = objective()
    errors = []
    for eta in (0.01, 0.005):
        with perturb(tiny, g, eta):
            change = objective() - base
        errors.append(abs(change - eta * expected_slope))
    assert errors[1] < errors[0] * 0.8 + 1e-8
    for n, p in tiny.named_parameters():
        assert torch.equal(p, baseline[n])
    rows[0]["sampling_token_logp"][0] += 0.1
    with pytest.raises(ValueError, match="score point"):
        aggregate_gradient(tiny, [(rows[0], 1)])


def test_reward_law_gradient_matches_context_enumeration(tiny):
    matrix = np.array([[1, 0], [0, 1]])
    rows = [dict(generate(tiny, [1], [15], s, 3), prompt_ids=[1]) for s in range(2)]
    from reward_coupling.advantages import advantages

    expected = expected_advantages(matrix)
    direct, _ = aggregate_gradient(tiny, list(zip(rows, expected["shared"] / 2, strict=True)))
    summed = {n: torch.zeros_like(g) for n, g in direct.items()}
    for column in matrix.T:
        part, _ = aggregate_gradient(tiny, list(zip(rows, advantages(column) / 4, strict=True)))
        for n in summed:
            summed[n] += part[n]
    for n in summed:
        torch.testing.assert_close(summed[n], direct[n])


def test_zero_credit_score_checks_do_not_retain_backward_graphs(tiny, monkeypatch):
    import reward_coupling.gradient_audit as audit

    row = dict(generate(tiny, [1], [15], 2, 3), prompt_ids=[1])
    original = audit.token_logps
    grad_modes = []

    def record(*args):
        grad_modes.append(torch.is_grad_enabled())
        return original(*args)

    monkeypatch.setattr(audit, "token_logps", record)
    gradient, _ = audit.aggregate_gradient(tiny, [(row, 0), (row, 0)])
    assert grad_modes == [False, False]
    assert all(torch.count_nonzero(g) == 0 for g in gradient.values())


def test_single_aggregate_audit_keeps_equal_prompt_weight(tiny, monkeypatch, tmp_path):
    import reward_coupling.gradient_audit as audit
    from dependent_rollouts.artifacts import write_jsonl

    bank = tmp_path / "bank"
    bank.mkdir()
    write_jsonl(bank / "rows.jsonl", [{"fixture": True}])
    groups, weighted = {}, []
    for prompt, count in (("a", 1), ("b", 2)):
        for group_id in range(count):
            group = []
            for slot in range(2):
                row = dict(
                    generate(tiny, [1], [15], len(weighted) + 50, 3),
                    prompt_ids=[1],
                    prompt_id=prompt,
                    trajectory_id=f"{prompt}:{group_id}:{slot}",
                    test_verdicts=[slot, slot],
                    test_manifest_hash="fixture",
                )
                group.append(row)
                weighted.append((row, slot / (2 * count * 2)))
            groups[(prompt, group_id)] = group
    config = {"feedback": "code", "epsilon": 1e-6, "token_logp_tolerance": 1e-5}
    monkeypatch.setattr(audit, "read_bank", lambda _: ({"config": config}, groups))
    monkeypatch.setattr(audit, "load_model", lambda _: (tiny, None))
    expected, _ = aggregate_gradient(tiny, weighted)
    output = tmp_path / "audit"
    audit.audit(bank, output, "average")
    _, actual = audit.load_gradient(output)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name])


def test_dev_calibration_script_restores_tiny_checkpoint(tiny, monkeypatch, tmp_path):
    import json
    import runpy
    import sys

    from safetensors.torch import save_file

    import reward_coupling.sample as sample
    from dependent_rollouts.artifacts import sha256, write_json, write_jsonl
    from reward_coupling.bank import digest, group_weights, seal

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file(tiny.state_dict(), str(checkpoint / "model.safetensors"))
    pinned = tmp_path / "checkpoint.json"
    write_json(pinned, {"files": {"model.safetensors": sha256(checkpoint / "model.safetensors")}})
    config = {
        "group_size": 4,
        "feedback": "code",
        "epsilon": 1e-6,
        "token_logp_tolerance": 1e-5,
        "model_path": str(checkpoint),
        "checkpoint_manifest": str(pinned),
        "checkpoint_manifest_sha256": sha256(pinned),
    }
    rows, weighted = [], []
    b = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]]
    for prompt in ("a", "b"):
        group = []
        for slot in range(4):
            row = dict(
                generate(tiny, [1, 2], [15], len(rows) + 70, 3),
                prompt_ids=[1, 2],
                prompt_hash=digest([1, 2]),
                prompt_id=prompt,
                split="Dev",
                group_id=0,
                group_seed=1 if prompt == "a" else 2,
                slot_id=slot,
                trajectory_id=f"{prompt}:0:{slot}",
                test_verdicts=b[slot],
                test_manifest_hash="fixture",
            )
            group.append(row)
            rows.append(row)
        weighted.extend(zip(group, group_weights(group, config)["difference"] / 8, strict=True))
    bank = tmp_path / "bank"
    bank.mkdir()
    write_json(
        bank / "manifest.json",
        {"actor_sampling": "iid", "split": "Dev", "eos_ids": [15], "config": config},
    )
    write_jsonl(bank / "rows.jsonl", rows)
    seal(bank, "rows.jsonl")
    difference = tmp_path / "difference"
    difference.mkdir()
    gradient, _ = aggregate_gradient(tiny, weighted)
    save_file(gradient, str(difference / "gradient.safetensors"))
    write_json(
        difference / "manifest.json",
        {"config": config, "arm": "difference", "bank_sha256": sha256(bank / "rows.jsonl")},
    )
    seal(difference, "gradient.safetensors")
    tasks = tmp_path / "tasks.jsonl"
    write_jsonl(
        tasks,
        [
            {
                "prompt_id": p,
                "prompt": p,
                "split": "Dev",
                "reference_code": "def f(): return 1",
                "entry_point": "f",
                "tests": ["assert f()==1"],
            }
            for p in ("a", "b")
        ],
    )

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]

        def encode(self, text, **kwargs):
            return [8] if "None" in text else [7]

    monkeypatch.setattr(sample, "load_model", lambda _: (tiny, Tokenizer()))
    output = tmp_path / "result"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dev_step_calibration.py",
            "--bank",
            str(bank),
            "--difference",
            str(difference),
            "--tasks",
            str(tasks),
            "--output",
            str(output),
        ],
    )
    before = {n: p.detach().clone() for n, p in tiny.named_parameters()}
    runpy.run_path("scripts/dev_step_calibration.py", run_name="__main__")
    assert (
        json.loads((output / "receipt.json").read_text())["status"]
        == "Dev_calibration_not_confirmation"
    )
    for name, parameter in tiny.named_parameters():
        assert torch.equal(parameter, before[name])


@pytest.mark.parametrize("project_only", [False, True])
def test_c_d_calibration_uses_frozen_dev_functions_and_independent_bank(
    tiny, monkeypatch, tmp_path, project_only
):
    import json
    import runpy
    import sys

    from safetensors.torch import save_file

    import reward_coupling.sample as sample
    from dependent_rollouts.artifacts import sha256, write_json, write_jsonl
    from reward_coupling.bank import digest, group_weights, seal

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file(tiny.state_dict(), str(checkpoint / "model.safetensors"))
    pinned = tmp_path / "checkpoint.json"
    write_json(pinned, {"files": {"model.safetensors": sha256(checkpoint / "model.safetensors")}})
    config = {
        "group_size": 4,
        "feedback": "code",
        "epsilon": 1e-6,
        "token_logp_tolerance": 1e-5,
        "model_id": "tiny",
        "model_revision": "a" * 40,
        "tokenizer_revision": "b" * 40,
        "model_path": str(checkpoint),
        "checkpoint_manifest": str(pinned),
        "checkpoint_manifest_sha256": sha256(pinned),
    }
    verdicts = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]]

    def make_bank(name, split, prompt_id, prompt_ids, first_seed):
        bank = tmp_path / name
        bank.mkdir()
        rows = []
        for slot in range(4):
            rows.append(
                dict(
                    generate(tiny, prompt_ids, [15], first_seed + slot, 3),
                    prompt_ids=prompt_ids,
                    prompt_hash=digest(prompt_ids),
                    prompt_id=prompt_id,
                    split=split,
                    group_id=0,
                    group_seed=first_seed,
                    slot_id=slot,
                    trajectory_id=f"{prompt_id}:0:{slot}",
                    test_verdicts=verdicts[slot],
                    test_manifest_hash="fixture",
                )
            )
        write_json(
            bank / "manifest.json",
            {"actor_sampling": "iid", "split": split, "eos_ids": [15], "config": config},
        )
        write_jsonl(bank / "rows.jsonl", rows)
        seal(bank, "rows.jsonl")
        return bank, rows

    c_bank, c_rows = make_bank("C", "C", "c", [1, 2], 100)
    d_bank, _ = make_bank("D", "D", "d", [3, 4], 200)
    weighted = []
    weights = group_weights(c_rows, config)["difference"] / 4
    weighted.extend(zip(c_rows, weights, strict=True))
    difference = tmp_path / "difference"
    difference.mkdir()
    gradient, _ = aggregate_gradient(tiny, weighted)
    save_file(gradient, str(difference / "gradient.safetensors"))
    write_json(
        difference / "manifest.json",
        {"config": config, "arm": "difference", "bank_sha256": sha256(c_bank / "rows.jsonl")},
    )
    seal(difference, "gradient.safetensors")

    dev = tmp_path / "Dev-functions"
    dev.mkdir()
    definitions = []
    for coefficient, seed in ((1.0, 300), (-1.0, 301)):
        row = dict(generate(tiny, [5], [15], seed, 3), prompt_ids=[5])
        row.update({"coefficient": coefficient, "split": "Dev"})
        definitions.append(row)
    features = [{"prompt_id": "dev", "definitions": definitions}]
    write_json(dev / "functions.json", features)
    write_json(dev / "plan.json", {"config": config})
    write_json(dev / "receipt.json", {"functions_sha256": sha256(dev / "functions.json")})
    tasks = tmp_path / "tasks.jsonl"
    write_jsonl(tasks, [{"unused": True}])

    monkeypatch.setattr(sample, "load_model", lambda _: (tiny, object()))
    output = tmp_path / "result"
    argv = [
        "dev_step_calibration.py",
        "--bank",
        str(c_bank),
        "--difference",
        str(difference),
        "--tasks",
        str(tasks),
        "--functions",
        str(dev / "functions.json"),
        "--evaluation-bank",
        str(d_bank),
        "--project-evaluation",
        "--output",
        str(output),
    ]
    if project_only:
        argv.append("--project-only")

        def forbidden(*args, **kwargs):
            raise AssertionError("project-only reached GI aggregation or perturbation")

        import reward_coupling.gradient_audit as gradient_audit
        import reward_coupling.local_steps as local_steps

        monkeypatch.setattr(gradient_audit, "aggregate_gradient", forbidden)
        monkeypatch.setattr(local_steps, "perturb_from_checkpoint", forbidden)
    monkeypatch.setattr(sys, "argv", argv)
    before = {name: parameter.detach().clone() for name, parameter in tiny.named_parameters()}
    if project_only:
        with pytest.raises(SystemExit) as stopped:
            runpy.run_path("scripts/dev_step_calibration.py", run_name="__main__")
        assert stopped.value.code == 0
    else:
        runpy.run_path("scripts/dev_step_calibration.py", run_name="__main__")
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == (
        "C_D_preliminary_direction_only" if project_only else "C_D_preliminary_local_check"
    )
    assert receipt["independent_target_slopes"] is not None
    assert receipt["readout_ids"] == ["dev"]
    assert receipt["evaluation_bank"] == str(d_bank.resolve())
    assert receipt["evaluation_bank_rows_sha256"] == sha256(d_bank / "rows.jsonl")
    assert receipt["evaluation_bank_manifest_sha256"] == sha256(d_bank / "manifest.json")
    assert receipt["script_source_sha256"] == sha256("scripts/dev_step_calibration.py")
    assert (output / "D_target_prediction.json").is_file()
    partial = [
        json.loads(line)
        for line in (output / "D_target_rows.partial.jsonl").read_text().splitlines()
    ]
    assert len(partial) == 4
    assert [row["trajectory_id"] for row in partial] == [f"d:0:{slot}" for slot in range(4)]
    assert {row["artifact_status"] for row in partial} == {"unsealed_diagnostic"}
    assert all(isinstance(row["score_projection"], float) for row in partial)
    assert (output / "independent_D_importance.json").is_file() is not project_only
    if project_only:
        assert receipt["independent_slopes"] is None
        assert receipt["steps"] == []
        assert receipt["local_step_gate"] == "NOT_PASSED_ON_DEV"
    for name, parameter in tiny.named_parameters():
        assert torch.equal(parameter, before[name])


def test_cpu_gradient_offload_matches_default_and_cleans_hooks_after_error(tiny):
    rows = [dict(generate(tiny, [1, 2], [15], seed, 4), prompt_ids=[1, 2]) for seed in range(3)]
    weighted = list(zip(rows, [0.75, -0.25, 0.5], strict=True))

    tiny._feedback_offload_gradients = False
    expected, expected_error = aggregate_gradient(tiny, weighted)
    tiny._feedback_offload_gradients = True
    actual, actual_error = aggregate_gradient(tiny, weighted)
    assert actual_error == expected_error
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])

    rows[1]["sampling_token_logp"][0] += 0.1
    with pytest.raises(ValueError, match="score point"):
        aggregate_gradient(tiny, weighted)
    rows[1]["sampling_token_logp"][0] -= 0.1

    # A leaked post-accumulate hook would clear these ordinary device gradients.
    tiny._feedback_offload_gradients = False
    after_error, _ = aggregate_gradient(tiny, weighted)
    for name in expected:
        torch.testing.assert_close(after_error[name], expected[name])


@pytest.mark.parametrize("as_mapping", [False, True])
def test_local_judge_accepts_tensor_and_mapping_chat_encodings(as_mapping):
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            ids = torch.tensor([[1, 2]])
            return {"input_ids": ids} if as_mapping else ids

        def decode(self, *args, **kwargs):
            return "[1]"

    class Model:
        parameter = torch.nn.Parameter(torch.zeros(1))

        def parameters(self):
            yield self.parameter

        def generate(self, ids, **kwargs):
            return torch.cat([ids, torch.tensor([[3]], device=ids.device)], dim=1)

    judge = object.__new__(LocalRubricJudge)
    judge.config = {"system_prompt": "judge", "max_new_tokens": 3}
    judge.tokenizer = Tokenizer()
    judge.model = Model()
    result = judge.evaluate("response", {"prompt": "task", "criteria": ["correct"]})
    assert result["test_verdicts"] == [1]


def test_checkpoint_perturb_matches_memory_copy_and_restores_after_exception(tmp_path):
    from safetensors.torch import save_file

    torch.manual_seed(19)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2)).float()
    checkpoint = {
        name: torch.randn_like(parameter, dtype=torch.float16).contiguous()
        for name, parameter in model.named_parameters()
    }
    checkpoint_path = tmp_path / "model.safetensors"
    save_file(checkpoint, checkpoint_path)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(checkpoint[name].to(parameter.dtype))
    baseline = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    direction = {
        name: torch.randn_like(parameter).cpu() for name, parameter in model.named_parameters()
    }
    manifest = tmp_path / "checkpoint.json"
    write_json(manifest, {"files": {checkpoint_path.name: sha256(checkpoint_path)}})
    config = {
        "model_path": str(tmp_path),
        "checkpoint_manifest": str(manifest),
        "checkpoint_manifest_sha256": sha256(manifest),
    }

    with perturb(model, direction, 0.125):
        expected = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
        }
    with perturb_from_checkpoint(model, direction, 0.125, config):
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, baseline[name], rtol=0, atol=0)

    with pytest.raises(RuntimeError, match="evaluation failed"):
        with perturb_from_checkpoint(model, direction, -0.25, config):
            raise RuntimeError("evaluation failed")
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, baseline[name], rtol=0, atol=0)


def test_streamed_projected_gradient_matches_materialized_gradient(tiny):
    rows = [dict(generate(tiny, [1, 2], [15], seed, 4), prompt_ids=[1, 2]) for seed in range(3)]
    weighted = list(zip(rows, [0.75, -0.25, 0.5], strict=True))
    gradient, expected_error = aggregate_gradient(tiny, weighted)
    direction = {name: torch.randn_like(value) for name, value in gradient.items()}
    expected = dot(gradient, direction)
    del gradient
    actual, actual_error = _projected_gradient(tiny, weighted, direction, 1e-5)
    assert actual_error == expected_error
    assert actual == pytest.approx(expected, rel=2e-6, abs=1e-6)


def test_projected_gradient_rejects_accumulated_sequence_score_error(tiny, monkeypatch):
    import reward_coupling.local_steps as local_steps

    row = {"prompt_ids": [1], "response_ids": [2, 3], "sampling_token_logp": [-1.0, -1.0]}
    direction = {name: torch.zeros_like(parameter) for name, parameter in tiny.named_parameters()}

    def individually_close(*args, **kwargs):
        return torch.tensor([-0.994, -0.994])

    monkeypatch.setattr(local_steps, "token_logps", individually_close)
    tiny._feedback_sequence_tolerance = 0.01
    with pytest.raises(ValueError, match="Sequence score point mismatch"):
        _projected_gradient(tiny, [(row, 0.0)], direction, tolerance=0.01)

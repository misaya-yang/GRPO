import json

import pytest

from dependent_rollouts.artifacts import sha256, write_json, write_jsonl
from stratified_grpo.online import run_common_warmup, run_online

torch = pytest.importorskip("torch")


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_B = torch.nn.Parameter(torch.zeros(1))
        self._stratified_lora = {
            "rank": 1,
            "alpha": 1,
            "targets": ["fake"],
            "seed": 7,
            "initialization": "test",
            "adapter_dtype": "float32",
            "dropout": 0.0,
        }


def _tasks(path):
    rows = [
        {
            "prompt_id": prompt_id,
            "prompt": f"prompt {prompt_id}",
            "task_type": "numeric",
            "target": "1",
            "split": split,
        }
        for prompt_id, split in (("train-a", "train"), ("train-b", "train"), ("eval", "evaluation"))
    ]
    write_jsonl(path, rows)


def _decision(path, evidence_path):
    from stratified_grpo.model import parameter_identity

    parameter_hash = parameter_identity(TinyAdapter())["parameter_space_hash"]
    method_hash = "stage-d-method"
    manifest_path = evidence_path.with_name("stage-d-manifest.json")
    analysis_path = evidence_path.with_name("stage-d-analysis.json")
    write_json(
        manifest_path,
        {
            "config": {
                "stage": "confirmation",
                "metric": "full_lora_gradient_trace",
                "K": 2,
                "B": 2,
                "m": 2,
            },
            "method_hash": method_hash,
            "parameter_identity": {"parameter_space_hash": parameter_hash},
        },
    )
    write_json(
        analysis_path,
        {
            "stage": "confirmation",
            "decision": "SUPPORT_CONTINUATION",
            "metric": "full_lora_gradient_trace",
            "K": 2,
            "N": 4,
        },
    )
    write_json(evidence_path, {"status": "COMPLETE", "manifest_sha256": sha256(manifest_path)})
    write_json(
        path,
        {
            "stage": "D",
            "decision": "SUPPORT_CONTINUATION",
            "method_hash": method_hash,
            "parameter_space_hash": parameter_hash,
            "evidence": [
                {"role": "analysis", "path": str(analysis_path), "sha256": sha256(analysis_path)},
                {"role": "manifest", "path": str(manifest_path), "sha256": sha256(manifest_path)},
                {"role": "receipt", "path": str(evidence_path), "sha256": sha256(evidence_path)},
            ],
        },
    )


def _config(tasks, decision):
    stage_d = json.loads(decision.read_text())
    return {
        "K": 2,
        "B": 2,
        "m": 2,
        "epsilon": 1e-6,
        "updates": 2,
        "prompts_per_update": 1,
        "macros_per_prompt": 1,
        "evaluation_macros_per_prompt": 1,
        "stage_max_seconds": 30,
        "on_policy": True,
        "epochs_per_bank": 1,
        "clip_policy_ratio": False,
        "loss_reduction": "sum_tokens_then_mean_responses",
        "optimizer": "sgd",
        "learning_rate": 0.1,
        "seed": 11,
        "evaluation_seed": 12,
        "shared_initialization_id": "common-0",
        "tasks_sha256": sha256(tasks),
        "stage_d_decision_sha256": sha256(decision),
        "stage_d_decision": str(decision),
        "stage_d_method_hash": stage_d.get("method_hash"),
        "stage_d_parameter_space_hash": stage_d.get("parameter_space_hash"),
        "train_prompt_ids": ["train-a", "train-b"],
        "evaluation_prompt_ids": ["eval"],
        "token_logp_tolerance": 1e-5,
        "sequence_logp_tolerance": 1e-4,
        "stage": "online",
    }


def test_common_warmup_and_single_online_run_are_fresh_on_policy_and_auditable(
    tmp_path, monkeypatch
):
    import stratified_grpo.online as online

    tasks = tmp_path / "tasks.jsonl"
    decision = tmp_path / "decision.json"
    evidence = tmp_path / "stage-d-receipt.json"
    _tasks(tasks)
    _decision(decision, evidence)
    config = _config(tasks, decision)
    calls = []

    def collector(model, tokenizer, current, task, macro, arm, deadline):
        assert deadline > 0
        rows = []
        for block in range(current["B"]):
            for slot in range(current["m"]):
                identity = f"{current['seed']}:{task['prompt_id']}:{macro}:{arm}:{block}:{slot}"
                rows.append(
                    {
                        "prompt_id": task["prompt_id"],
                        "prompt_ids": [1],
                        "response_id": identity,
                        "response_ids": [2, 3],
                        "response_mask": [1, 1],
                        "sampling_token_logp": [-0.2, -0.3],
                        "old_logp": -0.5,
                        "rng_seed": identity,
                        "block": block,
                        "slot": slot,
                        "reward": (block + slot) % 2,
                    }
                )
        calls.append((current["seed"], arm, len(rows)))
        return rows, {"generation_seconds": 0.01, "scoring_seconds": 0.02}

    models = []

    def load(_):
        model = TinyAdapter()
        models.append(model)
        return model, object()

    gradient_calls = []

    def gradient(model, rows, weights, token_tolerance, sequence_tolerance):
        gradient_calls.append((len(rows), weights.copy()))
        return {"lora_B": torch.ones(1)}, {
            "max_token_error": 0.0,
            "max_sequence_error": 0.0,
        }

    monkeypatch.setattr(online, "load_model", load)
    monkeypatch.setattr(online, "aggregate_gradient", gradient)
    warm_config = tmp_path / "warm.json"
    write_json(warm_config, config)
    warm_output = tmp_path / "warm"
    warm = run_common_warmup(warm_config, tasks, warm_output, collector)
    assert warm["status"] == "common_warmup_complete"
    assert [count for _, arm, count in calls if arm == "iid_all"] == [4, 4, 4]
    assert [count for count, _ in gradient_calls] == [4, 4]
    assert models[0].lora_B.item() == pytest.approx(0.2)
    assert len((warm_output / "update-0000.jsonl").read_text().splitlines()) == 4
    assert len((warm_output / "evaluation.jsonl").read_text().splitlines()) == 4
    update_row = json.loads((warm_output / "update-0000.jsonl").read_text().splitlines()[0])
    for key in ("response_ids", "sampling_token_logp", "row_hash", "fresh_bank_hash", "weight"):
        assert key in update_row

    calls.clear()
    gradient_calls.clear()
    online_config = {**config, "arm": "stratified_full"}
    online_config.update(
        {
            "adapter_path": warm["adapter"],
            "adapter_sha256": warm["adapter_sha256"],
            "common_checkpoint_receipt": str(warm_output / "receipt.json"),
            "common_checkpoint_receipt_sha256": sha256(warm_output / "receipt.json"),
        }
    )
    online_path = tmp_path / "online.json"
    write_json(online_path, online_config)
    online_output = tmp_path / "online"
    result = run_online(online_path, tasks, online_output, collector)
    assert result["status"] == "online_run_complete"
    assert result["arm"] == "stratified_full"
    assert [count for _, arm, count in calls if arm == "stratified_full"] == [4, 4]
    assert [count for count, _ in gradient_calls] == [4, 4]
    assert [count for seed, arm, count in calls if seed == 12 and arm == "iid_all"] == [4]
    assert (online_output / "adapter.safetensors").is_file()
    assert (online_output / "optimizer.pt").is_file()

    calls.clear()
    gradient_calls.clear()
    split_config = {
        **online_config,
        "arm": "iid_all",
        "iid_budget_mode": "more_prompts",
        "updates": 1,
    }
    split_path = tmp_path / "online-split.json"
    write_json(split_path, split_config)
    split_output = tmp_path / "online-split"
    split = run_online(split_path, tasks, split_output, collector)
    assert split["effective_prompts_per_update"] == 2
    assert split["responses_per_update"] == [4]
    assert [count for seed, arm, count in calls if seed == 11 and arm == "iid_all"] == [2, 2]
    assert [count for seed, arm, count in calls if seed == 12 and arm == "iid_all"] == [4]


def test_online_rejects_bare_or_unsealed_stage_d_go_before_model_loading(tmp_path, monkeypatch):
    import stratified_grpo.online as online

    tasks = tmp_path / "tasks.jsonl"
    _tasks(tasks)
    decision = tmp_path / "decision.json"
    write_json(decision, {"stage": "D", "decision": "SUPPORT_CONTINUATION", "evidence": []})
    config = _config(tasks, decision)
    config_path = tmp_path / "config.json"
    write_json(config_path, config)

    def forbidden(_):
        raise AssertionError("model loaded before Stage-D evidence validation")

    monkeypatch.setattr(online, "load_model", forbidden)
    with pytest.raises(ValueError, match="requires evidence"):
        run_common_warmup(config_path, tasks, tmp_path / "output", lambda *args: None)

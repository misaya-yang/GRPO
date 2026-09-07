import json

import numpy as np
import pytest

from stratified_grpo.statistics import centered_gram, compare, trace_variance


def test_macro_trace_and_bootstrap_scope():
    x = np.array([[1, 2], [2, 0], [-1, 3], [4, 5.0]])
    gram, _ = centered_gram(x)
    assert trace_variance(gram) == pytest.approx(np.var(x, axis=0, ddof=1).sum())
    records = [
        {
            "iid_gram": gram,
            "stratified_gram": gram * 0.1,
            "iid_costs": [1] * 4,
            "stratified_costs": [2] * 4,
        }
        for _ in range(12)
    ]
    result = compare(records, replicates=200)
    assert result["rho"] == pytest.approx(0.1)
    assert result["rho_time"] == pytest.approx(0.2)
    assert result["decision"] == "SUPPORT_CONTINUATION"
    assert compare(records[:2], replicates=100)["decision"] == "INCONCLUSIVE"
    records[0]["iid_gram"] = np.zeros((4, 4))
    assert compare(records[:1], replicates=100)["rho"] is None


def test_real_tiny_model_score_lora_and_pipeline(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from dependent_rollouts.artifacts import sha256
    from stratified_grpo import pipeline
    from stratified_grpo.model import aggregate_gradient, install_lora, parameter_identity
    from stratified_grpo.sampling import generate

    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=12,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            eos_token_id=11,
            attention_dropout=0.0,
            attn_implementation="eager",
        )
    )
    install_lora(model, rank=2, alpha=4, seed=13)
    identity = parameter_identity(model)
    row = generate(model, [1, 2], [11], 3, 4, 0, 2, use_cache=True)
    row["prompt_ids"] = [1, 2]
    gradient, error = aggregate_gradient(model, [row], [1.0], 1e-5, 1e-5)
    assert error["l2_A"] == 0
    assert error["l2_B"] > 0
    assert parameter_identity(model) == identity
    other = transformers.LlamaForCausalLM(model.config)
    install_lora(other, rank=2, alpha=4, seed=13)
    assert parameter_identity(other) == identity

    class Tokenizer:
        chat_template = "tiny-test"
        eos_token_id = 11

        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]

        def decode(self, ids, **kwargs):
            return "<answer>1</answer>"

    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "prompt_id": "tiny",
                "prompt": "one",
                "task_type": "numeric",
                "target": "1",
                "split": "calibration",
            }
        )
        + "\n"
    )
    config = {
        "K": 2,
        "B": 2,
        "m": 2,
        "macro_repeats": 2,
        "max_new_tokens": 4,
        "stage_max_seconds": 30,
        "arms": ["iid_all", "stratified_full"],
        "stage": "dev",
        "base_dtype": "float32",
        "model_revision": "a" * 40,
        "tokenizer_revision": "a" * 40,
        "score": "original_policy_sum_including_emitted_EOS",
        "epsilon": 1e-6,
        "sampling": "independent_conditional_interval",
        "onset": 0,
        "token_order": "vocabulary_id",
        "temperature": 1.0,
        "token_logp_tolerance": 1e-5,
        "sequence_logp_tolerance": 1e-5,
        "cdf_tv_tolerance": 1e-10,
        "practical_threshold": 0.85,
        "metric": "full_lora_gradient_trace",
        "tasks_sha256": sha256(tasks),
        "prompt_ids": ["tiny"],
        "seed": 17,
        "use_cache": True,
        "projection_seed": 18,
        "analysis_seed": 19,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(pipeline, "load_model", lambda _: (model, Tokenizer()))
    result = pipeline.run(path, tasks, tmp_path / "run")
    assert result["stage"] == "dev"
    assert (tmp_path / "run" / "REPORT.md").exists()
    assert json.loads((tmp_path / "run" / "receipt.json").read_text())["rows"] == 16

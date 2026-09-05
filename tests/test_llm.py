"""Random-weight tiny Qwen plumbing only; no pretrained evidence or downloads."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import Qwen2Config, Qwen2ForCausalLM

from dependent_rollouts.artifacts import sha256, write_json, write_jsonl
from dependent_rollouts.interventions import perturb
from dependent_rollouts.llm import (
    aggregate_gradient,
    bank_weights,
    generate_group,
    read_bank,
    sequence_logp,
)


@pytest.fixture
def model():
    torch.manual_seed(2027)
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
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).eval()


@pytest.mark.parametrize("sampler", ["iid", "stratified", "lattice"])
def test_generation_score_point_and_response_sum(model, sampler):
    rows = generate_group(model, [1, 2], [15], sampler, 4, 42, 8)
    for row in rows:
        assert 1 <= len(row["response_ids"]) <= 8
        lp = sequence_logp(model, [1, 2], row["response_ids"])
        assert lp.item() == pytest.approx(row["old_logp"], abs=2e-5)
        assert row["finish_reason"] == ("eos" if row["response_ids"][-1] == 15 else "length")
        manual = 0.0
        ids = [1, 2]
        with torch.no_grad():
            for token in row["response_ids"]:
                logits = model(torch.tensor([ids])).logits[0, -1].float()
                manual += logits.log_softmax(-1)[token].item()
                ids.append(token)
        assert lp.item() == pytest.approx(manual, abs=2e-5)


def test_weighted_backward_and_finite_difference(model):
    row = {"prompt_ids": [1, 2], "response_ids": [3, 4, 15]}
    row["old_logp"] = sequence_logp(model, row["prompt_ids"], row["response_ids"]).item()
    gradients, error = aggregate_gradient(model, [(row, 0.7)])
    assert error < 1e-6
    norm2 = sum(float(g.double().square().sum()) for g in gradients.values())
    baseline = {n: p.detach().clone() for n, p in model.named_parameters()}
    h = 1e-4
    with perturb(model, gradients, h):
        plus = 0.7 * sequence_logp(model, row["prompt_ids"], row["response_ids"]).item()
    with perturb(model, gradients, -h):
        minus = 0.7 * sequence_logp(model, row["prompt_ids"], row["response_ids"]).item()
    assert (plus - minus) / (2 * h) == pytest.approx(norm2, rel=2e-3)
    for n, p in model.named_parameters():
        assert torch.equal(p, baseline[n])
    with pytest.raises(RuntimeError), perturb(model, gradients, h):
        raise RuntimeError("Interrupted evaluation")
    for n, p in model.named_parameters():
        assert torch.equal(p, baseline[n])


def test_bank_prompt_separation_and_hash_guard(tmp_path):
    config = {"k": 2, "sampler": "iid"}
    write_json(tmp_path / "manifest.json", {"config": config})
    rows = [
        {
            "trajectory_id": f"{p}:{g}:{m}",
            "prompt_id": p,
            "group_id": g,
            "member": m,
            "reward": m,
            "latent_bin": m,
        }
        for p in ("a", "b")
        for g in range(2)
        for m in range(2)
    ]
    write_jsonl(tmp_path / "rollouts.jsonl", rows)
    write_json(
        tmp_path / "receipt.json",
        {"status": "complete", "rollouts_sha256": sha256(tmp_path / "rollouts.jsonl")},
    )
    manifest, pools = read_bank(tmp_path)
    weighted = bank_weights(manifest, pools, "count")
    assert len(weighted) == 8
    assert sum(w for _, w in weighted) == pytest.approx(0)
    with (tmp_path / "rollouts.jsonl").open("a") as f:
        f.write(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="modified"):
        read_bank(tmp_path)


def test_iid_forecast_backward_equals_explicit_virtual_groups(model):
    from dependent_rollouts.estimators import advantages

    rows = []
    for token, reward, bin_id in ((3, 1, 0), (4, 0, 0), (5, 1, 1), (6, 0, 1)):
        row = {
            "prompt_ids": [1, 2],
            "response_ids": [token, 15],
            "reward": reward,
            "latent_bin": bin_id,
        }
        row["old_logp"] = sequence_logp(model, row["prompt_ids"], row["response_ids"]).item()
        rows.append(row)
    weighted = bank_weights({"config": {"k": 2, "sampler": "iid"}}, {"prompt": rows}, "forecast")
    predicted, _ = aggregate_gradient(model, weighted)
    model.zero_grad(set_to_none=True)
    for left in rows[:2]:
        for right in rows[2:]:
            a = advantages([[left["reward"], right["reward"]]])[0]
            for row, weight in zip((left, right), a, strict=True):
                (
                    sequence_logp(model, row["prompt_ids"], row["response_ids"]) * float(weight) / 8
                ).backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            np.testing.assert_allclose(predicted[name].numpy(), parameter.grad.numpy(), atol=3e-7)

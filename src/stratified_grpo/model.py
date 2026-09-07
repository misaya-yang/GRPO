"""Frozen-base, shared-initialization LoRA backend; no PEFT runtime dependency."""

import hashlib
import json
from pathlib import Path

import numpy as np

from dependent_rollouts.artifacts import sha256
from reward_coupling.bank import digest


def install_lora(model, rank=8, alpha=16, targets=("q_proj", "v_proj"), seed=20260907):
    import torch
    from torch import nn

    if type(rank) is not int or rank < 1 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Invalid LoRA rank/alpha")
    model.requires_grad_(False)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    class LoRALinear(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.scale = alpha / rank
            a = torch.randn((rank, base.in_features), generator=generator, dtype=torch.float32)
            a *= base.in_features**-0.5
            self.lora_A = nn.Parameter(a.to(base.weight.device))
            self.lora_B = nn.Parameter(
                torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32)
            )

        def forward(self, x):
            result = self.base(x)
            update = (x.float() @ self.lora_A.T) @ self.lora_B.T
            return result + (update * self.scale).to(result.dtype)

    selected = [
        (name, module)
        for name, module in model.named_modules()
        if name.split(".")[-1] in targets and isinstance(module, nn.Linear)
    ]
    if not selected or {name.split(".")[-1] for name, _ in selected} != set(targets):
        raise ValueError("Requested LoRA target modules not all present")
    for name, module in selected:
        parent_name, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent_name), leaf, LoRALinear(module))
    model.eval()
    model._stratified_lora = {
        "rank": rank,
        "alpha": alpha,
        "targets": list(targets),
        "seed": seed,
        "initialization": "A_normal_std_1/sqrt(in_features)_B_zero",
        "adapter_dtype": "float32",
        "dropout": 0.0,
    }
    return model


def trainable(model):
    return {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def parameter_identity(model):
    parameters = trainable(model)
    h = hashlib.sha256()
    layout = []
    for name, parameter in parameters.items():
        value = parameter.detach().float().cpu().contiguous().numpy()
        layout.append({"name": name, "shape": list(value.shape), "dtype": str(parameter.dtype)})
        h.update(name.encode())
        h.update(value.tobytes())
    definition = {"lora": model._stratified_lora, "layout": layout}
    return {
        "definition": definition,
        "parameter_space_hash": digest(definition),
        "adapter_values_hash": h.hexdigest(),
        "trainable_count": sum(p.numel() for p in parameters.values()),
    }


def verify_assets(config):
    manifest = Path(config["checkpoint_manifest"])
    if sha256(manifest) != config["checkpoint_manifest_sha256"]:
        raise ValueError("Checkpoint manifest changed")
    files = json.loads(manifest.read_text())["files"]
    for filename, expected in files.items():
        if sha256(Path(config["model_path"]) / filename) != expected:
            raise ValueError(f"Checkpoint file changed: {filename}")
    return {"checkpoint_hash": config["checkpoint_manifest_sha256"], "files_verified": len(files)}


def load_model(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    verify_assets(config)
    torch.set_num_threads(config.get("cpu_threads", 4))
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        local_files_only=True,
        trust_remote_code=False,
        revision=config["model_revision"],
        torch_dtype=getattr(torch, config["base_dtype"]),
        attn_implementation="eager",
    ).to(config["device"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"],
        local_files_only=True,
        revision=config["tokenizer_revision"],
        trust_remote_code=False,
    )
    install_lora(model, **config["lora"])
    if config.get("adapter_path"):
        from safetensors.torch import load_file

        if sha256(config["adapter_path"]) != config["adapter_sha256"]:
            raise ValueError("Common adapter changed")
        values = load_file(config["adapter_path"])
        params = trainable(model)
        if values.keys() != params.keys() or any(
            values[n].shape != p.shape for n, p in params.items()
        ):
            raise ValueError("Common adapter parameter space differs")
        with torch.no_grad():
            for name, p in params.items():
                p.copy_(values[name].to(p.device, p.dtype))
    model.eval()
    return model, tokenizer


def token_logps(model, prompt_ids, response_ids):
    import torch

    if not prompt_ids or not response_ids:
        raise ValueError("Nonempty response and prompt required")
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([prompt_ids + response_ids], device=device)
    logits = model(input_ids=ids, use_cache=False).logits[0, len(prompt_ids) - 1 : -1].double()
    return logits.log_softmax(-1).gather(1, ids[0, len(prompt_ids) :, None]).squeeze(1)


def aggregate_gradient(model, rows, weights, token_tolerance, sequence_tolerance):
    """weights are advantages; the estimator is mean_i weights_i score_i."""
    import torch

    if len(rows) != len(weights) or not rows or not np.isfinite(weights).all():
        raise ValueError("Invalid bank/weights")
    model.eval().zero_grad(set_to_none=True)
    errors = []
    for row, weight in zip(rows, weights, strict=True):
        with torch.set_grad_enabled(bool(weight)):
            lp = token_logps(model, row["prompt_ids"], row["response_ids"])
        actual = lp.detach().cpu().numpy()
        expected = np.asarray(row["sampling_token_logp"])
        error = float(np.max(np.abs(actual - expected)))
        sequence_error = abs(float(actual.sum() - expected.sum()))
        if error > token_tolerance or sequence_error > sequence_tolerance:
            raise ValueError(f"Sampling/scoring mismatch: token={error}, sequence={sequence_error}")
        errors.append((error, sequence_error))
        if weight:
            (lp.sum() * float(weight) / len(rows)).backward()
        del lp
    gradients = {
        name: (
            torch.zeros_like(p, device="cpu")
            if p.grad is None
            else p.grad.detach().float().cpu().clone()
        ).contiguous()
        for name, p in trainable(model).items()
    }
    if not all(torch.isfinite(g).all() for g in gradients.values()):
        raise FloatingPointError("Nonfinite gradient")
    model.zero_grad(set_to_none=True)
    return gradients, {
        "max_token_error": max(e[0] for e in errors),
        "max_sequence_error": max(e[1] for e in errors),
        "l2_A": float(
            np.sqrt(
                sum(
                    float(v.double().square().sum())
                    for n, v in gradients.items()
                    if n.endswith("lora_A")
                )
            )
        ),
        "l2_B": float(
            np.sqrt(
                sum(
                    float(v.double().square().sum())
                    for n, v in gradients.items()
                    if n.endswith("lora_B")
                )
            )
        ),
    }

"""Matched two-independent-groups online RLOO reference runner, one epoch per batch."""

import json
import random
import time
from pathlib import Path

import numpy as np

from .artifacts import provenance, write_json, write_jsonl
from .estimators import advantages
from .llm import generate_group, load_model, sequence_logp
from .tasks import read_tasks, verify


def train(config_path, tasks_path, output):
    import torch

    config = json.loads(Path(config_path).read_text())
    if config["groups_per_prompt"] != 2 or config["arm"] not in ("within", "cross"):
        raise ValueError("Primary factorial uses two groups in every within/cross arm")
    if config.get("pilot_decision") != "GO":
        raise ValueError("Training config needs pilot_decision=GO after the four dossier gates")
    tasks = read_tasks(tasks_path)
    if any(t["split"] != "train" for t in tasks):
        raise ValueError("Training tasks must belong to the train split")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "manifest.json", provenance(config))
    model, tokenizer = load_model(config)
    optimizer_name = config.get("optimizer", "sgd")
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"], weight_decay=0
        )
    else:
        raise ValueError("Use sgd or adamw")
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    rng = random.Random(config["seed"])
    logs, tokens, start = [], 0, time.monotonic()
    for update in range(config["updates"]):
        if time.monotonic() - start > config["max_wall_seconds"]:
            write_jsonl(out / "partial_training.jsonl", logs)
            raise TimeoutError("Training time cap reached; matrix run incomplete")
        chosen = rng.sample(tasks, min(config["prompts_per_update"], len(tasks)))
        batch = []
        for task in chosen:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": task["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            rows = []
            for group in range(2):
                seed = rng.getrandbits(128)
                generated = generate_group(
                    model,
                    prompt,
                    eos,
                    config["sampler"],
                    config["k"],
                    seed,
                    config["max_new_tokens"],
                )
                for member, row in enumerate(generated):
                    text = tokenizer.decode(row["response_ids"], skip_special_tokens=True)
                    row.update(
                        {
                            "prompt_ids": prompt,
                            "prompt_id": task["prompt_id"],
                            "group_id": group,
                            "member": member,
                            "group_seed": seed,
                            "text": text,
                            **verify(text, task, row["finish_reason"]),
                        }
                    )
                    rows.append(row)
            rewards = np.array([r["reward"] for r in rows]).reshape(2, config["k"])
            weights = advantages(rewards, config["arm"]).reshape(-1) / (len(chosen) * len(rows))
            batch.extend(zip(rows, weights, strict=True))
        optimizer.zero_grad(set_to_none=True)
        objective, max_error = 0.0, 0.0
        for row, weight in batch:
            lp = sequence_logp(model, row["prompt_ids"], row["response_ids"])
            max_error = max(max_error, abs(lp.detach().item() - row["old_logp"]))
            if max_error > 5e-3:
                raise ValueError("Score-point check failed before optimizer step")
            loss = -float(weight) * lp
            objective += loss.detach().item()
            loss.backward()
            tokens += len(row["response_ids"])
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite gradient")
        optimizer.step()
        write_jsonl(out / f"rollouts-{update:04d}.jsonl", [row for row, _ in batch])
        logs.append(
            {
                "update": update,
                "loss": objective,
                "generated_tokens": tokens,
                "reward": float(np.mean([row["reward"] for row, _ in batch])),
                "max_score_error": max_error,
                "elapsed_seconds": time.monotonic() - start,
            }
        )
        if (update + 1) % config.get("checkpoint_every", 25) == 0:
            checkpoint = out / f"checkpoint-{update + 1}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
    model.save_pretrained(out / "final")
    tokenizer.save_pretrained(out / "final")
    write_jsonl(out / "training.jsonl", logs)
    write_json(
        out / "receipt.json",
        {
            "status": "complete",
            "updates": len(logs),
            "generated_tokens": tokens,
            "elapsed_seconds": time.monotonic() - start,
        },
    )
    return logs[-1]

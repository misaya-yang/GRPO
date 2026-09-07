"""Fresh IID candidates each update; native coupling factorial, one SGD step."""

import json
import time
from pathlib import Path

import numpy as np

from dependent_rollouts.artifacts import sha256, write_json, write_jsonl

from .advantages import advantages
from .bank import digest, read_tasks
from .contract import check_task_hash, read_config
from .execute_tests import make_executor
from .expectation import expected_advantages, rubric_matrix, sample_advantages
from .sample import encode_prompt, generate, load_model, token_logps
from .statistics import pilot_decision


def train(config_path, tasks_path, decision_path, output):
    import torch

    config = read_config(config_path)
    decision = json.loads(Path(decision_path).read_text())
    if pilot_decision(decision) != "GO" or sha256(decision_path) != config.get("decision_sha256"):
        raise ValueError("Training requires a frozen evidence-bearing pilot GO")
    for gate in decision["gates"].values():
        for item in gate["evidence"]:
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError("Pilot evidence changed")
    check_task_hash(config, tasks_path)
    tasks = [t for t in read_tasks(tasks_path) if t["split"] == "train"]
    learning_rate = config.get("learning_rate")
    if (
        config.get("optimizer") not in ("sgd", "adamw")
        or not isinstance(learning_rate, (int, float))
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("Invalid training settings")
    for key in ("updates", "prompts_per_update", "checkpoint_every"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"Invalid training setting: {key}")
    if config.get("advantage") not in ("standardized", "rloo", "mean_only"):
        raise ValueError("Invalid training advantage")
    if len(tasks) < config["prompts_per_update"]:
        raise ValueError("Insufficient independent training prompts")
    if config["coupling"] not in ("shared", "independent", "full", "expected_independent"):
        raise ValueError("Invalid native coupling arm")
    if config["feedback"] == "code":
        scorer = make_executor(config)
        scorer.preflight()
        scorer.check_references(tasks)
    else:
        from .judge import LocalRubricJudge

        scorer = LocalRubricJudge(config["judge"])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "manifest.json",
        {"config": config, "decision_sha256": sha256(decision_path), "status": "started"},
    )
    model, tokenizer = load_model(config)
    optimizer = (
        torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
        if config["optimizer"] == "sgd"
        else torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0)
    )
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    eos = eos or [tokenizer.eos_token_id]
    if None in eos:
        raise ValueError("Missing EOS")
    actor_rng = np.random.default_rng(config["seed"])
    feedback_rng = np.random.default_rng(config["seed"] + 1000000)
    start = time.monotonic()
    deadline = start + config["stage_max_seconds"]
    logs = []
    for update in range(config["updates"]):
        chosen = actor_rng.choice(len(tasks), size=config["prompts_per_update"], replace=False)
        weighted, records = [], []
        for task_id in chosen:
            task = tasks[int(task_id)]
            prompt = encode_prompt(tokenizer, task["prompt"])
            for group_id in range(config["groups_per_prompt"]):
                group = []
                for slot in range(config["group_size"]):
                    seed = int(actor_rng.integers(2**62))
                    row = generate(model, prompt, eos, seed, config["max_new_tokens"], deadline)
                    text = tokenizer.decode(row["response_ids"], skip_special_tokens=True)
                    row.update(
                        {
                            "prompt_ids": prompt,
                            "prompt_id": task["prompt_id"],
                            "update": update,
                            "group_id": group_id,
                            "slot_id": slot,
                            "text": text,
                            **scorer.evaluate(text, task),
                        }
                    )
                    group.append(row)
                matrix = np.array([r["test_verdicts"] for r in group], float)
                scale = None
                if config["feedback"] == "rubric":
                    scale = config["rubric_keep"]
                    matrix, _ = rubric_matrix(matrix, scale)
                mu = np.ones(matrix.shape[1]) / matrix.shape[1]
                if config["coupling"] == "full":
                    a = advantages(matrix.mean(axis=1), config["advantage"], config["epsilon"])
                    contexts = []
                elif config["coupling"] == "expected_independent":
                    a = expected_advantages(
                        matrix, mu, config["advantage"], config["epsilon"], scale
                    )["independent"]
                    contexts = []
                else:
                    a, contexts = sample_advantages(
                        matrix,
                        mu,
                        config["coupling"],
                        feedback_rng,
                        config["advantage"],
                        config["epsilon"],
                    )
                denominator = len(chosen) * config["groups_per_prompt"] * config["group_size"]
                weighted.extend(zip(group, a / denominator, strict=True))
                for row, weight in zip(group, a, strict=True):
                    row["advantage"] = float(weight)
                    row["sampled_contexts"] = list(map(int, contexts))
                    row["feedback_law_hash"] = digest([matrix.tolist(), mu.tolist()])
                    records.append(row)
        optimizer.zero_grad(set_to_none=True)
        for row, weight in weighted:
            lp = token_logps(model, row["prompt_ids"], row["response_ids"])
            if (
                np.max(np.abs(lp.detach().cpu().numpy() - row["sampling_token_logp"]))
                > config["token_logp_tolerance"]
            ):
                raise ValueError("Training score point mismatch")
            (-float(weight) * lp.sum()).backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite training gradient")
        if time.monotonic() >= deadline:
            raise TimeoutError("Budget exhausted before update")
        optimizer.step()
        write_jsonl(out / f"update-{update:04d}.jsonl", records)
        logs.append(
            {
                "update": update,
                "responses": len(records),
                "generated_tokens": sum(r["response_length"] for r in records),
                "elapsed_seconds": time.monotonic() - start,
            }
        )
        if (update + 1) % config["checkpoint_every"] == 0 or update + 1 == config["updates"]:
            directory = out / f"checkpoint-{update + 1:04d}"
            model.save_pretrained(directory)
            tokenizer.save_pretrained(directory)
            # Store optimizer and both independent RNG streams for diagnosis/replay.
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "actor_rng": actor_rng.bit_generator.state,
                    "feedback_rng": feedback_rng.bit_generator.state,
                    "update": update + 1,
                },
                directory / "training_state.pt",
            )
    write_jsonl(out / "training.jsonl", logs)
    result = {
        "status": "training_complete_evaluation_required",
        "updates": len(logs),
        "optimizer": config["optimizer"],
        "online_evaluation": "NOT_RUN",
        "elapsed_seconds": time.monotonic() - start,
    }
    write_json(out / "receipt.json", result)
    return result

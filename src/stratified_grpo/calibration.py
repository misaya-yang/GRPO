"""One fixed Dev numerical calibration before the independent pilot banks."""

import json
import time
from pathlib import Path

import numpy as np

from dependent_rollouts.artifacts import sha256, write_json
from reward_coupling.sample import encode_prompt

from .contract import selected_tasks, validate
from .model import aggregate_gradient, load_model, parameter_identity, token_logps
from .sampling import generate, stream_seed


def calibrate(config_path, tasks_path, output):
    import torch

    config = validate(json.loads(Path(config_path).read_text()))
    if config["stage"] != "dev":
        raise ValueError("Numerical calibration is Dev only")
    tasks = selected_tasks(config, tasks_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model, tokenizer = load_model(config)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    eos = eos or [tokenizer.eos_token_id]
    rows, errors = [], []
    # Fixed first two Dev tasks; both strata, separate streams, full horizon.
    for task in tasks[:2]:
        prompt = encode_prompt(tokenizer, task["prompt"])
        for j in range(config["m"]):
            row = generate(
                model,
                prompt,
                eos,
                stream_seed(config["seed"], "numeric", task["prompt_id"], 0, "s", 0, j),
                config["max_new_tokens"],
                j,
                config["m"],
                config["onset"],
                config["use_cache"],
                started + config["stage_max_seconds"],
            )
            row.update({"prompt_ids": prompt, "prompt_id": task["prompt_id"]})
            with torch.no_grad():
                actual = token_logps(model, prompt, row["response_ids"]).cpu().numpy()
            token_error = float(np.max(np.abs(actual - row["sampling_token_logp"])))
            seq_error = abs(float(actual.sum() - sum(row["sampling_token_logp"])))
            rows.append(row)
            errors.append({"token_error": token_error, "sequence_error": seq_error})
    # Engineering ceiling is frozen before measurements; no repeated tolerance search.
    token_max = max(e["token_error"] for e in errors)
    seq_max = max(e["sequence_error"] for e in errors)
    ceilings = config["numeric_calibration_ceiling"]
    passed = token_max <= ceilings["token"] and seq_max <= ceilings["sequence"]
    report = {
        "status": "PASS" if passed else "NUMERIC_CONTRACT_FAILED",
        "errors": errors,
        "elapsed_seconds": time.monotonic() - started,
        "parameter_identity": parameter_identity(model),
        "base_dtype": config["base_dtype"],
        "ceiling": ceilings,
        "source_config_sha256": sha256(config_path),
        "evidence": "Dev_numeric_only_not_effect",
    }
    write_json(out / "rows.json", rows)
    report["rows_sha256"] = sha256(out / "rows.json")
    if passed:
        # A frozen safety factor, capped by the declared engineering ceiling.
        config["token_logp_tolerance"] = min(ceilings["token"], max(1e-3, 5 * token_max))
        config["sequence_logp_tolerance"] = min(ceilings["sequence"], max(0.01, 5 * seq_max))
        _, score = aggregate_gradient(
            model,
            [rows[0]],
            [1.0],
            config["token_logp_tolerance"],
            config["sequence_logp_tolerance"],
        )
        report["score_probe"] = score
        if score["l2_B"] <= 0:
            report["status"] = "NO_LORA_SCORE_SIGNAL"
        write_json(out / "calibrated_config.json", config)
    write_json(out / "receipt.json", report)
    return report

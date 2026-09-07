"""Dev-only numeric calibration of prefix generation vs teacher-forced FP32 score."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dependent_rollouts.artifacts import write_json
from reward_coupling.bank import digest, read_tasks
from reward_coupling.sample import encode_prompt, generate, load_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", required=True)
parser.add_argument("--tasks", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
config = json.loads(Path(args.config).read_text())
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
model, tokenizer = load_model(config)
eos = model.generation_config.eos_token_id
eos = [eos] if isinstance(eos, int) else eos
report = []
for task in [t for t in read_tasks(args.tasks) if t["split"] == "Dev"][:2]:
    group_seed = int(digest([config["seed"], "Dev", task["prompt_id"], 0])[:15], 16)
    seed = int(digest([group_seed, 0])[:15], 16)
    start = time.monotonic()
    row = generate(
        model,
        encode_prompt(tokenizer, task["prompt"]),
        eos,
        seed,
        config["max_new_tokens"],
        use_cache=config.get("use_cache", False),
    )
    delta = np.array(row["scoring_token_logp"]) - row["sampling_token_logp"]
    result = {
        "prompt_id": task["prompt_id"],
        "tokens": row["response_length"],
        "elapsed_seconds": time.monotonic() - start,
        "max_abs_error": float(abs(delta).max()),
        "sequence_error": float(delta.sum()),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "row": row,
    }
    report.append(result)
    print(json.dumps({k: v for k, v in result.items() if k != "row"}), flush=True)
write_json(
    out / "receipt.json",
    {"status": "Dev_numeric_measurement", "config": config, "observations": report},
)

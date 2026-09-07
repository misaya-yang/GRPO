"""Small real-checkpoint method probe: no natural-effect or training claim."""

import argparse
import json
import time
from pathlib import Path

import torch

from dependent_rollouts.artifacts import write_json, write_jsonl
from reward_coupling.bank import read_tasks
from reward_coupling.gradient_audit import aggregate_gradient, dot
from reward_coupling.sample import encode_prompt, generate, load_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", required=True)
parser.add_argument("--tasks", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
config = json.loads(Path(args.config).read_text())
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
write_json(out / "config.json", config)
torch.set_num_threads(4)
start = time.monotonic()
model, tokenizer = load_model(config)
loaded = time.monotonic()
eos = model.generation_config.eos_token_id
eos = [eos] if isinstance(eos, int) else eos
task = next(t for t in read_tasks(args.tasks) if t["split"] == "Dev")
prompt = encode_prompt(tokenizer, task["prompt"])
row = generate(model, prompt, eos, config["seed"], 32)
row["prompt_ids"] = prompt
row["text"] = tokenizer.decode(row["response_ids"], skip_special_tokens=True)
write_jsonl(out / "rows.jsonl", [row])
generated = time.monotonic()
torch.cuda.reset_peak_memory_stats()
gradient, error = aggregate_gradient(model, [(row, 1.0)], config["token_logp_tolerance"])
finished = time.monotonic()
result = {
    "status": "PASS_pretrained_score_and_full_gradient_plumbing_only",
    "model": config["model_id"],
    "revision": config["model_revision"],
    "dtype": config["dtype"],
    "gradient_storage": "CPU_after_accumulation",
    "tokens": len(row["response_ids"]),
    "truncated": row["truncated"],
    "max_token_error": error,
    "loading_seconds": loaded - start,
    "generation_seconds": generated - loaded,
    "gradient_seconds": finished - generated,
    "gradient_norm": dot(gradient, gradient) ** 0.5,
    "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
    "gpu": torch.cuda.get_device_name(),
    "natural_coupling_effect": "NOT_MEASURED",
    "pilot_decision": "NOT_RUN",
}
write_json(out / "receipt.json", result)
print(json.dumps(result, indent=2), flush=True)

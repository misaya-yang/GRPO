"""Import an explicitly supplied SVAMP JSON; no benchmark download or tuning."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from dependent_rollouts.artifacts import sha256, write_json, write_jsonl

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source")
parser.add_argument("output_dir")
parser.add_argument("--seed", type=int, default=202704)
args = parser.parse_args()
raw = json.loads(Path(args.source).read_text())
unique = {}
for item in raw:
    text = (item["Body"].strip() + " " + item["Question"].strip()).strip()
    normalized = " ".join(text.casefold().split())
    identity = hashlib.sha256(normalized.encode()).hexdigest()
    if identity in unique and str(unique[identity]["target"]) != str(item["Answer"]):
        raise ValueError("Duplicate text with conflicting answers")
    unique[identity] = {
        "prompt_id": "svamp-" + identity[:16],
        "task_type": "numeric",
        "target": str(item["Answer"]),
        "source_id": item.get("ID"),
        "prompt": text + " You may reason first. End with exactly one "
        "<answer>number</answer> tag and nothing after it.",
    }
rows = list(unique.values())
random.Random(args.seed).shuffle(rows)
if len(rows) < 80:
    raise ValueError("Insufficient deduplicated data for fixed splits")
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=False)
offset = 0
for split, count in (
    ("calibration", 16),
    ("discovery", 8),
    ("confirmation", 16),
    ("evaluation", 32),
    ("train", len(rows) - 72),
):
    selected = [dict(row, split=split) for row in rows[offset : offset + count]]
    write_jsonl(out / f"{split}.jsonl", selected)
    offset += count
write_json(
    out / "manifest.json",
    {
        "source_sha256": sha256(args.source),
        "seed": args.seed,
        "original_rows": len(raw),
        "deduplicated_rows": len(rows),
        "rule": "seeded_permutation_after_normalized_text_deduplication",
    },
)

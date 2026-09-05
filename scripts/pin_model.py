"""Resolve one model revision into a NEW config without downloading its weights."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("template")
parser.add_argument("output")
args = parser.parse_args()
config = json.loads(Path(args.template).read_text())
config["revision"] = HfApi().model_info(config["model_id"]).sha
with Path(args.output).open("x") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
print(f"Pinned {config['model_id']} at {config['revision']}")

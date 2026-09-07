"""Run original source verifier and legacy identities with fresh output receipts."""

import argparse
import runpy
import tempfile
from pathlib import Path

from dependent_rollouts.artifacts import provenance, sha256, write_json
from dependent_rollouts.toy import run_exact

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
args = parser.parse_args()
source = "scripts/archive/verify_reward_coupling_v4.py"
module = runpy.run_path(source)
module["run"]()
with tempfile.TemporaryDirectory() as directory:
    legacy = run_exact(Path(directory) / "legacy.json")
write_json(
    args.output,
    {
        "status": "CPU_ONLY",
        "v4": module["REPORT"],
        "legacy": legacy,
        "verifier_sha256": sha256(source),
        "provenance": provenance(),
    },
)
print(args.output)

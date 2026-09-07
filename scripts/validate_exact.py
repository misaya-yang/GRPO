"""Run original source verifier and legacy identities with fresh output receipts."""

import argparse
import contextlib
import io
import json
import runpy
import shutil
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
    v51_source = Path("docs/theory/v5_1_original/verify_v5_1.py")
    v51_copy = Path(directory) / "verify_v5_1.py"
    shutil.copyfile(v51_source, v51_copy)
    with contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(v51_copy), run_name="__main__")
    v51 = json.loads((Path(directory) / "verify_v5_1_results.json").read_text())
write_json(
    args.output,
    {
        "status": "CPU_ONLY",
        "v4": module["REPORT"],
        "legacy": legacy,
        "v5_1": v51,
        "v5_1_verifier_sha256": sha256(v51_source),
        "verifier_sha256": sha256(source),
        "provenance": provenance(),
    },
)
print(args.output)

"""Budget supervisor: pass an explicit stage after --; never starts a matrix itself."""

import argparse
import json

from reward_coupling.budget import run_budgeted

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ledger", required=True)
parser.add_argument("--cap-seconds", type=float, required=True)
parser.add_argument("--deadline-epoch", type=float)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
result = run_budgeted(command, args.ledger, args.cap_seconds, args.deadline_epoch)
print(json.dumps(result))
raise SystemExit(result["returncode"])

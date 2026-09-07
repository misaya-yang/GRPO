"""Feedback mainline stages. Preparation does not launch a GPU job."""

import argparse
import json
from pathlib import Path

from dependent_rollouts.artifacts import jsonable, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("preflight")
    p.add_argument("--config", required=True)
    p.add_argument("--tasks")
    p.add_argument("--output", required=True)
    p = commands.add_parser("collect")
    p.add_argument("--config", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--split", choices=["Dev", "C", "D", "evaluation"], required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("score")
    p.add_argument("--bank", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--verdicts")
    p.add_argument("--output", required=True)
    p = commands.add_parser("integrate")
    p.add_argument("--bank", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("audit")
    p.add_argument("--bank", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--readout", action="append", default=[])
    p.add_argument("--output", required=True)
    p = commands.add_parser("freeze-readout")
    p.add_argument("--config", required=True)
    p.add_argument("--definitions", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("branch")
    p.add_argument("--shared", required=True)
    p.add_argument("--independent", required=True)
    p.add_argument("--evaluation-bank", required=True)
    p.add_argument("--step", type=float, required=True)
    p.add_argument("--readout", action="append", default=[])
    p.add_argument("--output", required=True)
    p = commands.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--decision", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        from .contract import preflight

        result = preflight(json.loads(Path(args.config).read_text()), args.tasks)
        write_json(args.output, result)
    elif args.command == "collect":
        from .sample import collect

        result = collect(args.config, args.tasks, args.output, args.split)
    elif args.command == "score":
        from .feedback import score

        result = score(args.bank, args.tasks, args.output, args.verdicts)
    elif args.command == "integrate":
        from .feedback import integrate

        result = integrate(args.bank, args.output)
    elif args.command == "audit":
        from .gradient_audit import audit

        result = audit(args.bank, args.output, args.arm, args.readout)
    elif args.command == "freeze-readout":
        from .contract import read_config
        from .gradient_audit import freeze_readout

        result = freeze_readout(read_config(args.config), args.definitions, args.output)
    elif args.command == "branch":
        from .local_steps import branch

        result = branch(
            args.shared,
            args.independent,
            args.evaluation_bank,
            args.output,
            args.step,
            args.readout,
        )
    else:
        from .training import train

        result = train(args.config, args.tasks, args.decision, args.output)
    print(json.dumps(jsonable(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

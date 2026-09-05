"""Small command surface, shared by uv and pre-existing Conda environments."""

import argparse
import json

from .artifacts import jsonable, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exact = commands.add_parser("exact", help="Run dossier finite-policy checks")
    exact.add_argument("--output", required=True)
    tasks = commands.add_parser("generate-tasks", help="Generate structural-split arithmetic JSONL")
    tasks.add_argument("--count", type=int, default=24)
    tasks.add_argument(
        "--split",
        choices=["calibration", "discovery", "confirmation", "train", "evaluation"],
        required=True,
    )
    tasks.add_argument("--seed", type=int, default=2027)
    tasks.add_argument("--output", required=True)
    for name in ("collect", "train"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--config", required=True)
        cmd.add_argument("--tasks", required=True)
        cmd.add_argument("--output", required=True)
    audit = commands.add_parser(
        "audit", help="Stored-data W/C/X, preregistered dose mixtures, or IID-only forecast"
    )
    audit.add_argument("--bank", required=True)
    audit.add_argument(
        "--arm", choices=["within", "count", "cross", "forecast", "dose"], required=True
    )
    audit.add_argument("--kind", choices=["rloo", "standardized"], default="rloo")
    audit.add_argument("--epsilon", type=float, default=1e-6)
    audit.add_argument(
        "--lam", type=float, default=None, help="Dose arm only; preregistered 0, 0.5 or 1"
    )
    audit.add_argument("--output", required=True)
    branch = commands.add_parser(
        "branch", help="Full/half SGD step, finite differences and IID IS reward"
    )
    branch.add_argument("--gradient", required=True)
    branch.add_argument("--evaluation-bank", required=True)
    branch.add_argument("--step", type=float, required=True)
    branch.add_argument("--fd-step", type=float, default=1e-3)
    branch.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "exact":
        from .toy import run_exact

        result = run_exact(args.output)
    elif args.command == "generate-tasks":
        from .tasks import generate_arithmetic

        rows = generate_arithmetic(args.count, args.split, args.seed)
        write_jsonl(args.output, rows)
        result = {"rows": len(rows), "output": args.output}
    elif args.command == "collect":
        from .llm import collect

        result = collect(args.config, args.tasks, args.output)
    elif args.command == "audit":
        from .llm import audit

        result = audit(args.bank, args.output, args.arm, args.kind, args.epsilon, args.lam)
    elif args.command == "branch":
        from .interventions import branch

        result = branch(args.gradient, args.evaluation_bank, args.output, args.step, args.fd_step)
    else:
        from .training import train

        result = train(args.config, args.tasks, args.output)
    print(json.dumps(jsonable(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

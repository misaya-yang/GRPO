"""Runnable v5.1 stages. All output directories are new and sealed."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "calibrate"):
        p = commands.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--tasks", required=True)
        p.add_argument("--output", required=True)
    p = commands.add_parser("analyze")
    p.add_argument("directory")
    p = commands.add_parser("freeze-dev")
    p.add_argument("directory")
    p.add_argument("--output", required=True)
    p = commands.add_parser("online")
    p.add_argument("--config", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("warmup")
    p.add_argument("--config", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        from .pipeline import run

        result = run(args.config, args.tasks, args.output)
    elif args.command == "calibrate":
        from .calibration import calibrate

        result = calibrate(args.config, args.tasks, args.output)
    elif args.command == "analyze":
        from .pipeline import analyze

        result = analyze(args.directory)
    elif args.command == "freeze-dev":
        from .pipeline import freeze_dev

        result = freeze_dev(args.directory, args.output)
    elif args.command == "warmup":
        from .online import run_common_warmup

        result = run_common_warmup(args.config, args.tasks, args.output)
    else:
        from .online import run_online

        result = run_online(args.config, args.tasks, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if isinstance(result, dict) and result.get("status") in (
        "NUMERIC_CONTRACT_FAILED",
        "NO_LORA_SCORE_SIGNAL",
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""One-button CPU preflight / detached bounded GPU cycle / readable status."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dependent_rollouts.artifacts import provenance, sha256, write_json
from reward_coupling.budget import run_budgeted
from stratified_grpo.contract import selected_tasks, validate
from stratified_grpo.model import verify_assets

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "runs/v5_1_active.json"
LEDGER = ROOT / "runs/v5_1_budget.jsonl"


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prior_budget():
    return json.loads((ROOT / "reports/v5_1_prepare/budget_inventory.json").read_text())


def used_seconds():
    return (
        sum(json.loads(s).get("elapsed_seconds", 0) for s in LEDGER.read_text().splitlines())
        if LEDGER.exists()
        else 0.0
    )


def preflight(full_assets=True):
    import torch

    checks = {}
    asset_checks = {}
    for stage, split in (("dev", "calibration"), ("screen", "confirmation")):
        config = validate(json.loads((ROOT / f"configs/v5_1/{stage}.json").read_text()))
        tasks = selected_tasks(config, ROOT / f"data/v5_1/svamp/{split}.jsonl")
        checks[stage] = {
            "prompts": len(tasks),
            "macros_per_arm": len(tasks) * config["macro_repeats"],
            "responses": len(tasks) * config["macro_repeats"] * config["B"] * config["m"] * 2,
        }
        if full_assets:
            key = (config["model_path"], config["checkpoint_manifest_sha256"])
            if key not in asset_checks:
                asset_checks[key] = verify_assets(config)
            checks[stage]["assets"] = asset_checks[key]
    old = ROOT / "runs/C8_difference_2h_20260907/gradient.safetensors"
    checks["old_direction_present"] = old.is_file()
    checks["free_bytes"] = shutil.disk_usage(ROOT).free
    checks["cuda_available"] = torch.cuda.is_available()
    checks["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    checks["prior_budget"] = prior_budget()
    checks["new_cycle_charged_seconds"] = used_seconds()
    checks["provenance"] = provenance()
    return checks


def status():
    if not ACTIVE.exists():
        return {"status": "NOT_STARTED", "message": "Run start only after GPU mode is enabled"}
    data = json.loads(ACTIVE.read_text())
    run = Path(data["run"])
    state = run / "state.json"
    if state.exists():
        data["state"] = json.loads(state.read_text())
    try:
        os.kill(data["pid"], 0)
        data["process_alive"] = True
    except ProcessLookupError:
        data["process_alive"] = False
    for stage in ("numeric", "dev", "screen"):
        progress = run / stage / "progress.jsonl"
        if progress.exists():
            lines = progress.read_text().splitlines()
            if lines:
                last = json.loads(lines[-1])
                data[stage] = {
                    "complete_macros": len(lines),
                    "last_prompt": last["prompt_id"],
                    "last_arm": last["arm"],
                }
    data["charged_seconds_this_version"] = used_seconds()
    return data


def execute_stage(run, name, command, deadline, seconds):
    atomic(
        run / "state.json",
        {"status": "RUNNING", "stage": name, "updated_utc": datetime.now(UTC).isoformat()},
    )
    timeout = min(seconds, deadline - time.time())
    if timeout <= 0:
        raise TimeoutError("Cycle budget exhausted")
    with (run / (name + ".log")).open("x") as log:
        # Child inherits the supervisor process group; outer budget kills the group.
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise TimeoutError(f"{name} exceeded {timeout:.0f}s") from None
    if code:
        raise RuntimeError(f"{name} failed ({code}); see {name}.log")


def workflow(run, seconds, skip_v4=False, source_run=None):
    from stratified_grpo.pipeline import freeze_dev

    run = Path(run)
    deadline = time.time() + seconds
    python = sys.executable
    try:
        checks = preflight()
        write_json(run / "preflight.json", checks)
        if not checks["cuda_available"]:
            raise RuntimeError("GPU is unavailable; switch server to GPU mode before start")
        if checks["free_bytes"] < 6 * 1024**3:
            raise RuntimeError(
                "Less than 6 GiB free; preserve assets and resolve storage before collection"
            )
        if source_run is not None:
            source = Path(source_run).resolve()
            state = json.loads((source / "state.json").read_text())
            if state.get("status") != "READY_NEEDS_COSTED_WINDOW" or (source / "screen").exists():
                raise ValueError("Only an unstarted, costed confirmation can resume automatically")
            config = json.loads((source / "screen_config.json").read_text())
            forecast = json.loads((source / "screen_forecast.json").read_text())[
                "estimated_seconds_with_25pct_margin"
            ]
            if forecast > deadline - time.time() - 60:
                raise ValueError("Requested window cannot fit the already measured screen forecast")
            config["stage_max_seconds"] = max(1, int(deadline - time.time() - 60))
            write_json(run / "screen_config.json", config)
            write_json(
                run / "source_cycle.json",
                {"path": str(source), "config_sha256": sha256(source / "screen_config.json")},
            )
            execute_stage(
                run,
                "screen",
                [
                    python,
                    "-m",
                    "stratified_grpo.cli",
                    "run",
                    "--config",
                    str(run / "screen_config.json"),
                    "--tasks",
                    "data/v5_1/svamp/confirmation.jsonl",
                    "--output",
                    str(run / "screen"),
                ],
                deadline,
                deadline - time.time() - 30,
            )
            analysis = json.loads((run / "screen/analysis.json").read_text())
            atomic(
                run / "state.json",
                {
                    "status": "COMPLETE",
                    "stage": "screen",
                    "decision": analysis["decision"],
                    "online_started": False,
                },
            )
            return
        if not skip_v4:
            if not checks["old_direction_present"]:
                write_json(
                    run / "v4_unavailable.json",
                    {"status": "OLD_DIRECTION_MISSING", "resampling": False},
                )
            else:
                execute_stage(
                    run,
                    "v4_projection",
                    [
                        python,
                        "scripts/repair_v4_diagnostics.py",
                        "repair-projection",
                        "--bank",
                        "runs/D4_scored_2h_20260907",
                        "--direction",
                        "runs/C8_difference_2h_20260907",
                        "--projection-rows",
                        "reports/cd_prefix_2h/direction/D_target_rows.partial.jsonl",
                        "--output",
                        str(run / "v4_projection"),
                    ],
                    deadline,
                    1200,
                )
                execute_stage(
                    run,
                    "v4_local",
                    [
                        python,
                        "scripts/repair_v4_diagnostics.py",
                        "local-diagnostic",
                        "--config",
                        "reports/cd_prefix_2h/config.json",
                        "--difference",
                        "runs/C8_difference_2h_20260907",
                        "--functions",
                        "reports/dev_first2/step_calibration/functions.json",
                        "--h",
                        "0.000005",
                        "0.0000025",
                        "--output",
                        str(run / "v4_local"),
                    ],
                    deadline,
                    1200,
                )
        execute_stage(
            run,
            "numeric",
            [
                python,
                "-m",
                "stratified_grpo.cli",
                "calibrate",
                "--config",
                "configs/v5_1/dev.json",
                "--tasks",
                "data/v5_1/svamp/calibration.jsonl",
                "--output",
                str(run / "numeric"),
            ],
            deadline,
            900,
        )
        execute_stage(
            run,
            "dev",
            [
                python,
                "-m",
                "stratified_grpo.cli",
                "run",
                "--config",
                str(run / "numeric/calibrated_config.json"),
                "--tasks",
                "data/v5_1/svamp/calibration.jsonl",
                "--output",
                str(run / "dev"),
            ],
            deadline,
            1800,
        )
        freeze = freeze_dev(run / "dev", run / "dev_freeze.json")
        config = json.loads((ROOT / "configs/v5_1/screen.json").read_text())
        calibrated = json.loads((run / "numeric/calibrated_config.json").read_text())
        for key in ("token_logp_tolerance", "sequence_logp_tolerance"):
            config[key] = calibrated[key]
        config["dev_freeze"] = str(run / "dev_freeze.json")
        config["dev_freeze_sha256"] = sha256(run / "dev_freeze.json")
        count = len(config["prompt_ids"]) * config["macro_repeats"] * 2
        forecast = freeze["model_load_seconds"] + 1.25 * freeze["seconds_per_macro"] * count
        remaining = deadline - time.time()
        write_json(
            run / "screen_forecast.json",
            {
                "estimated_seconds_with_25pct_margin": forecast,
                "remaining_cycle_seconds": remaining,
                "macros": count,
                "method": "measured Dev macro cost; not a hardware throughput promise",
            },
        )
        config["stage_max_seconds"] = max(1, int(remaining - 60))
        write_json(run / "screen_config.json", config)
        if forecast > remaining - 60:
            atomic(
                run / "state.json",
                {
                    "status": "READY_NEEDS_COSTED_WINDOW",
                    "stage": "screen_not_started",
                    "forecast_seconds": forecast,
                    "remaining_seconds": remaining,
                    "message": "No partial confirmation started. Use measured forecast to allocate one complete window.",
                },
            )
            return
        execute_stage(
            run,
            "screen",
            [
                python,
                "-m",
                "stratified_grpo.cli",
                "run",
                "--config",
                str(run / "screen_config.json"),
                "--tasks",
                "data/v5_1/svamp/confirmation.jsonl",
                "--output",
                str(run / "screen"),
            ],
            deadline,
            remaining - 30,
        )
        analysis = json.loads((run / "screen/analysis.json").read_text())
        atomic(
            run / "state.json",
            {
                "status": "COMPLETE",
                "stage": "screen",
                "decision": analysis["decision"],
                "online_started": False,
                "message": "Stage E is separate; review screen REPORT.md and final handoff report.",
            },
        )
    except BaseException as error:
        atomic(
            run / "state.json",
            {"status": "NEEDS_ATTENTION", "error": repr(error), "automatic_retry": False},
        )
        raise
    finally:
        report_cycle(run)


def report_cycle(run):
    run = Path(run)
    state = (
        json.loads((run / "state.json").read_text())
        if (run / "state.json").exists()
        else {"status": "UNKNOWN"}
    )
    text = ["# 实验周期交接报告", "", f"状态：{state['status']}。", ""]
    for name in ("v4_projection", "v4_local", "numeric", "dev", "screen"):
        p = run / name / "receipt.json"
        if p.exists():
            r = json.loads(p.read_text())
            text.append(f"- {name}: {r.get('status')}，回执 `{name}/receipt.json`。")
        elif (run / (name + ".log")).exists():
            text.append(f"- {name}: 未形成完成回执，检查 `{name}.log`。")
        else:
            text.append(f"- {name}: 未运行。")
    text.extend(
        [
            "",
            "Luna 最终报告须解释已有结果的证据边界、数值/成本失败原因及是否有值得继续的收益；不可把 CPU 例子或 v4 修复点估计当作 v5/在线成功。",
            "",
            json.dumps(state, ensure_ascii=False, indent=2),
        ]
    )
    (run / "FINAL_REPORT.md").write_text("\n".join(text) + "\n")


def start(seconds, skip_v4, source_run=None):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected. Preparation is complete; enable GPU mode first.")
    if seconds < 60:
        raise ValueError("Cycle seconds must be >=60")
    ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    with (ACTIVE.parent / "v5_1_launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = status()
        if old.get("process_alive"):
            raise RuntimeError("An existing v5.1 supervisor is still alive; inspect status")
        cap = prior_budget()["remaining_pilot_seconds_before_v5_1"]
        seconds = min(seconds, cap - used_seconds())
        if seconds <= 0:
            raise TimeoutError("Carried project pilot budget exhausted")
        run = ROOT / "runs" / ("v5_1_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
        run.mkdir(exist_ok=False)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_supervise",
            "--run",
            str(run),
            "--seconds",
            str(int(seconds)),
        ]
        if skip_v4:
            command.append("--skip-v4")
        if source_run is not None:
            command.extend(["--source-run", str(Path(source_run).resolve())])
        with (run / "supervisor.log").open("x") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        data = {
            "pid": process.pid,
            "run": str(run),
            "cycle_seconds": seconds,
            "started_utc": datetime.now(UTC).isoformat(),
        }
        atomic(ACTIVE, data)
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "preflight",
            "start",
            "start-screen",
            "status",
            "report",
            "_supervise",
            "_workflow",
        ],
    )
    parser.add_argument("--seconds", type=int, default=7200)
    parser.add_argument("--run")
    parser.add_argument("--source-run")
    parser.add_argument("--skip-v4", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == "preflight":
        result = preflight()
    elif args.command == "start":
        result = start(args.seconds, args.skip_v4)
    elif args.command == "start-screen":
        if not args.run:
            raise ValueError("start-screen requires --run pointing to the prepared prior cycle")
        result = start(args.seconds, True, args.run)
    elif args.command == "status":
        result = status()
    elif args.command == "report":
        report_cycle(args.run or json.loads(ACTIVE.read_text())["run"])
        result = {"status": "REPORT_WRITTEN"}
    elif args.command == "_workflow":
        workflow(args.run, args.seconds, args.skip_v4, args.source_run)
        result = {"status": "WORKFLOW_ENDED"}
    else:
        run = Path(args.run)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_workflow",
            "--run",
            str(run),
            "--seconds",
            str(args.seconds),
        ]
        if args.skip_v4:
            cmd.append("--skip-v4")
        if args.source_run:
            cmd.extend(["--source-run", args.source_run])
        result = run_budgeted(
            cmd,
            LEDGER,
            prior_budget()["remaining_pilot_seconds_before_v5_1"],
            time.time() + args.seconds,
        )
        if result["returncode"]:
            atomic(
                run / "state.json",
                {"status": "NEEDS_ATTENTION", "budget_result": result, "automatic_retry": False},
            )
        report_cycle(run)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
